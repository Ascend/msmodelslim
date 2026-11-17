#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

from torch import nn

from msmodelslim.core.quantization_schema import (
    AutoFakeQuantActivation,
    AutoFakeQuantDynamicCache,
    AutoFakeQuantLinear,
    WrapperIR,
)
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.memory import format_memory_size

from .base_observer import QuantStatisticsObserver
from ..summary_context import SUMMARY_LOGGER_NAME, SummaryContext, iter_sections, ranked_key
from ..utils import format_bit_set

logger = get_logger(SUMMARY_LOGGER_NAME)

_ATOMIC_IRS = (AutoFakeQuantLinear, AutoFakeQuantActivation, AutoFakeQuantDynamicCache)
_W_BITS = re.compile(r"^W(FP8|\d+)")


def ir_bits(class_name: str) -> Optional[int]:
    match = _W_BITS.match(class_name or "")
    if match is None:
        return None
    width = match.group(1)
    return 8 if width == "FP8" else int(width)


def is_linear_module(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) or ir_bits(type(module).__name__) is not None


def module_bits(module: nn.Module) -> Optional[int]:
    bits = ir_bits(type(module).__name__)
    if bits is not None:
        return bits
    if isinstance(module, nn.Linear):
        weight = getattr(module, "weight", None)
        if weight is not None:
            return int(weight.element_size()) * 8
    return None


def _iter_linears(module: nn.Module) -> Iterator[Tuple[str, nn.Module]]:
    seen = set()

    def walk(current: nn.Module, prefix: str) -> Iterator[Tuple[str, nn.Module]]:
        if id(current) in seen:
            return
        seen.add(id(current))

        if isinstance(current, WrapperIR):
            yield from walk(current.wrapped_module, prefix)
            return

        if is_linear_module(current):
            yield prefix, current
            return

        if isinstance(current, _ATOMIC_IRS) and current.is_atomic():
            return

        for child_name, child in current.named_children():
            child_prefix = "%s.%s" % (prefix, child_name) if prefix else child_name
            yield from walk(child, child_prefix)

    yield from walk(module, "")


def _snapshot(
    module: nn.Module,
    layer_name: str = "",
    source_bits: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict[str, int]]:
    result: Dict[str, Dict[str, int]] = {}
    for prefix, submodule in _iter_linears(module):
        weight = getattr(submodule, "weight", None)
        if weight is None:
            continue
        memory_bits = module_bits(submodule)
        if memory_bits is None:
            logger.debug("Unable to resolve weight bit width for %s, layer skipped", type(submodule).__name__)
            continue
        # 适配器可能把源 fp8 权重反量化成 bf16 再量化，此时原始位宽按源 checkpoint 统计。
        source = (source_bits or {}).get("%s.%s" % (layer_name, prefix), memory_bits)
        result[prefix] = {
            "numel": int(weight.numel()),
            "bits": min(source, memory_bits),
            "quantized": ir_bits(type(submodule).__name__) is not None,
        }
    return result


def _pair(
    layer_name: str,
    pre: Dict[str, Dict[str, int]],
    post: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, int]]:
    entries: Dict[str, Dict[str, int]] = {}
    for prefix, pre_item in pre.items():
        post_item = post.get(prefix)
        if post_item is None:
            logger.debug("Linear module %s.%s disappeared after quantization, skipped", layer_name, prefix)
            continue
        entries[prefix] = {
            "numel": pre_item["numel"],
            "original_bits": pre_item["bits"],
            "quantized_bits": post_item["bits"],
            "quantized": bool(post_item["quantized"]),
        }
    for prefix in post:
        if prefix not in pre:
            logger.debug("Linear module %s.%s appeared after quantization, skipped", layer_name, prefix)
    return entries


@dataclass(frozen=True)
class LinearStats:
    """Linear 维度的聚合结果。"""

    original_bytes: float
    quantized_bytes: float
    widths: Tuple[int, ...] = ()

    @property
    def ratio(self) -> float:
        if self.original_bytes <= 0:
            return 0.0
        return self.quantized_bytes / self.original_bytes


class LinearQuantStatisticsObserver(QuantStatisticsObserver):
    """线性层权重压缩率统计：逐层对比量化前后的 IR，仅统计权重本身。

    ``record`` 在层处理前后各调用一次（``is_before_quant`` 区分），按模块路径
    配对；每个 rank 独占 ``container["linear@rank_N"][layer][prefix]``，聚合时
    合并所有 rank 分片并按 ``(layer, prefix)`` 去重。
    """

    key = "linear"

    def __init__(self) -> None:
        self._pre: Dict[str, Dict[str, Dict[str, int]]] = {}

    def record(
        self,
        context: SummaryContext,
        layer_name: str,
        module: nn.Module,
        is_before_quant: bool,
        source_bits: Optional[Dict[str, int]] = None,
    ) -> None:
        state = context.state
        if state is None:
            return
        if is_before_quant:
            self._record_before(layer_name, module, source_bits)
        else:
            self._record_after(state, layer_name, module)

    def aggregate(self, context: SummaryContext) -> Optional[LinearStats]:
        seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for section in iter_sections(context.state, self.key):
            for layer_name, entries in section.items():
                if not isinstance(entries, dict):
                    continue
                for prefix, payload in entries.items():
                    if isinstance(payload, dict):
                        seen.setdefault((layer_name, prefix), payload)

        if not seen:
            return None

        original = 0.0
        quantized = 0.0
        widths = set()
        for payload in seen.values():
            numel = float(payload.get("numel", 0))
            original += numel * float(payload.get("original_bits", 0)) / 8.0
            quantized += numel * float(payload.get("quantized_bits", 0)) / 8.0
            if payload.get("quantized"):
                widths.add(int(payload.get("quantized_bits", 0)))
        if original <= 0:
            return None
        return LinearStats(original_bytes=original, quantized_bytes=quantized, widths=tuple(sorted(widths)))

    def format(self, result: LinearStats) -> str:
        return "[Linear] quantized to %s, compression ratio %.2f%% (original %s -> quantized %s)" % (
            format_bit_set(result.widths),
            result.ratio * 100,
            format_memory_size(int(round(result.original_bytes))),
            format_memory_size(int(round(result.quantized_bytes))),
        )

    def _record_before(
        self,
        layer_name: str,
        module: nn.Module,
        source_bits: Optional[Dict[str, int]] = None,
    ) -> None:
        try:
            self._pre[layer_name] = _snapshot(module, layer_name, source_bits)
        except Exception as error:
            logger.warning("Failed to snapshot linear layers for %s: %s", layer_name, error)

    def _record_after(self, state, layer_name: str, module: nn.Module) -> None:
        pre = self._pre.pop(layer_name, None)
        if not pre:
            return
        try:
            entries = _pair(layer_name, pre, _snapshot(module))
            self._write(state, layer_name, entries)
        except Exception as error:
            logger.warning("Failed to record linear compression for %s: %s", layer_name, error)

    def _write(self, state, layer_name: str, entries: Dict[str, Dict[str, int]]) -> None:
        if not entries:
            return
        container = state
        key = ranked_key(self.key)
        section = container.get(key)
        section = dict(section) if isinstance(section, dict) else {}
        section[layer_name] = {
            name: {
                "numel": int(item["numel"]),
                "original_bits": int(item["original_bits"]),
                "quantized_bits": int(item["quantized_bits"]),
                "quantized": bool(item["quantized"]),
            }
            for name, item in entries.items()
        }
        container[key] = section
