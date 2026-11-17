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

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from torch import nn

from msmodelslim.core.quantization_schema import AutoFakeQuantDynamicCache
from msmodelslim.utils.logging import get_logger

from .base_observer import QuantStatisticsObserver
from ..utils import BASELINE_BITS, format_bit_set, logical_bits
from ..summary_context import SUMMARY_LOGGER_NAME, SummaryContext, iter_sections, ranked_key

logger = get_logger(SUMMARY_LOGGER_NAME)

_BRANCH_BY_SUFFIX = {
    "key_states_quantizer": "key_states",
    "value_states_quantizer": "value_states",
}
BRANCHES = ("key_states", "value_states")


@dataclass(frozen=True)
class KVStats:
    """KV Cache 维度的聚合结果。"""

    saving: float
    widths: Tuple[int, ...]

    @property
    def ratio(self) -> float:
        return 1.0 - self.saving


class KVQuantStatisticsObserver(QuantStatisticsObserver):
    """KV Cache 量化统计：层处理完成后扫描量化 IR，读取 K/V 分支位宽。

    独占布局 ``container["kv@rank_N"][layer][attention]``；仅统计实际部署
    ``key_states_quantizer`` / ``value_states_quantizer`` 的分支，展示位宽集合
    不含未量化分支；压缩率以 BF16 为基线、缺失分支按基线计入分母，聚合口径
    与 FA 一致。
    """

    key = "kv"

    def record(self, context: SummaryContext, layer_name: str, module: nn.Module) -> None:
        state = context.state
        if state is None or not layer_name:
            return
        try:
            entries = self._snapshot(module)
            self._write(state, layer_name, entries)
        except Exception as error:
            logger.warning("Failed to record KV cache compression for %s: %s", layer_name, error)

    def aggregate(self, context: SummaryContext) -> Optional[KVStats]:
        layer_states: Dict[str, Dict[str, Dict[str, int]]] = {}
        for section in iter_sections(context.state, self.key):
            for layer_name, entries in section.items():
                if not isinstance(entries, dict):
                    continue
                attentions = layer_states.setdefault(layer_name, {})
                for attention, branches in entries.items():
                    if isinstance(branches, dict):
                        attentions.setdefault(attention, {}).update(branches)

        if not layer_states:
            return None

        saving_sum = 0.0
        widths = set()
        for attentions in layer_states.values():
            per_attention = []
            for branches in attentions.values():
                widths.update(branches[branch] for branch in BRANCHES if branch in branches)
                branch_bits = [branches.get(branch, BASELINE_BITS) for branch in BRANCHES]
                per_attention.append(sum(1.0 - bits / BASELINE_BITS for bits in branch_bits) / len(BRANCHES))
            if per_attention:
                saving_sum += sum(per_attention) / len(per_attention)
        return KVStats(saving=saving_sum / len(layer_states), widths=tuple(sorted(widths)))

    def format(self, result: KVStats) -> str:
        return "[KVCache] quantized to %s, compression ratio %.2f%% (original activations assumed BF16)" % (
            format_bit_set(result.widths),
            result.ratio * 100,
        )

    @staticmethod
    def _snapshot(module: nn.Module) -> Dict[str, Dict[str, int]]:
        entries: Dict[str, Dict[str, int]] = {}
        for name, submodule in module.named_modules():
            if not isinstance(submodule, AutoFakeQuantDynamicCache):
                continue
            branch = _BRANCH_BY_SUFFIX.get(name.rsplit(".", 1)[-1])
            if branch is None:
                continue
            dtype = getattr(getattr(submodule, "x_q_scheme", None), "dtype", None)
            bits = logical_bits(dtype)
            if bits is None:
                logger.debug("KV branch %s has no logical bit width (%s), skipped", name, dtype)
                continue
            attention = name.rsplit(".", 1)[0] if "." in name else ""
            entries.setdefault(attention, {})[branch] = bits
        return entries

    def _write(self, state, layer_name: str, entries: Dict[str, Dict[str, int]]) -> None:
        if not entries:
            return
        container = state
        key = ranked_key(self.key)
        section = container.get(key)
        section = dict(section) if isinstance(section, dict) else {}
        section[layer_name] = {
            attention: {branch: int(bits) for branch, bits in branches.items()}
            for attention, branches in entries.items()
        }
        container[key] = section
