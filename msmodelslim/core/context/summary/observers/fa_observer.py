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
from typing import Dict, Optional, Tuple

from torch import nn

from msmodelslim.ir import AutoFakeQuantActivation
from msmodelslim.utils.logging import get_logger

from .base_observer import QuantStatisticsObserver
from ..utils import BASELINE_BITS, format_bit_set, logical_bits
from ..summary_context import SUMMARY_LOGGER_NAME, SummaryContext, iter_sections, ranked_key

logger = get_logger(SUMMARY_LOGGER_NAME)

_BRANCH = re.compile(r"^fa3?_(q|k|v)$")
BRANCHES = ("fa_q", "fa_k", "fa_v")


@dataclass(frozen=True)
class FAStats:
    """FA 维度的聚合结果。"""

    saving: float
    widths: Tuple[int, ...]

    @property
    def ratio(self) -> float:
        return 1.0 - self.saving


class FAQuantStatisticsObserver(QuantStatisticsObserver):
    """FA 激活量化统计：层处理完成后扫描量化 IR，读取 Q/K/V 分支位宽。

    独占布局 ``container["fa@rank_N"][layer][attention]``；仅统计真实存在的量化
    分支，展示位宽集合不含未量化分支；压缩率以 BF16 为基线、缺失分支按基线
    计入分母，层内 attention 先平均，再对出现 IR 的层取平均。
    """

    key = "fa"

    def record(self, context: SummaryContext, layer_name: str, module: nn.Module) -> None:
        state = context.state
        if state is None or not layer_name:
            return
        try:
            entries = self._snapshot(module)
            self._write(state, layer_name, entries)
        except Exception as error:
            logger.warning("Failed to record FA compression for %s: %s", layer_name, error)

    def aggregate(self, context: SummaryContext) -> Optional[FAStats]:
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
        return FAStats(saving=saving_sum / len(layer_states), widths=tuple(sorted(widths)))

    def format(self, result: FAStats) -> str:
        return "[FA] quantized to %s, compression ratio %.2f%% (original activations assumed BF16)" % (
            format_bit_set(result.widths),
            result.ratio * 100,
        )

    @staticmethod
    def _snapshot(module: nn.Module) -> Dict[str, Dict[str, int]]:
        entries: Dict[str, Dict[str, int]] = {}
        for name, submodule in module.named_modules():
            if not isinstance(submodule, AutoFakeQuantActivation):
                continue
            match = _BRANCH.match(name.rsplit(".", 1)[-1])
            if match is None:
                continue
            dtype = getattr(getattr(submodule, "x_q_scheme", None), "dtype", None)
            bits = logical_bits(dtype)
            if bits is None:
                logger.debug("FA branch %s has no logical bit width (%s), skipped", name, dtype)
                continue
            attention = name.rsplit(".", 1)[0] if "." in name else ""
            entries.setdefault(attention, {})["fa_" + match.group(1)] = bits
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
