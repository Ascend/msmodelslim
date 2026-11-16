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
from typing import Any, Dict, Optional

from msmodelslim.utils.distributed import is_rank_zero
from msmodelslim.utils.logging import get_logger

from .base_observer import QuantStatisticsObserver
from ..summary_context import SUMMARY_LOGGER_NAME, SummaryContext

logger = get_logger(SUMMARY_LOGGER_NAME)


def format_duration(seconds: float) -> str:
    seconds = float(seconds)
    if seconds >= 3600:
        return "%.2fh" % (seconds / 3600)
    if seconds >= 60:
        return "%.2fmin" % (seconds / 60)
    return "%.2fs" % seconds


@dataclass(frozen=True)
class TimeStats:
    """Time 维度的聚合结果。"""

    layer_count: int
    total_elapsed: float

    @property
    def avg_elapsed(self) -> float:
        if self.layer_count <= 0:
            return 0.0
        return self.total_elapsed / self.layer_count


class TimeQuantStatisticsObserver(QuantStatisticsObserver):
    """逐层耗时统计：记录每层耗时与运行层均，并输出逐层日志。

    独占布局 ``container["time"][layer]``，耗时仅由 rank 0 写入。
    """

    key = "time"

    def record(self, context: SummaryContext, layer_name: str, elapsed: float) -> None:
        state = context.state
        if state is None or not is_rank_zero():
            return
        try:
            elapsed = float(elapsed)
            section = self._read(state)
            total = sum(float(item.get("elapsed", 0.0)) for item in section.values() if isinstance(item, dict))
            avg = (total + elapsed) / (len(section) + 1)
            section[layer_name] = {"elapsed": elapsed, "avg": avg}
            state[self.key] = section
        except Exception as error:
            logger.warning("Failed to record time for %s: %s", layer_name, error)
            return
        logger.info(
            "%s quant done, this layer elapsed %s, overall avg %s",
            layer_name,
            format_duration(elapsed),
            format_duration(avg),
        )

    def aggregate(self, context: SummaryContext) -> Optional[TimeStats]:
        elapsed = {
            layer_name: float(item.get("elapsed", 0.0))
            for layer_name, item in self._read(context.state).items()
            if isinstance(item, dict)
        }
        if not elapsed:
            return None
        return TimeStats(layer_count=len(elapsed), total_elapsed=sum(elapsed.values()))

    def format(self, result: TimeStats) -> str:
        return "[Time] %d layers total %s, avg %s/layer" % (
            result.layer_count,
            format_duration(result.total_elapsed),
            format_duration(result.avg_elapsed),
        )

    def _read(self, state) -> Dict[str, Any]:
        section = state.get(self.key) if state is not None else None
        return dict(section) if isinstance(section, dict) else {}
