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

from typing import Any, Dict, List, Tuple

from msmodelslim.utils.distributed import is_rank_zero
from msmodelslim.utils.logging import get_logger, logger_setter

from .observers import (
    FAQuantStatisticsObserver,
    KVQuantStatisticsObserver,
    LinearQuantStatisticsObserver,
    QuantStatisticsObserver,
    TimeQuantStatisticsObserver,
)
from .summary_context import SUMMARY_LOGGER_NAME, SummaryContext


@logger_setter(prefix=SUMMARY_LOGGER_NAME)
class QuantSummary:
    """量化汇总入口：观测器内置于类中，``finish()`` 聚合并输出各维度结果。

    全局单例 ``quant_summary`` 在导入时创建，采集侧直接调用
    ``quant_summary.<dimension>.record(context, ...)``，把 context 按次传入，
    观测器自身不持有 context。``finish(context)`` 在量化流程结束时聚合并仅在
    rank 0 打印；单个维度异常只告警，不影响量化主流程。
    """

    def __init__(self) -> None:
        self.linear = LinearQuantStatisticsObserver()
        self.fa = FAQuantStatisticsObserver()
        self.kv = KVQuantStatisticsObserver()
        self.time = TimeQuantStatisticsObserver()
        self._observers: Tuple[QuantStatisticsObserver, ...] = (self.linear, self.fa, self.kv, self.time)

    def finish(self, context: Any) -> Dict[str, Any]:
        """聚合 ``context`` 中的统计结果，rank 0 打印并返回各维度聚合值。"""
        summary_context = SummaryContext.from_context(context)
        values: Dict[str, Any] = {}
        lines: List[str] = []

        for observer in self._observers:
            self._collect(values, lines, observer, summary_context)

        if is_rank_zero():
            self._log_result(lines)
        return values

    @staticmethod
    def _collect(
        values: Dict[str, Any],
        lines: List[str],
        observer: QuantStatisticsObserver,
        context: SummaryContext,
    ) -> None:
        key = observer.key
        try:
            value = observer.aggregate(context)
        except Exception as error:
            get_logger().warning("Failed to aggregate %s summary: %s", key, error)
            return
        if value is None:
            return
        values[key] = value
        try:
            lines.append(observer.format(value))
        except Exception as error:
            get_logger().warning("Failed to format %s summary: %s", key, error)

    @staticmethod
    def _log_result(lines: List[str]) -> None:
        if not lines:
            return

        get_logger().info("==================== QUANTIZATION SUMMARY ====================")
        for line in lines:
            get_logger().info(line)
        get_logger().info("===========================================================")


quant_summary = QuantSummary()
