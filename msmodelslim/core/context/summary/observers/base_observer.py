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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Optional

if TYPE_CHECKING:
    from ..summary_context import SummaryContext


class QuantStatisticsObserver(ABC):
    """统计维度接口：实现负责单维度的采集、聚合与格式化。

    实现独占容器顶层键 ``key`` 的段内布局，采集入口统一命名 ``record``，参数由
    各维度自行定义。``context`` 由调用方按次传入，观测器自身不持有；``aggregate``
    返回该维度聚合结果，无数据时返回 ``None``；``format`` 将聚合结果渲染为一行
    日志，二者由 ``QuantSummary`` 多态调用。
    """

    key: ClassVar[str]

    @abstractmethod
    def aggregate(self, context: "SummaryContext") -> Optional[Any]: ...

    @abstractmethod
    def format(self, result: Any) -> str: ...
