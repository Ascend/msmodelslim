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
from typing import Any, List, Dict, Tuple, Optional

from torch import nn


@dataclass
class DataUnit:
    input: Any
    output: Any


@dataclass
class ProcessRequest:
    """
    处理请求数据类，封装了处理器需要处理的事件信息。

    该类包含了处理事件、目标模块以及相关的数据，用于向处理器传递处理请求。

    属性:
        name: 目标模块的名称
        module: 目标模块，指定需要处理的PyTorch模块
        args: 处理数据列表，包含与处理事件相关的数据
        kwargs: 处理数据列表，包含与处理事件相关的数据
    """

    name: str
    module: nn.Module
    args: Tuple[Any]
    kwargs: Dict[str, Any]


@dataclass
class BatchProcessRequest:
    """
    批量处理请求数据类，封装了处理器需要处理的事件信息。

    该类包含了处理事件、目标模块以及相关的数据，用于向处理器传递处理请求。
    """

    name: str
    module: nn.Module
    datas: Optional[List[Tuple[Tuple[Any], Dict[str, Any]]]] = None
    outputs: Optional[List[Any]] = None
