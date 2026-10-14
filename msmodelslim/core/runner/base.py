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

import abc
from typing import Optional, List, Any

from torch import nn

from msmodelslim.core.const import DeviceType
from msmodelslim.processor import AutoProcessorConfig


class BaseRunner:
    """
    推理调度器的基类，用于管理和执行模型推理过程中的调度操作。

    该类提供了添加处理器、创建前向传播生成器以及执行推理调度的基础接口。
    所有具体的推理调度器实现都应该继承此类并实现抽象方法。
    """

    @abc.abstractmethod
    def add_processor(self, processor_cfg: AutoProcessorConfig, append: bool = True):
        """
        添加处理器以及与当前处理器关联的输入数据。

        该方法用于在推理调度过程中注册处理器，处理器将负责处理特定事件。
        可以为处理器关联特定的输入数据，这些数据将用于创建相应的前向传播生成器。

        参数:
            processor: 处理器实例，用于处理推理调度过程中的特定事件
            input_datas: 与当前处理器关联的输入数据列表，将传递给create_forward_generator创建相应生成器
            append: 控制处理器添加位置，True表示添加到列表尾部，False表示添加到列表头部
        """
        pass

    @abc.abstractmethod
    def run(
        self,
        model: Optional[nn.Module] = None,
        calib_data: Optional[List[Any]] = None,
        device: DeviceType = DeviceType.NPU,
        device_indices: Optional[List[int]] = None,
    ):
        """
        执行推理调度流程。

        该方法根据已添加的处理器和关联的输入数据，生成用于前向传播的生成器，
        并按照预定义的顺序执行推理调度。在调度过程中，会调用各个处理器处理特定事件。

        参数:
            model: 可选的模型实例，如果未提供则会通过adapter初始化
            calib_data: 可选的校准数据，用于需要数据的处理器
            device: 目标设备类型，默认为NPU

        整个调度过程是确定性的，确保推理结果的可重复性。
        """
        pass
