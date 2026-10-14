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

from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest


class BaseProcessor:
    """
    处理器的基类，用于处理模型推理过程中的特定事件。

    处理器负责在模型推理的不同阶段执行特定的操作，如数据预处理、后处理等。
    所有具体的处理器实现都应该继承此类并实现抽象方法。

    """

    def __init__(self, model: nn.Module):
        self.model = model

    def process(self, request: BatchProcessRequest) -> None:
        """
        处理请求并返回响应。

        该方法根据请求中的事件类型、目标模块和相关数据，执行相应的处理操作，
        并返回处理结果和控制选项，指导后续处理流程。

        参数:
            request: 处理请求，包含事件类型、目标模块和相关数据
        """
        self._run_forward_if_need(request)

    def pre_run(self) -> None:
        """
        Runner进行调度前，执行的操作。
        """
        pass

    def post_run(self) -> None:
        """
        Runner结束调度后，执行的操作。
        """
        pass

    def preprocess(self, request: BatchProcessRequest) -> None:
        """
        在Runner调度该Processor的process前，执行的操作。
        """
        pass

    def postprocess(self, request: BatchProcessRequest) -> None:
        """
        在Runner调度该Processor的process后，执行的操作。
        """
        pass

    def _run_forward_if_need(self, request: BatchProcessRequest) -> None:
        _ = self
        outputs = []
        for data in request.datas:
            args, kwargs = data
            if args or kwargs:
                output = request.module(*args, **kwargs)
                outputs.append(output)
        request.outputs = outputs
