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
# pylint: disable=logging-fstring-interpolation

from typing import Callable, Optional, Literal

from torch import nn

from msmodelslim.ir.qal import QABCRegistry
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.base import AutoSessionProcessor, AutoProcessorConfig
from msmodelslim.utils.logging import get_logger, logger_setter

ModuleFuncType = Callable[[str, nn.Module], None]  # name, module


class ModuleFuncProcessorConfig(AutoProcessorConfig):
    type: Literal["module_func"] = "module_func"
    name: str
    func: ModuleFuncType


@QABCRegistry.register(dispatch_key=ModuleFuncProcessorConfig, abc_class=AutoSessionProcessor)
@logger_setter(prefix="msmodelslim.processor.module_func")
class ModuleFuncProcessor(AutoSessionProcessor):
    def __init__(
        self,
        model: nn.Module,
        config: ModuleFuncProcessorConfig,
        _: Optional[object] = None,
    ):
        super().__init__(model)
        self.config = config

    def __repr__(self) -> str:
        return f"ModuleFuncProcessor(name={self.config.name})"

    def preprocess(self, request: BatchProcessRequest) -> None:
        get_logger().debug(f"Running module func {self.config.name} on {request.name}")
        self.config.func(request.name, request.module)

    def is_data_free(self) -> bool:
        """
        判断处理器是否需要数据。

        Returns:
            True，因为加载处理器不需要输入数据
        """
        return True
