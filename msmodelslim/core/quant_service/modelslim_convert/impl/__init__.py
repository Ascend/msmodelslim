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

Convert 各阶段默认实现（由 ``app.convert.factory`` 装配）。

模块对应设计文档阶段：
  PreprocessExecutor §6
  VirtualModelTreeBuilder §8
  DefaultIRTaskBuilder §9
  ConvertExecutor §10
  SaveProcessorAdapter §11
"""

from msmodelslim.core.quant_service.modelslim_convert.impl.executor import ConvertExecutor
from msmodelslim.core.quant_service.modelslim_convert.impl.preprocess import PreprocessExecutor
from msmodelslim.core.quant_service.modelslim_convert.impl.save_adapter import SaveProcessorAdapter
from msmodelslim.core.quant_service.modelslim_convert.impl.task_builder import DefaultIRTaskBuilder
from msmodelslim.core.quant_service.modelslim_convert.impl.virtual_tree import VirtualModelTreeBuilder

__all__ = [
    "ConvertExecutor",
    "PreprocessExecutor",
    "SaveProcessorAdapter",
    "DefaultIRTaskBuilder",
    "VirtualModelTreeBuilder",
]
