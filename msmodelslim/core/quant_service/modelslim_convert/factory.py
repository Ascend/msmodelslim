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

Convert 默认组件装配（convert_design.md §7.2）。

唯一产品入口应通过本 factory 构造 ``ConvertApplication``，保证：
  - CheckpointReader（infra/io）
  - register_convert_processors() 注册的 IR 边
  - impl/* 各阶段实现
"""

from __future__ import annotations

from msmodelslim.core.quant_service.modelslim_convert.application import ConvertApplication
from msmodelslim.core.quant_service.modelslim_convert.impl import (
    ConvertExecutor,
    DefaultIRTaskBuilder,
    PreprocessExecutor,
    SaveProcessorAdapter,
    VirtualModelTreeBuilder,
)
from msmodelslim.infra.io.checkpoint_reader import CheckpointReader
from msmodelslim.processor.convert.registry import register_convert_processors


def create_convert_application() -> ConvertApplication:
    """
    创建可运行的 ConvertApplication。

    ``router`` 同时注入 Application（路由规划）与 ConvertExecutor（边执行），
    避免重复 register 导致边表不一致。
    """
    router = register_convert_processors()
    return ConvertApplication(
        checkpoint_reader_factory=CheckpointReader,
        preprocess_executor=PreprocessExecutor(),
        tree_builder=VirtualModelTreeBuilder(),
        task_builder=DefaultIRTaskBuilder(),
        executor=ConvertExecutor(router=router),
        save_adapter=SaveProcessorAdapter(),
        router=router,
    )
