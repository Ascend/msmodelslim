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
from typing import List

from typing_extensions import Self

from msmodelslim.processor.base import AutoProcessorConfig


class PipelineBuilderInfra(ABC):
    """建造者接口：链式设置模板参数，create() 渲染并返回 Processor 配置列表。"""

    @abstractmethod
    def template_modules(self, modules: List[str]) -> Self:
        """设置模板占位符 ``patterns`` / ``quant_modules`` 共用的模块列表，返回 self 以链式调用。"""
        raise NotImplementedError

    @abstractmethod
    def create(self) -> List[AutoProcessorConfig]:
        """渲染模板并返回 List[AutoProcessorConfig]。"""
        raise NotImplementedError


class AnalysisPipelineLoaderInfra(ABC):
    """分析流水线加载的抽象接口：按 metrics 返回建造者。"""

    @abstractmethod
    def get_pipeline_builder(self, metrics: str) -> PipelineBuilderInfra:
        """
        返回用于构建该 metrics 对应流水线配置的建造者。
        建造者已绑定对应模板，调用 template_modules() 后 create() 得到配置列表。
        """
        raise NotImplementedError
