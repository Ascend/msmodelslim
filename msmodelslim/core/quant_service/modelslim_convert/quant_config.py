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

modelslim_convert 量化任务配置（apiversion + spec）。
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from typing_extensions import Self

from msmodelslim.core.quant_service.interface import BaseQuantConfig
from .config_mapper import ModelslimConvertServiceConfig


class ModelslimConvertQuantConfig(BaseQuantConfig):
    """`modelslim_convert` 量化（权重转换）任务配置，位于 YAML 根节点。

    `apiversion` 取值固定为 `modelslim_convert`；`spec` 声明权重名重命名/变换（`preprocess`）、
    线性层转换规则（`linears`）与保存格式（`save`）。该配置可省略 `--model_type`，
    由 `msmodelslim quant --config_path` 加载执行纯权重转换。
    """

    apiversion: Literal["modelslim_convert"] = Field(
        default="modelslim_convert",
        description="权重转换任务类型（协议版本），固定为 `modelslim_convert`；框架据此选择权重转换任务配置插件。",
    )
    spec: ModelslimConvertServiceConfig = Field(
        description="`modelslim_convert` 服务的 spec 结构，声明权重名重命名/变换（`preprocess`）、"
        "线性层转换规则（`linears`）、保存格式（`save`）、并行执行（`parallel`）与默认值（`defaults`）。"
    )

    @classmethod
    def from_base(cls, quant_config: BaseQuantConfig) -> Self:
        return cls.model_validate({'apiversion': quant_config.apiversion, 'spec': quant_config.spec})


def get_plugin():
    """获取 modelslim_convert 量化任务配置插件（由框架完成注册）。

    Returns:
        (ModelslimConvertQuantConfig, ModelslimConvertQuantConfig) 元组
        ——组件槽暂与配置槽同体，预留给后续演进。
    """
    return ModelslimConvertQuantConfig, ModelslimConvertQuantConfig
