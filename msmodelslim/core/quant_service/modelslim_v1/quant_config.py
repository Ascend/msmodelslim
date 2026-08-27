#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

from typing import List, Literal, Optional, Annotated
from pydantic import BaseModel, Field, AfterValidator
from typing_extensions import Self

from msmodelslim.core.const import RunnerType
from msmodelslim.format.registry import QuantFormatConfigList
from msmodelslim.processor.base import AutoProcessorConfigList
from msmodelslim.utils.validation.pydantic import validate_str_length
from ..interface import BaseQuantConfig


class PriorStageConfig(BaseModel):
    """前置阶段配置：仅 process + dataset，用于如 adapt_rotation stage1 等先验阶段。"""

    process: AutoProcessorConfigList = Field(default_factory=list, description="该阶段处理器列表")
    dataset: Optional[Annotated[str, AfterValidator(validate_str_length(max_len=4096))]] = Field(
        default=None, description="该阶段数据集名称，不提供则使用 spec.dataset"
    )


class ModelslimV1ServiceConfig(BaseModel):
    """`modelslim_v1` 服务的 spec 结构，声明量化流水线、保存格式与校准数据。"""

    runner: RunnerType = Field(
        default=RunnerType.AUTO,
        description="流水线执行方式：`auto` 按设备数量自动选择（单设备 `layer_wise`，多设备 `dp_layer_wise`）、"
        "`model_wise` 整模型计算、`layer_wise` 逐层计算、`dp_layer_wise` 数据并行逐层计算。",
    )
    prior: List[PriorStageConfig] = Field(default_factory=list, description="前置阶段列表，每阶段含 process 与 dataset")
    process: AutoProcessorConfigList = Field(
        default_factory=list,
        description="量化处理器链，按顺序执行；每个元素是 `type` 分派的处理器配置，如 `linear_quant`、`awq`、`smooth_quant` 等。",
    )
    save: QuantFormatConfigList = Field(
        default_factory=list,
        description="保存格式列表，每个元素是 `type` 分派的保存格式配置，如 `ascendv1_saver`、`compressed_tensors`、`mindie_format_saver`。",
    )
    dataset: Annotated[str, AfterValidator(validate_str_length(max_len=4096))] = Field(
        default='mix_calib.jsonl',
        description="校准数据集名称（`lab_calib` 下的文件名）或数据集路径，用于量化的参数估计与敏感性校准。",
    )


class ModelslimV1QuantConfig(BaseQuantConfig):
    """`modelslim_v1` 量化任务配置，位于 YAML 根节点。

    `apiversion` 取值固定为 `modelslim_v1`；`spec` 声明量化流水线（`process`）、
    保存格式（`save`）与校准数据（`dataset`），由 `msmodelslim quant --config_path` 加载。
    """

    apiversion: Literal["modelslim_v1"] = "modelslim_v1"  # 注册表推导 plugin_type 的依据
    spec: ModelslimV1ServiceConfig  # quantization config specification

    @classmethod
    def from_base(cls, quant_config: BaseQuantConfig) -> Self:
        return cls.model_validate({'apiversion': quant_config.apiversion, 'spec': quant_config.spec})


def get_plugin():
    """获取 modelslim_v1 量化任务配置插件（返回配置类与组件类，由框架完成注册）。

    Returns:
        (ModelslimV1QuantConfig, ModelslimV1QuantConfig) 元组——组件槽暂与配置槽同体，
        预留给后续演进（后端任务组件）。
    """
    return ModelslimV1QuantConfig, ModelslimV1QuantConfig


def load_specific_config(yaml_spec: object) -> ModelslimV1ServiceConfig:
    """Load specific configuration from YAML spec"""
    if isinstance(yaml_spec, ModelslimV1ServiceConfig):
        return yaml_spec
    if not isinstance(yaml_spec, dict):
        raise ValueError("task spec must be dict")
    return ModelslimV1ServiceConfig.model_validate(yaml_spec)
