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

# pylint: disable=duplicate-code

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator
from typing_extensions import Self

from msmodelslim.core.const import RunnerType
from msmodelslim.core.quant_service.modelslim_v1.quant_config import ModelslimV1QuantConfig, PriorStageConfig
from msmodelslim.core.quant_service.modelslim_v1.save.saver import AutoSaverConfigList
from msmodelslim.processor.base import AutoProcessorConfigList
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.exception_decorator import exception_handler
from msmodelslim.utils.validation.value import non_empty_string
from ..interface import BaseQuantConfig


class MultimodalVLMServiceConfig(BaseModel):
    """`multimodal_vlm_modelslim_v1` 服务的 spec 结构。

    面向多模态理解（VLM）模型：在 `ModelslimV1ServiceConfig` 基础上增加
    `default_text` 提示词，用于图像类校准数据缺省文本时的默认输入。
    """

    # auto: single device → layer_wise; multi-device (--device npu:0,1,...) → dp_layer_wise
    runner: RunnerType = Field(
        default=RunnerType.AUTO,
        description="流水线执行方式：`auto` 按设备数量自动选择（单设备 `layer_wise`，多设备 `dp_layer_wise`）、"
        "`model_wise` 整模型计算、`layer_wise` 逐层计算、`dp_layer_wise` 数据并行逐层计算。",
    )
    prior: List[PriorStageConfig] = Field(default_factory=list, description="前置阶段列表，每阶段含 process 与 dataset")
    process: AutoProcessorConfigList = Field(
        default_factory=list,
        description="量化处理器链，按顺序执行；每个元素是 `type` 分派的处理器配置。",
    )
    save: AutoSaverConfigList = Field(
        default_factory=list,
        description="保存格式列表，每个元素是 `type` 分派的保存格式配置。",
    )
    dataset: str = Field(
        default='mix_calib.jsonl',
        description="校准数据集名称（`lab_calib` 下的文件名）或数据集路径。",
    )
    default_text: str = Field(
        default="Describe this image in detail.",
        description="校准数据缺少文本模态时，图像类样本使用的默认提示词。",
    )

    @field_validator("default_text")
    @classmethod
    def validate_default_text(cls, v: str) -> str:
        return non_empty_string(v, "default_text")


class MultimodalVLMModelslimV1QuantConfig(ModelslimV1QuantConfig):
    """`multimodal_vlm_modelslim_v1` 量化任务配置，位于 YAML 根节点。

    `apiversion` 取值固定为 `multimodal_vlm_modelslim_v1`；`spec` 声明多模态理解模型的
    量化流水线与校准数据，兼容 `NaiveQuantizationApplication` 与最佳实践系统。
    """

    apiversion: Literal["multimodal_vlm_modelslim_v1"] = "multimodal_vlm_modelslim_v1"
    spec: MultimodalVLMServiceConfig

    @classmethod
    def from_base(cls, quant_config: BaseQuantConfig) -> Self:
        """Convert from base config"""
        return cls.model_validate({'apiversion': quant_config.apiversion, 'spec': quant_config.spec})


def get_plugin():
    """获取 multimodal_vlm_modelslim_v1 量化任务配置插件（由框架完成注册）。

    Returns:
        (MultimodalVLMModelslimV1QuantConfig, MultimodalVLMModelslimV1QuantConfig) 元组
        ——组件槽暂与配置槽同体，预留给后续演进。
    """
    return MultimodalVLMModelslimV1QuantConfig, MultimodalVLMModelslimV1QuantConfig


@exception_handler(
    err_cls=Exception,
    ms_err_cls=SchemaValidateError,
    keyword="validation error",
    action="Please check the spec parameters of the YAML file.",
)
def load_specific_config(yaml_spec: object) -> MultimodalVLMServiceConfig:
    """Load specific configuration from YAML spec"""
    if not isinstance(yaml_spec, dict):
        raise ValueError("task spec must be dict")
    return MultimodalVLMServiceConfig.model_validate(yaml_spec)
