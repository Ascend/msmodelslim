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

from typing import List, Optional, Annotated
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
    runner: RunnerType = RunnerType.AUTO
    prior: List[PriorStageConfig] = Field(default_factory=list, description="前置阶段列表，每阶段含 process 与 dataset")
    process: AutoProcessorConfigList = Field(default_factory=list)
    save: QuantFormatConfigList = Field(default_factory=list)
    dataset: Annotated[str, AfterValidator(validate_str_length(max_len=4096))] = Field(
        default='mix_calib.jsonl', description="数据集名称"
    )


class ModelslimV1QuantConfig(BaseQuantConfig):
    spec: ModelslimV1ServiceConfig  # quantization config specification

    @classmethod
    def from_base(cls, quant_config: BaseQuantConfig) -> Self:
        return cls(
            apiversion=quant_config.apiversion,
            spec=load_specific_config(quant_config.spec),
        )


def load_specific_config(yaml_spec: object) -> ModelslimV1ServiceConfig:
    """Load specific configuration from YAML spec"""
    if isinstance(yaml_spec, ModelslimV1ServiceConfig):
        return yaml_spec
    if not isinstance(yaml_spec, dict):
        raise ValueError("task spec must be dict")
    return ModelslimV1ServiceConfig.model_validate(yaml_spec)
