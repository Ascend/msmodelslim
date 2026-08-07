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

from typing import List

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
    # auto: single device → layer_wise; multi-device (--device npu:0,1,...) → dp_layer_wise
    runner: RunnerType = RunnerType.AUTO
    prior: List[PriorStageConfig] = Field(default_factory=list, description="前置阶段列表，每阶段含 process 与 dataset")
    process: AutoProcessorConfigList = Field(default_factory=list)
    save: AutoSaverConfigList = Field(default_factory=list)
    dataset: str = Field(default='mix_calib.jsonl')
    default_text: str = Field(
        default="Describe this image in detail.",
        description="Default prompt used for image-only calibration data when text is not provided.",
    )

    @field_validator("default_text")
    @classmethod
    def validate_default_text(cls, v: str) -> str:
        return non_empty_string(v, "default_text")


class MultimodalVLMModelslimV1QuantConfig(ModelslimV1QuantConfig):
    """
    Quantization configuration for Multimodal VLM V1 service.

    Compatible with NaiveQuantizationApplication and best practice system.
    """

    spec: MultimodalVLMServiceConfig

    @classmethod
    def from_base(cls, quant_config: BaseQuantConfig) -> Self:
        """Convert from base config"""
        return cls(
            apiversion=quant_config.apiversion,
            spec=load_specific_config(quant_config.spec),
        )


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
