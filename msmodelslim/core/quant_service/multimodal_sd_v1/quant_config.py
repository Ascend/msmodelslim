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

import logging
from dataclasses import dataclass
from typing import Dict, Any, Union, Optional, List
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, model_validator
from typing_extensions import Self, Literal
from torch import nn

from msmodelslim.core.const import RunnerType
from msmodelslim.core.quant_service.interface import BaseQuantConfig
from msmodelslim.core.quant_service.modelslim_v1.quant_config import ModelslimV1QuantConfig, PriorStageConfig
from msmodelslim.core.quant_service.modelslim_v1.save.saver import AutoSaverConfigList
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.exception_decorator import exception_handler
from msmodelslim.processor.base import AutoProcessorConfigList
from .legacy_pipeline_interface import LegacyMultimodalPipelineInterface
from .pipeline_interface import MultimodalPipelineInterface

MultimodalSDPipelineAdapter = Union[LegacyMultimodalPipelineInterface, MultimodalPipelineInterface]


class DumpConfig(BaseModel):
    enable_dump: bool = Field(default=True)
    capture_mode: Literal["args"] = Field(default="args")
    dump_data_dir: str = Field(default="")


# 多模态基础配置
class MultimodalSDConfig(BaseModel):
    dump_config: DumpConfig
    # 推理参数
    inference_config: Optional[Dict[str, Any]] = Field(default=None)
    # 允许接收未定义的额外参数（迁移期旧字段 model_config 等可落在 model_extra）
    model_config = {"extra": "allow"}

    # 可选：将额外参数转换为字典属性，方便访问
    @property
    def extra_params(self) -> Dict[str, Any]:
        return self.model_extra or {}

    def resolve_inference_raw(self) -> Dict[str, Any]:
        """
        解析 YAML 中的推理配置字典。

        inference_config 为 MultimodalSDConfig 声明字段，由 Pydantic 解析到 self.inference_config
        （不会进入 model_extra）。与迁移期 extra 字段 model_config 互斥；仅 model_config 时告警并回退读取。
        """
        extra = self.model_extra or {}
        has_legacy_model = "model_config" in extra
        if self.inference_config is not None and has_legacy_model:
            raise SchemaValidateError(
                "inference_config and model_config are mutually exclusive; please keep only one.",
            )

        if self.inference_config is not None:
            if not isinstance(self.inference_config, dict):
                raise SchemaValidateError(
                    f"inference_config must be dict, got {type(self.inference_config)}",
                )
            return self.inference_config

        if "model_config" in extra:
            logging.warning(
                "multimodal_sd_config.model_config will be deprecated.",
            )
            raw = extra["model_config"]
            if not isinstance(raw, dict):
                raise SchemaValidateError(f"model_config must be dict, got {type(raw)}")
            return raw

        return {}


class MultimodalSDServiceConfig(BaseModel):
    runner: RunnerType = RunnerType.LAYER_WISE
    prior: List[PriorStageConfig] = Field(default_factory=list, description="前置阶段列表，每阶段含 process 与 dataset")
    process: AutoProcessorConfigList = Field(default_factory=list)
    save: AutoSaverConfigList = Field(default_factory=list)
    dataset: str = Field(default='mix_calib.jsonl')
    # 支持直接传入字典作为配置，或使用 MultimodalSDConfig 实例
    multimodal_sd_config: Union[Dict[str, Any], MultimodalSDConfig] = Field(
        default_factory=lambda: MultimodalSDConfig().model_dump()
    )

    # 验证并转换配置格式
    @model_validator(mode="after")
    def normalize_config(self) -> Self:
        if isinstance(self.multimodal_sd_config, dict):
            # 将字典转换 MultimodalSDConfig 实例（会保留额外字段）
            self.multimodal_sd_config = MultimodalSDConfig(**self.multimodal_sd_config)
        return self


class MultimodalSDModelslimV1QuantConfig(ModelslimV1QuantConfig):
    """支持多模态的量化配置类"""

    spec: MultimodalSDServiceConfig  # 使用新的多模态配置

    @classmethod
    def from_base(cls, quant_config: BaseQuantConfig) -> Self:
        return cls(
            apiversion=quant_config.apiversion,
            spec=load_specific_config(quant_config.spec),
        )


@exception_handler(
    err_cls=Exception,
    ms_err_cls=SchemaValidateError,
    keyword="validation error",
    action="Please check the multimodal_sd_config parameter of the YAML file.",
)
def load_specific_config(yaml_spec: object) -> MultimodalSDServiceConfig:
    """Load specific configuration from YAML spec"""
    if isinstance(yaml_spec, MultimodalSDServiceConfig):
        return yaml_spec
    if not isinstance(yaml_spec, dict):
        raise SchemaValidateError("task spec must be dict")
    return MultimodalSDServiceConfig.model_validate(yaml_spec)


def validate_inference_config(
    model_adapter: MultimodalPipelineInterface,
    sd_config: MultimodalSDConfig,
) -> BaseModel:
    """
    从 multimodal_sd_config 解析推理 dict，并用适配器 InferenceConfig 类做 model_validate。
    """
    raw = sd_config.resolve_inference_raw()
    try:
        return model_adapter.get_inference_config_class().model_validate(raw)
    except ValidationError as e:
        raise SchemaValidateError(
            f"invalid inference_config for {model_adapter.model_type}: {e}",
        ) from e


@dataclass
class MultiExpertQuantConfig:
    """多专家模型量化配置"""

    model_adapter: MultimodalSDPipelineAdapter
    models: dict[str, nn.Module]
    calib_data: dict[str, Any]
    quant_config: MultimodalSDModelslimV1QuantConfig
    save_path: Path
    device: str = "npu"
