#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
modelslim_convert 量化任务配置（apiversion + spec）。
"""

from __future__ import annotations

from typing_extensions import Self

from msmodelslim.core.quant_service.interface import BaseQuantConfig
from .config_mapper import ModelslimConvertServiceConfig, load_specific_config


class ModelslimConvertQuantConfig(BaseQuantConfig):
    """`modelslim_convert` 量化（权重转换）任务配置，位于 YAML 根节点。

    `apiversion` 取值固定为 `modelslim_convert`；`spec` 声明权重名重命名/变换（`preprocess`）、
    线性层转换规则（`linears`）与保存格式（`save`）。该配置可省略 `--model_type`，
    由 `msmodelslim quant --config_path` 加载执行纯权重转换。
    """

    spec: ModelslimConvertServiceConfig

    @classmethod
    def from_base(cls, quant_config: BaseQuantConfig) -> Self:
        return cls(
            apiversion=quant_config.apiversion,
            spec=load_specific_config(quant_config.spec),
        )
