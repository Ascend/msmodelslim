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

from abc import ABC, abstractmethod
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, Field, AfterValidator, model_validator

from msmodelslim.core.const import DeviceType
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.utils.validation.pydantic import validate_str_length


class AnalysisScope(str, Enum):
    """与 CLI ``linear`` / ``attn`` / ``layer`` 一致；各 scope 使用不同参数字段。"""

    LINEAR = "linear"
    ATTN = "attn"
    LAYER = "layer"


class AnalysisConfig(BaseModel):
    """分析服务入参：按 scope 区分 linear_pattern / quant_modules（attn 无用户列表）。"""

    scope: AnalysisScope = Field(..., description="分析范围")
    metrics: str = Field(
        ...,
        description="分析指标名，如 std / quantile / kurtosis / mse / mse_model_wise",
    )
    calib_dataset: str = Field(..., description="校准数据集名称，用于前向收集激活")
    linear_pattern: Optional[List[str]] = Field(
        default=None,
        description="仅 linear：线性层名匹配（fnmatch），对应 CLI --pattern。",
    )
    quant_modules: Optional[List[str]] = Field(
        default=None,
        description="仅 layer：块级量化模块范围，对应 CLI --quant_modules。",
    )

    def template_substitute_list(self) -> List[str]:
        """供 YAML 模板占位符替换：linear 用 linear_pattern，layer 用 quant_modules，attn 为 ['*']。"""
        if self.scope == AnalysisScope.LINEAR:
            return list(self.linear_pattern or ["*"])
        if self.scope == AnalysisScope.LAYER:
            return list(self.quant_modules or ["*"])
        return ["*"]

    @model_validator(mode="after")
    def _normalize_scope_fields(self) -> "AnalysisConfig":
        if self.scope == AnalysisScope.LINEAR:
            if self.quant_modules is not None:
                raise ValueError("scope=linear 不应设置 quant_modules")
            lp = self.linear_pattern if self.linear_pattern is not None else ["*"]
            return self.model_copy(update={"linear_pattern": list(lp), "quant_modules": None})
        if self.scope == AnalysisScope.LAYER:
            if self.linear_pattern is not None:
                raise ValueError("scope=layer 不应设置 linear_pattern")
            qm = self.quant_modules if self.quant_modules is not None else ["*"]
            return self.model_copy(update={"quant_modules": list(qm), "linear_pattern": None})
        if self.linear_pattern is not None or self.quant_modules is not None:
            raise ValueError("scope=attn 不应设置 linear_pattern 或 quant_modules")
        return self.model_copy(update={"linear_pattern": None, "quant_modules": None})


class AnalysisResult(BaseModel):
    """分析结果数据：层分数列表及方法、patterns 等元数据。"""

    layer_scores: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of {'name': str, 'score': float}",
    )
    method: str = Field(..., description="分析方法名，如 std / kurtosis")
    patterns: List[Annotated[str, AfterValidator(validate_str_length())]] = Field(
        default_factory=list, description="层匹配模式"
    )


class IAnalysisService(ABC):
    """Abstract base class for model analysis services"""

    @abstractmethod
    def analyze(
        self,
        model_adapter: PipelineInterface,
        analysis_config: AnalysisConfig,
        device: DeviceType = DeviceType.NPU,
    ) -> AnalysisResult:
        """
        Analyze model layers based on given configuration.

        Args:
            model_adapter: The model to analyze
            analysis_config: 分析配置（scope / metrics / calib_dataset / linear_pattern | quant_modules）
            device: 运行设备
        """
        raise NotImplementedError
