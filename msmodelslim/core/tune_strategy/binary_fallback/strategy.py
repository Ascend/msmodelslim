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

from typing import Any, Dict, Generator, List, Literal, Optional

from pydantic import Field, model_validator

from msmodelslim.core.analysis_service import (
    AnalysisConfig,
    AnalysisScope,
    PipelineAnalysisService,
)
from msmodelslim.core.const import DeviceType
from msmodelslim.core.context import ContextFactory
from msmodelslim.core.practice import Metadata, PracticeConfig
from msmodelslim.core.quant_service.modelslim_v1.quant_config import (
    ModelslimV1ServiceConfig,
    load_specific_config,
)
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.core.tune_strategy import ITuningStrategy
from msmodelslim.core.tune_strategy.base import BaseTuningStrategy
from msmodelslim.core.tune_strategy.dataset_loader_infra import DatasetLoaderInfra
from msmodelslim.core.tune_strategy.interface import EvaluateResult, StrategyConfig
from msmodelslim.infra.analysis_pipeline_loader import YamlAnalysisPipelineLoader
from msmodelslim.model import IModel
from msmodelslim.utils.exception import SchemaValidateError, SpecError, UnsupportedError
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.pydantic_model_path import set_pydantic_model_path, validate_pydantic_model_list_path

MODELSLIM_V1_APIVERSION = "modelslim_v1"


class BinaryFallbackStrategyConfig(StrategyConfig):
    """二分回退调优策略配置。"""

    type: Literal["binary_fallback"] = "binary_fallback"
    template: PracticeConfig = Field(description="完整最佳实践 PracticeConfig，apiversion 须为 modelslim_v1")
    rollback_path: str = Field(description="点分路径，指向 template 内必须为 list 的回退字段")
    rollback_candidates: Optional[List[str]] = Field(
        default=None,
        description="有序回退候选；非空则跳过敏感层分析",
    )
    analysis_dataset: Optional[str] = Field(
        default=None,
        description="敏感层分析校准集名称；未填则使用 template.spec.dataset",
    )

    @model_validator(mode="after")
    def validate_binary_fallback_config(self):
        if self.template.apiversion != MODELSLIM_V1_APIVERSION:
            raise SchemaValidateError(
                f"binary_fallback only supports apiversion={MODELSLIM_V1_APIVERSION!r}, "
                f"got {self.template.apiversion!r}"
            )
        spec = load_specific_config(self.template.spec)
        self.template.spec = spec
        validate_pydantic_model_list_path(self.template, self.rollback_path)
        return self


@logger_setter("msmodelslim.core.tune_strategy.binary_fallback")
class BinaryFallbackStrategy(BaseTuningStrategy, ITuningStrategy):
    def __init__(
        self,
        config: BinaryFallbackStrategyConfig,
        dataset_loader: DatasetLoaderInfra,
        **kwargs,
    ):
        self.config = config
        self.__counter = 0
        self._analysis_layer_scores: List[Dict[str, Any]] = []
        super().__init__(config, dataset_loader, **kwargs)

    def generate_practice(
        self,
        model: IModel,
        device: DeviceType = DeviceType.NPU,
    ) -> Generator[PracticeConfig, Optional[EvaluateResult], None]:
        self.__counter = 0
        candidates = self._resolve_candidates(model=model, device=device)

        zero_evaluation = yield self._build_practice(rollback_list=[])
        if zero_evaluation.is_satisfied:
            get_logger().info("[BinaryFallback] Practice without rollback satisfies demand.")
            return

        if not candidates:
            raise SpecError(
                "Accuracy demand not satisfied with zero rollback and no rollback candidates available.",
                action="Provide rollback_candidates or ensure sensitivity analysis returns layers.",
            )

        min_k = yield from self._find_satisfied_rollback_count(candidates)
        yield self._build_practice(rollback_list=candidates[:min_k])

    def _resolve_candidates(self, model: IModel, device: DeviceType) -> List[str]:
        user_candidates = self.config.rollback_candidates
        if user_candidates:
            get_logger().info(
                "[BinaryFallback] Using user-provided rollback candidates, count=%d", len(user_candidates)
            )
            return list(user_candidates)

        if not isinstance(model, PipelineInterface):
            raise UnsupportedError(
                f"model must implement PipelineInterface for sensitivity analysis, got {type(model)}"
            )

        self._run_sensitive_layer_analysis(model=model, device=device)

        candidates = [row["name"] for row in self._analysis_layer_scores if "name" in row]
        get_logger().info("[BinaryFallback] Sensitivity analysis returned %d rollback candidates.", len(candidates))
        return candidates

    def _get_v1_spec(self) -> ModelslimV1ServiceConfig:
        spec = self.config.template.spec
        if isinstance(spec, ModelslimV1ServiceConfig):
            return spec
        return load_specific_config(spec)

    def _analysis_dataset_name(self) -> str:
        if self.config.analysis_dataset:
            return self.config.analysis_dataset
        return self._get_v1_spec().dataset

    def _run_sensitive_layer_analysis(
        self,
        model: PipelineInterface,
        device: DeviceType,
    ) -> None:
        analysis_service = PipelineAnalysisService(
            dataset_loader=self.dataset_loader,
            context_factory=ContextFactory(enable_debug=True),
            pipeline_loader=YamlAnalysisPipelineLoader(),
        )
        result = analysis_service.analyze(
            model_adapter=model,
            analysis_config=AnalysisConfig(
                scope=AnalysisScope.LAYER,
                metrics="mse_layer_wise",
                calib_dataset=self._analysis_dataset_name(),
                quant_modules=["*"],
            ),
            device=device,
        )
        layer_scores = list((result.layer_scores if result is not None else []) or [])
        self._analysis_layer_scores = sorted(
            layer_scores,
            key=lambda item: item.get("score", 0.0),
            reverse=True,
        )
        get_logger().info("[BinaryFallback] Sensitivity analysis completed. Top layers (descending):")
        for i, item in enumerate(self._analysis_layer_scores):
            get_logger().info("  [%d] %s (score=%.6f)", i, item.get("name", "?"), item.get("score", 0.0))

    def _build_practice(self, rollback_list: List[str]) -> PracticeConfig:
        # 敏感层分析返回 block 级别名称（如 "model.layers.15"），
        # 需要转换为 glob 通配符（如 "*model.layers.15.*"）才能匹配到具体线性层。
        # 用户提供的 rollback_candidates 可能已包含通配符，保持原样。
        wildcard_list = [name if "*" in name else f"*{name}.*" for name in rollback_list]
        get_logger().info(
            "[BinaryFallback] Building practice with %d rollback layers: %s",
            len(wildcard_list),
            wildcard_list,
        )
        practice = set_pydantic_model_path(self.config.template, self.config.rollback_path, list(wildcard_list))
        new_config_id = f"{practice.metadata.config_id}_{self.__counter}"
        self.__counter += 1
        practice.metadata = Metadata(
            config_id=new_config_id,
            score=practice.metadata.score,
            label=dict(practice.metadata.label),
            verified_model_types=list(practice.metadata.verified_model_types),
            verified_tags=dict(practice.metadata.verified_tags),
        )
        return practice

    def _find_satisfied_rollback_count(
        self,
        candidates: List[str],
    ) -> Generator[PracticeConfig, Optional[EvaluateResult], int]:
        min_k, max_k = 1, len(candidates)
        get_logger().info("[BinaryFallback] Binary search rollback count in [%d, %d]", min_k, max_k)

        while min_k < max_k:
            mid = (min_k + max_k) // 2
            get_logger().debug("[BinaryFallback] Trying rollback count: %r", mid)
            evaluation: EvaluateResult = yield self._build_practice(rollback_list=candidates[:mid])
            if evaluation.is_satisfied:
                max_k = mid
            else:
                min_k = mid + 1

        get_logger().info("[BinaryFallback] Fundamental rollback count: %r", min_k)
        final_evaluation: EvaluateResult = yield self._build_practice(rollback_list=candidates[:min_k])
        if not final_evaluation.is_satisfied:
            raise SpecError(
                "Accuracy demand not satisfied with maximum rollback.",
                action="Relax accuracy expectations or extend rollback_candidates.",
            )
        return min_k


def get_plugin():
    """
    获取 binary_fallback 策略插件（返回配置类与组件类，由框架完成注册）。
    Returns:
        (BinaryFallbackStrategyConfig, BinaryFallbackStrategy) 元组
    """
    return BinaryFallbackStrategyConfig, BinaryFallbackStrategy
