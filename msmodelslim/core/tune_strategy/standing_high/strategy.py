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

from typing import Any, Dict, Generator, List, Optional, Literal, Annotated

from pydantic import ConfigDict, Field, model_validator, AfterValidator

from msmodelslim.core.analysis_service import AnalysisConfig, AnalysisScope, PipelineAnalysisService
from msmodelslim.core.const import DeviceType
from msmodelslim.core.context import ContextFactory
from msmodelslim.core.quant_service.modelslim_v1.quant_config import ModelslimV1ServiceConfig
from msmodelslim.core.quant_service.modelslim_v1.save import AscendV1Config
from msmodelslim.core.practice import PracticeConfig, Metadata
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.core.tune_strategy import ITuningStrategy
from msmodelslim.core.tune_strategy.base import BaseTuningStrategy
from msmodelslim.core.tune_strategy.dataset_loader_infra import DatasetLoaderInfra
from msmodelslim.core.tune_strategy.interface import StrategyConfig, EvaluateResult
from msmodelslim.infra.analysis_pipeline_loader import YamlAnalysisPipelineLoader
from msmodelslim.ir.qal import QScope, QDType
from msmodelslim.model import IModel
from msmodelslim.processor.base import AutoProcessorConfigList
from msmodelslim.processor.quant.linear import LinearProcessorConfig
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.validation.pydantic import at_least_one_element
from msmodelslim.utils.config_map import ConfigSet


def _create_default_template() -> ModelslimV1ServiceConfig:
    """创建默认的V1模板PracticeConfig"""
    return ModelslimV1ServiceConfig(
        process=[
            LinearProcessorConfig(
                type="linear_quant",
                qconfig=LinearQConfig(
                    act=QConfig(scope=QScope.PER_TENSOR, dtype=QDType.INT8, symmetric=False, method="minmax"),
                    weight=QConfig(scope=QScope.PER_CHANNEL, dtype=QDType.INT8, symmetric=True, method="minmax"),
                ),
                include=["*"],
                exclude=[],
            ),
        ],
        save=[
            AscendV1Config(type="ascendv1_saver", part_file_size=4),
        ],
        dataset="mix_calib.jsonl",
    )


def _get_default_metadata():
    return Metadata(
        config_id='standing_high',
        label={'w_bit': 8, 'a_bit': 8, 'is_sparse': False, 'kv_cache': False},
    )


_FULL_EXAMPLE = {
    'type': 'standing_high',
    'anti_outlier_strategies': [[{'type': 'iter_smooth', 'alpha': 0.5}], [{'type': 'flex_smooth_quant'}]],
    'template': {
        'runner': 'auto',
        'process': [
            {
                'type': 'linear_quant',
                'qconfig': {
                    'act': {'scope': 'per_tensor', 'dtype': 'int8', 'symmetric': False, 'method': 'minmax'},
                    'weight': {'scope': 'per_channel', 'dtype': 'int8', 'symmetric': True, 'method': 'minmax'},
                },
                'include': ['*'],
                'exclude': [],
            }
        ],
        'save': [{'type': 'ascendv1_saver', 'part_file_size': 4}],
        'dataset': 'mix_calib.jsonl',
    },
    'metadata': {
        'config_id': 'standing_high',
        'label': {'w_bit': 8, 'a_bit': 8, 'is_sparse': False, 'kv_cache': False},
    },
}


class StandingHighStrategyConfig(StrategyConfig):
    """摸高算法策略配置（V1框架）：先跑敏感层分析，再逐步回退不满意的层并尝试不同离群值抑制策略。"""

    model_config = ConfigDict(json_schema_extra={"examples": [_FULL_EXAMPLE]})

    type: Literal["standing_high"] = Field(default="standing_high", description="策略类型，固定为 `standing_high`")

    anti_outlier_strategies: Annotated[List[AutoProcessorConfigList], AfterValidator(at_least_one_element)] = Field(
        description="离群值抑制处理器链列表，至少 1 个；每个元素是一条处理器链，链内每个处理器是 `type` 分派的处理器配置（如 smooth_quant 等）"
    )

    template: ModelslimV1ServiceConfig = Field(
        default_factory=_create_default_template,
        description="完整的PracticeConfig模板，用于提取所有配置（包括线性层量化）。如果未提供，将使用默认的V1模板",
    )

    metadata: Metadata = Field(
        default_factory=_get_default_metadata, description="量化配置元数据（config_id / label 等）"
    )

    @model_validator(mode='after')
    def validate_template_has_linear_quant(self):
        """校验template中至少有一个linear_quant配置"""
        linear_quant_count = sum(1 for proc in self.template.process if isinstance(proc, LinearProcessorConfig))
        if linear_quant_count < 1:
            raise SchemaValidateError(
                f"template_practice must contain at least one linear_quant processor, found {linear_quant_count}"
            )
        return self


@logger_setter("msmodelslim.core.tune_strategy.standing_high")
class StandingHighStrategy(BaseTuningStrategy, ITuningStrategy):
    def __init__(self, config: StandingHighStrategyConfig, dataset_loader: DatasetLoaderInfra, **kwargs):
        self.config = config
        self.__counter = 0
        self._current_index = 0
        self._analysis_layer_scores: List[Dict[str, Any]] = []
        super().__init__(config, dataset_loader, **kwargs)

    def generate_practice(
        self,
        model: IModel,
        device: DeviceType = DeviceType.NPU,
    ) -> Generator[PracticeConfig, Optional[EvaluateResult], None]:
        """生成实践配置（V1框架）"""
        if not isinstance(model, PipelineInterface):
            raise UnsupportedError(
                f"model must implement PipelineInterface for sensitivity analysis, got {type(model)}"
            )

        # 每次生成时重置计数器
        self.__counter = 0

        self._run_sensitive_layer_analysis(model=model, device=device)

        zero_evaluation = yield self._build_practice_config(self.config.anti_outlier_strategies[0], [])

        if zero_evaluation.is_satisfied:
            get_logger().info("Practice without disable layers satisfies demand.")
            return

        init_disable_level = yield from self._find_satisfied_disable_level()
        yield from self._stand_high(init_disable_level)

    def _run_sensitive_layer_analysis(self, model: PipelineInterface, device: DeviceType) -> None:
        """Run sensitivity analysis and cache sorted layer scores for selection."""
        include_patterns: List[str] = []
        for proc in self.config.template.process:
            if isinstance(proc, LinearProcessorConfig):
                include_patterns.extend(proc.include or ["*"])
        include_patterns = list(dict.fromkeys(include_patterns)) or ["*"]

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
                calib_dataset=self.config.template.dataset,
                quant_modules=include_patterns,
            ),
            device=device,
        )

        layer_scores = list((result.layer_scores if result is not None else []) or [])
        self._analysis_layer_scores = sorted(
            layer_scores,
            key=lambda x: x.get("score", 0.0),
            reverse=True,
        )

    def select_layers_by_disable_level(self, disable_level: int) -> List[str]:
        """Select top-k fallback layers by disable_level."""
        if disable_level <= 0 or not self._analysis_layer_scores:
            return []
        topk = self._analysis_layer_scores[:disable_level]
        return [row["name"] for row in topk if "name" in row]

    def _build_practice_config(
        self,
        anti_outlier_processors: AutoProcessorConfigList,
        linear_quant_exclude: List[str],
    ) -> PracticeConfig:
        """构建PracticeConfig：离群值抑制processors放在最前面，然后拼接模板中的所有processors

        将回退层写入所有 linear_quant 的 exclude。
        """
        process: AutoProcessorConfigList = list(anti_outlier_processors)
        template = self.config.template

        # 回退层名改为通配符形式，敏感层分析结果："model.layers.2" 改为 "*model.layers.2.*"，
        wildcard_excludes = [f"*{name}.*" for name in linear_quant_exclude]
        wildcard_excludes = list(dict.fromkeys(wildcard_excludes))

        # 为每个 linear_quant 分配回退层
        for proc in template.process:
            if isinstance(proc, LinearProcessorConfig):
                # 合并原始 exclude（去重，保持顺序）
                exclude_list = list(dict.fromkeys(proc.exclude or []))
                if exclude_list:
                    exclude_set = ConfigSet(exclude_list)
                    to_add = [pat for pat in wildcard_excludes if pat not in exclude_set]
                else:
                    to_add = wildcard_excludes
                exclude_list.extend(to_add)

                process.append(
                    LinearProcessorConfig(
                        type="linear_quant", qconfig=proc.qconfig, include=proc.include, exclude=exclude_list
                    )
                )
            else:
                process.append(proc)

        # 生成新的 config_id
        new_config_id = f"{self.config.metadata.config_id}_{self.__counter}"
        self.__counter += 1

        return PracticeConfig(
            apiversion="modelslim_v1",
            spec=ModelslimV1ServiceConfig(
                runner=template.runner, process=process, save=template.save, dataset=template.dataset
            ),
            metadata=Metadata(
                config_id=new_config_id,
                label=self.config.metadata.label,
            ),
        )

    def _find_satisfied_disable_level(
        self,
    ) -> Generator[PracticeConfig, Optional[EvaluateResult], int]:
        """二分搜索找到满足需求的最小disable level（zero_evaluation一定不满足，所以从level 1开始）"""
        min_level, max_level = 1, len(self._analysis_layer_scores)

        while min_level < max_level:
            mid = (min_level + max_level) // 2
            exclude_names = self.select_layers_by_disable_level(mid)
            get_logger().debug("Trying disable level: %r, exclude: %r", mid, exclude_names)

            evaluation: EvaluateResult = yield self._build_practice_config(
                self.config.anti_outlier_strategies[0], exclude_names
            )

            if evaluation.is_satisfied:
                max_level = mid
            else:
                min_level = mid + 1

        get_logger().info("Fundamental disable level: %r", min_level)
        return min_level

    def _permute(self) -> Generator[AutoProcessorConfigList, None, None]:
        """排列离群值抑制策略，具有记忆功能：从上次成功的策略开始遍历"""
        choices = self.config.anti_outlier_strategies
        for _ in range(len(choices)):
            yield choices[self._current_index]
            self._current_index = (self._current_index + 1) % len(choices)

    def _stand_high(
        self,
        init_disable_level: int,
    ) -> Generator[PracticeConfig, Optional[EvaluateResult], None]:
        """执行摸高算法：逐步减少disable level，尝试不同的离群值抑制策略"""
        # 获得最小回退层级的量化配置
        anti_outlier_processors = self.config.anti_outlier_strategies[0]
        exclude_names = self.select_layers_by_disable_level(init_disable_level)
        best_practice_config = self._build_practice_config(anti_outlier_processors, exclude_names)

        # 初始化步长和当前回退级别
        reduce_level = 1
        current_disable_level = init_disable_level

        while current_disable_level - reduce_level >= 0:
            exclude_names = self.select_layers_by_disable_level(current_disable_level - reduce_level)
            get_logger().info("Current disable level: %r, try to reduce: %r", current_disable_level, reduce_level)

            for anti_outlier_processors in self._permute():
                get_logger().debug(
                    "Trying config: exclude=%r, anti_outlier=%r",
                    exclude_names,
                    [p.type for p in anti_outlier_processors],
                )
                practice = self._build_practice_config(anti_outlier_processors, exclude_names)
                evaluation: EvaluateResult = yield practice

                if evaluation.is_satisfied:
                    best_practice_config = practice
                    current_disable_level -= reduce_level
                    reduce_level = max(min(reduce_level * 2, current_disable_level), 1)
                    break
            else:
                if reduce_level == 1:
                    yield best_practice_config
                    return
                reduce_level = 1
                get_logger().info("Reset reduce level to 1")

        yield best_practice_config

        get_logger().info("No disable layer can satisfy demand")


def get_plugin():
    """
    获取 standing_high 策略插件（返回配置类与组件类，由框架完成注册）。
    Returns:
        (StandingHighStrategyConfig, StandingHighStrategy) 元组
    """
    return StandingHighStrategyConfig, StandingHighStrategy
