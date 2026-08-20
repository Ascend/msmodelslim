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

from typing import List, Literal, Optional, Generator

from pydantic import ConfigDict, Field, field_validator

from msmodelslim.core.const import DeviceType, QuantType
from msmodelslim.core.practice import PracticeConfig
from msmodelslim.core.tune_strategy import ITuningStrategy
from msmodelslim.core.tune_strategy.base import BaseTuningStrategy
from msmodelslim.core.tune_strategy.dataset_loader_infra import DatasetLoaderInfra
from msmodelslim.core.tune_strategy.common.config_builder.expert_experience import (
    ExpertExperienceConfigBuilder,
    ExpertExperienceLoader,
    StructureConfig,
)
from msmodelslim.core.tune_strategy.interface import StrategyConfig, EvaluateResult
from msmodelslim.core.tune_strategy.standing_high.strategy import StandingHighStrategy, StandingHighStrategyConfig
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.core.tune_strategy.standing_high_with_experience.standing_high_with_experience_interface import (
    StandingHighWithExperienceInterface,
)
from msmodelslim.model import IModel
from msmodelslim.processor.base import AutoSessionProcessor, AutoProcessorConfigList
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import logger_setter, get_logger


_FULL_EXAMPLE = {
    'type': 'standing_high_with_experience',
    'quant_type': 'w8a8',
    'structure_configs': [{'type': 'GQA', 'include': ['*self_attn*']}, {'type': 'FFN', 'include': ['*mlp*']}],
}


class StandingHighWithExperienceStrategyConfig(StrategyConfig):
    """基于专家经验的摸高算法策略配置"""

    model_config = ConfigDict(json_schema_extra={"examples": [_FULL_EXAMPLE]})

    type: Literal["standing_high_with_experience"] = Field(
        default="standing_high_with_experience", description="策略类型，固定为 `standing_high_with_experience`"
    )

    structure_configs: List[StructureConfig] = Field(
        description="结构配置列表，每个配置包含结构类型和对应的 include/exclude，例如 [{'type': 'GQA', 'include': ['*self_attn*'], 'exclude': ['*kv_b_proj']}]（必填，无默认配置）"
    )
    quant_type: QuantType = Field(
        default=QuantType.W8A8, description="量化类型（QuantType 枚举），须在专家经验 supported_quant_types 范围内。"
    )

    @field_validator("quant_type", mode="before")
    @classmethod
    def _quant_type_only_expert_supported(cls, v):
        allowed_values = set(ExpertExperienceLoader.get_supported_quant_types())
        value_str = v.value if isinstance(v, QuantType) else v
        if value_str not in allowed_values:
            raise SchemaValidateError(
                f"quant_type must be one of expert experience supported_quant_types. "
                f"Input should be one of {sorted(allowed_values)}, got {v!r}.",
                action="Please set quant_type to one of the values in expert_experience.yaml supported_quant_types.",
            )
        return v


@logger_setter("msmodelslim.core.tune_strategy.standing_high_with_experience")
class StandingHighWithExperienceStrategy(BaseTuningStrategy, ITuningStrategy):
    """
    基于专家经验的摸高算法策略

    使用组合关系，内部使用 StandingHighStrategy 来执行摸高算法。
    职责：
    1. 根据简化的配置利用专家经验生成完整的 StandingHighStrategyConfig
    2. 委托 StandingHighStrategy 执行实际的摸高算法
    """

    def __init__(self, config: StrategyConfig, dataset_loader: DatasetLoaderInfra):
        super().__init__(config, dataset_loader)

        if not isinstance(config, StandingHighWithExperienceStrategyConfig):
            raise SchemaValidateError(
                f"StandingHighWithExperienceStrategy requires StandingHighWithExperienceStrategyConfig, "
                f"got {type(config)}"
            )

        self.config: StandingHighWithExperienceStrategyConfig = config

    def generate_practice(
        self,
        model: IModel,
        device: DeviceType = DeviceType.NPU,
    ) -> Generator[PracticeConfig, Optional[EvaluateResult], None]:
        """
        生成实践配置

        在首次调用时，根据原始配置生成完整的 StandingHighStrategyConfig，
        然后创建 StandingHighStrategy 实例并委托其执行摸高算法。

        Note:
        - 模型适配器需要实现 StandingHighWithExperienceInterface（离群值抑制能力探测需 load_model）
        - 同时需要实现 PipelineInterface（敏感层分析 pipeline，与 ModelSlimPipelineInterfaceV1 相同）
        """
        if not isinstance(model, StandingHighWithExperienceInterface):
            raise UnsupportedError(f"model must be StandingHighWithExperienceInterface, got {type(model)}")
        if not isinstance(model, PipelineInterface):
            raise UnsupportedError(
                f"model must implement PipelineInterface for sensitivity analysis, got {type(model)}"
            )
        # 处理 structure_configs：如果为 None，尝试自动检测
        structure_configs = self.config.structure_configs
        if structure_configs is None:
            get_logger().info("structure_configs not provided, attempting to auto-detect model structure from model")
            auto_detected = self._auto_detect_structure_configs(model)  # pylint: disable=assignment-from-none
            if auto_detected is not None:
                get_logger().info("Successfully auto-detected model structure")
                structure_configs = auto_detected
                self.config.structure_configs = structure_configs
            else:
                get_logger().warning(
                    "Auto-detect not implemented yet (interface reserved), using default structure configs."
                )
                structure_configs = self._get_default_structure_configs()

        # 生成完整的基础配置
        base_config = self._generate_base_config(self.config, structure_configs, model=model)

        # 创建 StandingHighStrategy 实例并委托其执行摸高算法
        standing_high_strategy = StandingHighStrategy(config=base_config, dataset_loader=self.dataset_loader)

        yield from standing_high_strategy.generate_practice(model, device)

    def _generate_base_config(
        self,
        config: StandingHighWithExperienceStrategyConfig,
        structure_configs: List[StructureConfig],
        model: Optional[IModel] = None,
    ) -> StandingHighStrategyConfig:
        """
        根据输入配置生成完整的 StandingHighStrategyConfig。
        会过滤掉当前环境不支持的离群值抑制策略并打日志警告。

        Args:
            config: StandingHighWithExperienceStrategyConfig 配置对象
            structure_configs: 结构配置列表（已处理，不会是 None）
            model: 可选，模型适配器（需同时为 PipelineInterface），用于检测离群值抑制是否支持

        Returns:
            生成的 StandingHighStrategyConfig 配置对象
        """
        quant_type_str = config.quant_type.value

        get_logger().info(
            "Generating full config for standing_high_with_experience with structure_configs=%s, quant_type=%s",
            structure_configs,
            quant_type_str,
        )

        builder = ExpertExperienceConfigBuilder()
        quant_config = builder.build(
            quant_type=quant_type_str,
            structure_configs=structure_configs,
        )
        search_space = builder.get_tuning_search_space(
            quant_type=quant_type_str,
            structure_configs=structure_configs,
        )
        if not search_space.anti_outlier_strategies:
            raise SchemaValidateError(
                "Expert experience builder must provide anti_outlier_strategies in tuning search space."
            )

        get_logger().info(
            "Tuning search space: anti_outlier_strategies_count=%s",
            len(search_space.anti_outlier_strategies or []),
        )

        get_logger().info("Tuning search space (anti_outlier_strategies): %s", search_space.anti_outlier_strategies)

        anti_outlier_strategies = search_space.anti_outlier_strategies
        if model is not None:
            anti_outlier_strategies = self._filter_supported_anti_outlier_strategies(anti_outlier_strategies, model)
            if not anti_outlier_strategies:
                raise SchemaValidateError(
                    "No supported anti_outlier strategy left after filtering. "
                    "Please check expert_experience.yaml and model adapter capabilities."
                )

        return StandingHighStrategyConfig(
            type="standing_high",
            anti_outlier_strategies=anti_outlier_strategies,
            template=quant_config.spec,
            metadata=quant_config.metadata,
        )

    def _filter_supported_anti_outlier_strategies(
        self,
        strategies: List[AutoProcessorConfigList],
        model: StandingHighWithExperienceInterface,
    ) -> List[AutoProcessorConfigList]:
        """
        过滤出当前模型支持的离群值抑制策略。
        """
        try:
            loaded_model = model.load_model(device="cpu")
        except Exception as e:
            get_logger().warning(
                "Cannot load model for anti_outlier support check, using all strategies: %s",
                e,
            )
            return strategies

        filtered: List[AutoProcessorConfigList] = []
        for idx, strategy_list in enumerate(strategies):
            for proc_cfg in strategy_list:
                try:
                    AutoSessionProcessor.from_config(loaded_model, proc_cfg, model)
                except UnsupportedError as e:
                    proc_type = getattr(proc_cfg, "type", type(proc_cfg).__name__)
                    get_logger().warning(
                        "Anti-outlier strategy not supported, skipping: index=%s, type=%s, reason=%s",
                        idx,
                        proc_type,
                        e,
                    )
                    break
            else:
                filtered.append(strategy_list)

        try:
            loaded_model.to("meta")
        except Exception as e:
            get_logger().warning(
                "Failed to release loaded_model (to('meta')), possible resource leak: %s",
                e,
            )

        return filtered

    def _auto_detect_structure_configs(self, model: IModel) -> Optional[List[StructureConfig]]:
        """
        自动检测模型结构配置

        根据模型适配器自动识别结构类型并生成对应的配置。

        Args:
            model: 模型适配器对象，可以通过 model.load_model() 等方法访问模型结构

        Returns:
            如果成功检测到模型结构，返回结构配置列表；否则返回 None
        """
        # 暂未实现，保留接口
        return None

    def _get_default_structure_configs(self) -> List[StructureConfig]:
        """
        获取默认的结构配置

        默认配置不区分模型结构，对所有层统一应用量化配置。

        Returns:
            默认结构配置列表
        """
        return [
            StructureConfig(
                type="MHA",  # 使用MHA作为通用结构类型，对所有层应用w8a8量化
                include=["*"],  # 匹配所有层
                exclude=[],  # 不排除任何层
            )
        ]


def get_plugin():
    """
    获取 standing_high_with_experience 策略插件（返回配置类与组件类，由框架通过 config.ini 完成注册）。

    Returns:
        (StandingHighWithExperienceStrategyConfig, StandingHighWithExperienceStrategy) 元组
    """
    return StandingHighWithExperienceStrategyConfig, StandingHighWithExperienceStrategy
