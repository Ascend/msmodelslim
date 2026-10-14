"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

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

from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, AfterValidator
from typing import Annotated

from msmodelslim.core.practice import Metadata
from msmodelslim.processor.base import AutoProcessorConfig, AutoProcessorConfigList
from msmodelslim.processor.quant.linear import LinearProcessorConfig
from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.validation.pydantic import validate_str_length

from msmodelslim.core.tune_strategy.common.config_builder.base_builder import QuantizationConfigBuilder
from msmodelslim.core.tune_strategy.common.config_builder.quantization_config import TuningSearchSpace


class StructureConfig(BaseModel):
    """结构配置，定义模型结构的量化配置"""

    type: str = Field(min_length=1, description="结构类型，非空，例如 'GQA', 'MoE', 'FFN'")
    include: List[str] = Field(
        ..., min_length=1, description="包含的模式列表，必选，不可为空且不能包含空字符串，例如 ['*self_attn*']"
    )
    exclude: Optional[List[Annotated[str, AfterValidator(validate_str_length())]]] = Field(
        default=None, description="排除的模式列表，例如 ['*kv_b_proj', '*wq_b']"
    )

    @field_validator("include")
    @classmethod
    def include_no_empty_string(cls, v: List[str]) -> List[str]:
        for s in v:
            if not s or not str(s).strip():
                raise ValueError(
                    "structure_configs.include is a list of module name patterns; "
                    "include pattern must not be empty (e.g. '*self_attn*', '*mlp*')."
                )
        return v


class ExpertExperienceLoader:
    """
    专家经验配置加载器（Loader）。

    职责：负责从 expert_experience.yaml 加载配置并提供只读访问接口，
    不参与构建量化配置或策略列表。
    """

    _CONFIG: Optional[Dict[str, Any]] = None

    def __init__(self):
        if ExpertExperienceLoader._CONFIG is None:
            config_path = Path(__file__).parent / "expert_experience.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                ExpertExperienceLoader._CONFIG = yaml.safe_load(f)

    @classmethod
    def get_supported_quant_types(cls) -> List[str]:
        """从专家经验配置中读取支持的量化类型列表（与 supported_quant_types 一致）。"""
        if cls._CONFIG is None:
            cls()  # 触发加载
        return list(cls._CONFIG.get("supported_quant_types", []))

    def get_qconfig(self, quant_type: str, structure_type: str) -> Optional[Dict[str, Any]]:
        """获取结构类型对应的量化配置模板"""
        quantization_mapping = ExpertExperienceLoader._CONFIG.get("quantization_mapping", {})
        mapping = quantization_mapping.get(quant_type)
        if mapping is None:
            raise UnsupportedError(
                f"Unsupported quant_type: '{quant_type}'. Supported quant_types: {list(quantization_mapping.keys())}"
            )
        structure_qconfig_mapping = mapping.get("structure_qconfig_mapping", {})
        template_name = structure_qconfig_mapping.get(structure_type)
        if template_name is None:
            supported_structures = sorted(structure_qconfig_mapping.keys())
            raise UnsupportedError(
                f"Unsupported combination: quant_type '{quant_type}' + structure '{structure_type}'. "
                f"Supported structures for quant_type '{quant_type}': {supported_structures}. "
            )
        if template_name == "bf16":
            return None
        qconfig = ExpertExperienceLoader._CONFIG.get("quantization_configs", {}).get(template_name)
        return qconfig

    def get_anti_outlier_strategy_templates(self, quant_type: str) -> List[Dict[str, Any]]:
        """获取离群值抑制策略模板列表，按推荐优先级从高到低排序。"""
        quantization_mapping = ExpertExperienceLoader._CONFIG.get("quantization_mapping", {})
        mapping = quantization_mapping.get(quant_type)
        if mapping is None:
            raise UnsupportedError(
                f"Unsupported quant_type: '{quant_type}'. Supported quant_types: {list(quantization_mapping.keys())}"
            )
        template_names = mapping.get("anti_outlier_strategies", [])
        if not template_names:
            raise UnsupportedError(f"No anti_outlier_strategies configured for quant_type '{quant_type}'. ")
        strategies = ExpertExperienceLoader._CONFIG.get("anti_outlier_strategies", {})
        return [strategies[name] for name in template_names if strategies.get(name) is not None]


class ExpertExperienceConfigBuilder(QuantizationConfigBuilder):
    """
    基于专家经验的量化配置建造者。

    仅实现专家经验涉及的部分：metadata、spec.process，以及调优搜索空间中的
    anti_outlier_strategies；spec.dataset、spec.save 使用基类默认。

    期望调用方在 **kwargs 中传入：quant_type（str）、structure_configs（列表）。
    """

    def __init__(self, loader: Optional[ExpertExperienceLoader] = None):
        self._loader = loader if loader is not None else ExpertExperienceLoader()

    def build_metadata(self, **kwargs: Any) -> Metadata:
        """根据 quant_type 解析 w_bit/a_bit，构建 metadata。"""
        quant_type: str = kwargs.get("quant_type", "w8a8")
        try:
            parts = quant_type[1:].split("a")
            w_bit, a_bit = int(parts[0]), int(parts[1])
        except (ValueError, IndexError, AttributeError):
            w_bit, a_bit = 8, 8
        return Metadata(
            config_id="standing_high_with_experience",
            label={
                "w_bit": w_bit,
                "a_bit": a_bit,
                "is_sparse": False,
                "kv_cache": False,
            },
        )

    def build_spec_process(self, **kwargs: Any) -> List[Any]:
        """根据 structure_configs 与 quant_type 构建 spec.process（仅 linear_quant）。"""
        structure_configs = kwargs.get("structure_configs")
        if not structure_configs:
            raise UnsupportedError("ExpertExperienceConfigBuilder requires structure_configs in kwargs.")
        structure_configs = [StructureConfig.model_validate(c) for c in structure_configs]
        quant_type: str = kwargs.get("quant_type", "w8a8")
        get_logger().info(
            "Building spec.process with structure_configs=%s, quant_type=%s", structure_configs, quant_type
        )
        null_structure_includes: List[str] = []
        structure_configs_with_qconfig: List[tuple] = []
        for structure_config in structure_configs:
            qconfig_dict = self._loader.get_qconfig(quant_type, structure_config.type)
            if qconfig_dict is not None:
                structure_configs_with_qconfig.append((structure_config, qconfig_dict))
            else:
                get_logger().info(
                    "Structure %s is skipped (no quantization), "
                    "its include patterns will be excluded from other linear_quant processors",
                    structure_config.type,
                )
                null_structure_includes.extend(structure_config.include)
        process = []
        for structure_config, qconfig_dict in structure_configs_with_qconfig:
            exclude_list = list(structure_config.exclude or [])
            if null_structure_includes:
                exclude_list.extend(null_structure_includes)
            process.append(
                LinearProcessorConfig(
                    type="linear_quant", qconfig=qconfig_dict, include=structure_config.include, exclude=exclude_list
                )
            )
        if not process:
            raise UnsupportedError(
                f"No valid structure configs found for quant_type '{quant_type}'. "
                "All structures were skipped or no structure_configs provided. "
                "Please check model structure configuration."
            )
        return process

    def get_tuning_search_space(self, **kwargs: Any) -> TuningSearchSpace:
        """限定调优搜索范围：设置离群值抑制策略候选列表（摸高算法使用）。"""
        quant_type: str = kwargs.get("quant_type", "w8a8")
        get_logger().info("Building tuning search space (anti_outlier_strategies) for quant_type=%s", quant_type)
        templates = self._loader.get_anti_outlier_strategy_templates(quant_type)
        strategies: List[AutoProcessorConfigList] = [[AutoProcessorConfig.model_validate(t)] for t in templates]
        return TuningSearchSpace(anti_outlier_strategies=strategies)
