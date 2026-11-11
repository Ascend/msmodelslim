#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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

配置文档生成器的精选完整示例和场景加载器。

示例与场景统一以 YAML 文件维护，便于直接复制、审阅和校验；字段默认值、枚举和
约束仍由 schema 抽取。README 只负责说明和索引，不作为机器解析的数据源。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import yaml

from tools.config_docgen.records import ScenarioRecord


TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
PRACTICE_CONFIG_FILE = "practice_config.yaml"
EXPLICIT_EXAMPLE_FILES: Dict[str, str] = {
    "LinearProcessorConfig": "linear_processor.yaml",
    "AdaptRotationProcessorConfig": "adapt_rotation_processor.yaml",
    "ModelslimConvertQuantConfig": "modelslim_convert_quant.yaml",
    "RenamePattern": "rename_pattern.yaml",
    "ConvertOpConfig": "convert_op.yaml",
    "GroupProcessorConfig": "group_processor.yaml",
    "TLQOpConfig": "tlq_op.yaml",
    "QuantSaveProcessorConfig": "quant_save_processor.yaml",
}
TUNING_EXAMPLE_FILES: Dict[str, str] = {
    "StandingHighStrategyConfig": "strategy_standing_high.yaml",
    "StandingHighWithExperienceStrategyConfig": "strategy_standing_high_with_experience.yaml",
    "BinaryFallbackStrategyConfig": "strategy_binary_fallback.yaml",
    "ServiceOrientedEvaluateServiceConfig": "evaluation_service_oriented.yaml",
}


class ExampleLoadError(ValueError):
    """Raised when a YAML example cannot be loaded as a complete config mapping."""


@dataclass(frozen=True)
class ScenarioSpec:
    """A class-keyed scenario backed by one registered YAML file."""

    title: str
    file: str
    description: str = ""


SCENARIO_EXAMPLE_FILES: Dict[str, Tuple[ScenarioSpec, ...]] = {
    "UnaryAnalysisProcessorConfig": (
        ScenarioSpec(
            title="RA Compress 覆盖默认比例",
            description="为长序列压缩筛选需要保留的 KV head，并覆盖默认保留比例。",
            file="unary_analysis_ra_compress.yaml",
        ),
    )
}


def load_yaml_example(path: Union[Path, str]) -> Dict[str, Any]:
    """Load one YAML example and require a mapping at the document root."""
    example_path = Path(path)
    try:
        text = example_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExampleLoadError(f"无法读取配置示例 {example_path}: {exc}") from exc
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ExampleLoadError(f"配置示例不是合法 YAML {example_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExampleLoadError(f"配置示例根节点必须是 YAML 对象 {example_path}")
    return payload


@lru_cache(maxsize=None)
def load_practice_config_example() -> Dict[str, Any]:
    """Load the stable PracticeConfig example."""
    return load_yaml_example(TEMPLATE_DIR / PRACTICE_CONFIG_FILE)


@lru_cache(maxsize=None)
def load_explicit_examples() -> Dict[str, Dict[str, Any]]:
    """Load all class-keyed complete examples declared by the generator."""
    return {
        class_name: load_yaml_example(TEMPLATE_DIR / file_name)
        for class_name, file_name in EXPLICIT_EXAMPLE_FILES.items()
    }


@lru_cache(maxsize=None)
def load_tuning_examples() -> Dict[str, Dict[str, Any]]:
    """Load the strategy/evaluation subtrees used to compose tuning pages."""
    return {
        class_name: load_yaml_example(TEMPLATE_DIR / file_name)
        for class_name, file_name in TUNING_EXAMPLE_FILES.items()
    }


@lru_cache(maxsize=None)
def load_scenarios() -> Dict[str, list[ScenarioRecord]]:
    """Load class-keyed scenarios and preserve the registered YAML text verbatim."""
    scenarios: Dict[str, list[ScenarioRecord]] = {}
    for class_name, specs in SCENARIO_EXAMPLE_FILES.items():
        records = []
        for spec in specs:
            path = TEMPLATE_DIR / spec.file
            load_yaml_example(path)
            try:
                yaml_text = path.read_text(encoding="utf-8").rstrip()
            except OSError as exc:
                raise ExampleLoadError(f"无法读取配置示例 {path}: {exc}") from exc
            records.append(
                ScenarioRecord(
                    title=spec.title,
                    description=spec.description,
                    yaml=yaml_text,
                )
            )
        scenarios[class_name] = records
    return scenarios
