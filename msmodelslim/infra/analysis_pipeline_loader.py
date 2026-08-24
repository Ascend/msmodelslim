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
"""
from pathlib import Path
from string import Template
from typing import Any, Dict, List

import yaml
from typing_extensions import Self

from msmodelslim.core.analysis_service.pipeline_analysis.pipeline_loader_infra import (
    AnalysisPipelineLoaderInfra,
    PipelineBuilderInfra,
)
from msmodelslim.processor.base import AutoProcessorConfig
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.security.path import get_valid_read_path


METRIC_TO_YAML: Dict[str, str] = {
    "quantile": "quantile.yaml",
    "std": "std.yaml",
    "kurtosis": "kurtosis.yaml",
    "mse": "attention_mse.yaml",
    "mse_layer_wise": "mse_layer_wise.yaml",
    "mse_model_wise": "mse_model_wise.yaml",
}

def _get_analysis_pipeline_dir() -> Path:
    cur_dir = Path(__file__).resolve().parent
    rel = Path("core/analysis_service/pipeline_analysis/pipeline_template")
    analysis_pipeline_dir = get_valid_read_path(str(cur_dir.parent / rel), is_dir=True)
    return Path(analysis_pipeline_dir)


class YamlAnalysisPipelineLoader(AnalysisPipelineLoaderInfra):
    """按 metrics 加载模板，返回建造者；建造者链式设置参数后 create() 得到配置列表。"""

    def get_pipeline_builder(self, metrics: str) -> PipelineBuilderInfra:
        return TemplatePipelineBuilder(metrics)
        

class TemplatePipelineBuilder(PipelineBuilderInfra):
    """模板渲染建造者：链式设置占位符后 create() 得到配置列表。"""

    def __init__(self, metrics: str):
        self._template_dir = _get_analysis_pipeline_dir()
        self._metrics = metrics.strip().lower()
        self._template_modules: List[str] = []

    def template_modules(self, modules: List[str]) -> Self:
        self._template_modules = list(modules)
        return self

    def create(self) -> List[AutoProcessorConfig]:
        template_path = self._get_template_path()
        template_text = Path(template_path).read_text(encoding="utf-8")
        dumped = yaml.dump(
            self._template_modules,
            default_flow_style=True,
            allow_unicode=True,
        ).strip()
        substitute_dict: Dict[str, Any] = {
            "patterns": dumped,
            "quant_modules": dumped,
        }
        rendered = Template(template_text).safe_substitute(**substitute_dict)
        data = yaml.safe_load(rendered)
        if not isinstance(data, dict) or "process" not in data:
            raise SchemaValidateError(
                "Pipeline template must has 'process' key",
                action=f"Please check the pipeline template {template_path}.",
            )
        process = data["process"]
        if not isinstance(process, list):
            process = []

        configs: List[AutoProcessorConfig] = []
        for item in process:
            if not isinstance(item, dict):
                continue
            configs.append(AutoProcessorConfig.model_validate(dict(item)))
        return configs

    def _get_template_path(self) -> str:
        if self._metrics not in METRIC_TO_YAML:
            raise UnsupportedError(
                f"Unsupported analysis metric: {self._metrics!r}. Supported: {list(METRIC_TO_YAML.keys())}"
                )
        template_path = self._template_dir / METRIC_TO_YAML[self._metrics]
        resolved = get_valid_read_path(str(template_path))
        return resolved
