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

msmodelslim.infra.analysis_pipeline_loader 模块的单元测试
"""

import pytest

# 触发分析处理器配置子类注册，使 AutoProcessorConfig 能按 type 分派
import msmodelslim.processor.analysis.unary_operator.processor  # noqa: F401  pylint: disable=unused-import
import msmodelslim.processor.analysis.binary_operator.processor  # noqa: F401  pylint: disable=unused-import

from msmodelslim.infra.analysis_pipeline_loader import (
    METRIC_TO_YAML,
    TemplatePipelineBuilder,
    YamlAnalysisPipelineLoader,
)
from msmodelslim.processor.base import AutoProcessorConfig
from msmodelslim.utils.exception import UnsupportedError


class TestYamlAnalysisPipelineLoader:
    """测试 YamlAnalysisPipelineLoader 的建造者创建与模板渲染"""

    def test_get_pipeline_builder_return_template_builder(self):
        """正常路径：get_pipeline_builder 返回模板渲染建造者"""
        builder = YamlAnalysisPipelineLoader().get_pipeline_builder("mse")

        assert isinstance(builder, TemplatePipelineBuilder)

    def test_create_unary_pipeline(self):
        """正常路径：quantile 模板渲染出单个 unary_analysis 处理器配置"""
        configs = (
            YamlAnalysisPipelineLoader()
            .get_pipeline_builder("quantile")
            .template_modules(["a.weight", "b.weight"])
            .create()
        )

        assert len(configs) == 1
        assert isinstance(configs[0], AutoProcessorConfig)
        assert configs[0].type == "unary_analysis"

    def test_create_binary_pipeline_with_nested_config(self):
        """正常路径：mse 模板渲染出 binary_analysis 并保留嵌套子配置"""
        configs = YamlAnalysisPipelineLoader().get_pipeline_builder("mse").template_modules(["a.weight"]).create()

        assert len(configs) == 1
        assert configs[0].type == "binary_analysis"

    def test_create_raise_unsupported_when_metric_unknown(self):
        """异常：未知 metric 时抛 UnsupportedError"""
        with pytest.raises(UnsupportedError):
            YamlAnalysisPipelineLoader().get_pipeline_builder("not_a_metric").create()

    def test_metric_to_yaml_covers_known_metrics(self):
        """正常路径：度量到模板映射包含全部受支持度量"""
        assert "mse" in METRIC_TO_YAML
        assert METRIC_TO_YAML["quantile"] == "quantile.yaml"
