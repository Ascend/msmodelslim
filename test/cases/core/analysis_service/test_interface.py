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

import pytest

from msmodelslim.core.analysis_service.interface import (
    AnalysisConfig,
    AnalysisResult,
    AnalysisScope,
)
from msmodelslim.utils.exception import SchemaValidateError


class TestAnalysisConfig:
    """Tests for AnalysisConfig."""

    def test_template_substitute_list_returns_linear_pattern_when_scope_linear(self):
        """场景：scope=linear。预期：返回 linear_pattern。"""
        cfg = AnalysisConfig(
            scope=AnalysisScope.LINEAR,
            metrics="std",
            calib_dataset="calib.jsonl",
            linear_pattern=["*.linear"],
        )
        assert cfg.template_substitute_list() == ["*.linear"]

    def test_template_substitute_list_returns_quant_modules_when_scope_layer(self):
        """场景：scope=layer。预期：返回 quant_modules。"""
        cfg = AnalysisConfig(
            scope=AnalysisScope.LAYER,
            metrics="mse",
            calib_dataset="calib.jsonl",
            quant_modules=["block.*"],
        )
        assert cfg.template_substitute_list() == ["block.*"]

    def test_template_substitute_list_returns_wildcard_when_scope_attn(self):
        """场景：scope=attn。预期：返回 ['*']。"""
        cfg = AnalysisConfig(
            scope=AnalysisScope.ATTN,
            metrics="mse",
            calib_dataset="calib.jsonl",
        )
        assert cfg.template_substitute_list() == ["*"]

    def test_analysis_config_raises_schema_validate_error_when_linear_has_quant_modules(self):
        """场景：linear scope 设置 quant_modules。预期：SchemaValidateError。"""
        with pytest.raises(SchemaValidateError, match="quant_modules"):
            AnalysisConfig(
                scope=AnalysisScope.LINEAR,
                metrics="std",
                calib_dataset="calib.jsonl",
                linear_pattern=["*"],
                quant_modules=["block.*"],
            )

    def test_analysis_config_raises_schema_validate_error_when_layer_has_linear_pattern(self):
        """场景：layer scope 设置 linear_pattern。预期：SchemaValidateError。"""
        with pytest.raises(SchemaValidateError, match="linear_pattern"):
            AnalysisConfig(
                scope=AnalysisScope.LAYER,
                metrics="mse",
                calib_dataset="calib.jsonl",
                quant_modules=["*"],
                linear_pattern=["*.linear"],
            )

    def test_analysis_config_raises_schema_validate_error_when_attn_has_patterns(self):
        """场景：attn scope 设置 linear_pattern。预期：SchemaValidateError。"""
        with pytest.raises(SchemaValidateError, match="linear_pattern"):
            AnalysisConfig(
                scope=AnalysisScope.ATTN,
                metrics="mse",
                calib_dataset="calib.jsonl",
                linear_pattern=["*"],
            )


class TestAnalysisResult:
    """Tests for AnalysisResult."""

    def test_analysis_result_defaults_empty_layer_scores_when_omitted(self):
        """场景：仅传 method。预期：layer_scores 为空列表。"""
        result = AnalysisResult(method="std")
        assert result.layer_scores == []
        assert result.patterns == []

    def test_analysis_result_stores_layer_scores_when_provided(self):
        """场景：传入 layer_scores。预期：字段保留。"""
        scores = [{"name": "layer0", "score": 1.5}]
        result = AnalysisResult(layer_scores=scores, method="kurtosis", patterns=["*"])
        assert result.layer_scores == scores
        assert result.method == "kurtosis"
