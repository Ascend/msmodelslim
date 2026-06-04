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

from unittest.mock import MagicMock, patch

import torch

from msmodelslim.core.analysis_service import (
    AnalysisConfig,
    AnalysisResult,
    AnalysisScope,
    PipelineAnalysisService,
)
from msmodelslim.core.const import DeviceType
from msmodelslim.core.runner.pipeline_interface import PipelineInterface


def create_mock_analysis_result(layer_scores: list) -> AnalysisResult:
    """构建 AnalysisResult 对象，用于测试输入。"""
    return AnalysisResult(
        layer_scores=layer_scores,
        method="kurtosis",
        patterns=["conv2d", "linear", "mlp"],
    )


class TestPipelineAnalysisService:
    """Tests for PipelineAnalysisService."""

    def test_init_stores_dependencies_when_constructed(self):
        """场景：正常构造。预期：依赖注入字段与传入 mock 一致。"""
        mock_dataset_loader = MagicMock()
        mock_context_factory = MagicMock()
        mock_pipeline_loader = MagicMock()
        service = PipelineAnalysisService(mock_dataset_loader, mock_context_factory, mock_pipeline_loader)
        assert service.dataset_loader is mock_dataset_loader
        assert service.context_factory is mock_context_factory
        assert service.pipeline_loader is mock_pipeline_loader

    @patch("msmodelslim.core.analysis_service.pipeline_analysis.service.get_logger")
    def test_analyze_returns_analysis_result_when_flow_succeeds(self, _mock_logger):
        """场景：校准数据与 context 正常。预期：返回含 layer_scores 的 AnalysisResult。"""
        mock_dataset_loader = MagicMock()
        mock_dataset_loader.get_dataset_by_name.return_value = [{"input_ids": torch.tensor([[1, 2]])}]
        mock_context_factory = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ns = MagicMock()
        mock_ns.debug = {
            "layer_scores": [{"name": "layer1", "score": 1.0}],
            "method": "std",
            "patterns": ["*"],
        }
        mock_ctx.__getitem__ = lambda _self, k: mock_ns if k == "layer_analysis" else mock_ctx
        mock_context_factory.create.return_value = mock_ctx
        mock_pipeline_loader = MagicMock()
        mock_builder = MagicMock()
        mock_builder.template_modules.return_value = mock_builder
        mock_builder.create.return_value = []
        mock_pipeline_loader.get_pipeline_builder.return_value = mock_builder

        service = PipelineAnalysisService(mock_dataset_loader, mock_context_factory, mock_pipeline_loader)
        mock_model_adapter = MagicMock(spec=PipelineInterface)
        analysis_config = AnalysisConfig(
            scope=AnalysisScope.LINEAR,
            metrics="std",
            calib_dataset="test.jsonl",
            linear_pattern=["*"],
        )

        with patch("msmodelslim.core.analysis_service.pipeline_analysis.service.LayerWiseRunner"):
            result = service.analyze(
                model_adapter=mock_model_adapter,
                analysis_config=analysis_config,
                device=DeviceType.CPU,
            )

        assert result is not None
        assert result.layer_scores == [{"name": "layer1", "score": 1.0}]
        assert result.method == "std"
        assert result.patterns == ["*"]

    @patch("msmodelslim.core.analysis_service.pipeline_analysis.service.get_logger")
    def test_analyze_returns_none_when_calib_dataset_missing(self, _mock_logger):
        """场景：dataset_loader 返回 None。预期：analyze 返回 None。"""
        mock_dataset_loader = MagicMock()
        mock_dataset_loader.get_dataset_by_name.return_value = None
        service = PipelineAnalysisService(mock_dataset_loader, MagicMock(), MagicMock())
        config = AnalysisConfig(
            scope=AnalysisScope.LINEAR,
            metrics="std",
            calib_dataset="missing.jsonl",
            linear_pattern=["*"],
        )
        result = service.analyze(
            model_adapter=MagicMock(spec=PipelineInterface),
            analysis_config=config,
            device=DeviceType.CPU,
        )
        assert result is None

    @patch("msmodelslim.core.analysis_service.pipeline_analysis.service.get_logger")
    def test_analyze_uses_quant_modules_patterns_when_layer_scope(self, _mock_logger):
        """场景：layer scope 且 debug 含 quant_modules。预期：patterns 来自 quant_modules。"""
        mock_dataset_loader = MagicMock()
        mock_dataset_loader.get_dataset_by_name.return_value = [{"data": 1}]
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ns = MagicMock()
        mock_ns.debug = {
            "layer_scores": [],
            "method": "mse",
            "quant_modules": ["block.a", "block.b"],
        }
        mock_ctx.__getitem__ = lambda _self, k: mock_ns if k == "layer_analysis" else mock_ctx
        mock_context_factory = MagicMock()
        mock_context_factory.create.return_value = mock_ctx
        mock_builder = MagicMock()
        mock_builder.template_modules.return_value = mock_builder
        mock_builder.create.return_value = []
        mock_pipeline_loader = MagicMock()
        mock_pipeline_loader.get_pipeline_builder.return_value = mock_builder

        service = PipelineAnalysisService(mock_dataset_loader, mock_context_factory, mock_pipeline_loader)
        config = AnalysisConfig(
            scope=AnalysisScope.LAYER,
            metrics="mse",
            calib_dataset="test.jsonl",
            quant_modules=["block.*"],
        )
        with patch("msmodelslim.core.analysis_service.pipeline_analysis.service.LayerWiseRunner"):
            result = service.analyze(
                model_adapter=MagicMock(spec=PipelineInterface),
                analysis_config=config,
                device=DeviceType.CPU,
            )
        assert result.patterns == ["block.a", "block.b"]
