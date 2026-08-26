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

import pytest

import torch

from msmodelslim.core.analysis_service import (
    AnalysisConfig,
    AnalysisResult,
    AnalysisScope,
    PipelineAnalysisService,
)
from msmodelslim.core.const import DeviceType
from msmodelslim.utils.exception import InvalidDatasetError, SecurityError
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

        with patch("msmodelslim.core.analysis_service.pipeline_analysis.service.LayerWiseRunner") as mock_lw_cls:
            result = service.analyze(
                model_adapter=mock_model_adapter,
                analysis_config=analysis_config,
                device=DeviceType.CPU,
            )

        mock_lw_cls.assert_called_once_with(adapter=mock_model_adapter)
        assert result is not None
        assert result.layer_scores == [{"name": "layer1", "score": 1.0}]
        assert result.method == "std"
        assert result.patterns == ["*"]

    @patch("msmodelslim.core.analysis_service.pipeline_analysis.service.get_logger")
    def test_analyze_raises_when_calib_dataset_missing(self, _mock_logger):
        """场景：dataset_loader 返回 None。预期：抛出 InvalidDatasetError。"""
        mock_dataset_loader = MagicMock()
        mock_dataset_loader.get_dataset_by_name.return_value = None
        service = PipelineAnalysisService(mock_dataset_loader, MagicMock(), MagicMock())
        config = AnalysisConfig(
            scope=AnalysisScope.LINEAR,
            metrics="std",
            calib_dataset="missing.jsonl",
            linear_pattern=["*"],
        )
        with pytest.raises(InvalidDatasetError):
            service.analyze(
                model_adapter=MagicMock(spec=PipelineInterface),
                analysis_config=config,
                device=DeviceType.CPU,
            )

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

    @patch("msmodelslim.core.analysis_service.pipeline_analysis.service.get_logger")
    def test_analyze_uses_dp_runner_when_multi_device(self, _mock_logger):
        """场景：device_indices 长度 > 1（CPU 环境）。预期：使用 DPLayerWiseRunner + shared context。"""
        mock_dataset_loader = MagicMock()
        mock_dataset_loader.get_dataset_by_name.return_value = [{"data": 1}]
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ns = MagicMock()
        mock_ns.debug = {
            "layer_scores": [{"name": "layer1", "score": 1.0}],
            "method": "kurtosis",
            "patterns": ["*"],
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
            scope=AnalysisScope.LINEAR,
            metrics="kurtosis",
            calib_dataset="test.jsonl",
            linear_pattern=["*"],
        )
        mock_model_adapter = MagicMock(spec=PipelineInterface)
        with (
            patch("msmodelslim.core.runner.dp_layer_wise_runner.DPLayerWiseRunner") as mock_dp_cls,
            patch("msmodelslim.core.analysis_service.pipeline_analysis.service.LayerWiseRunner") as mock_lw_cls,
        ):
            mock_dp = MagicMock()
            mock_dp_cls.return_value = mock_dp
            result = service.analyze(
                model_adapter=mock_model_adapter,
                analysis_config=config,
                device=DeviceType.CPU,
                device_indices=[0, 1],
            )

        mock_context_factory.create.assert_called_once_with(is_distributed=True)
        mock_dp_cls.assert_called_once_with(adapter=mock_model_adapter)
        mock_lw_cls.assert_not_called()
        mock_dp.run.assert_called_once()
        assert mock_dp.run.call_args.kwargs["device_indices"] == [0, 1]
        assert result.method == "kurtosis"

    @patch("msmodelslim.core.analysis_service.pipeline_analysis.service.get_logger")
    def test_analyze_falls_back_to_next_loader_when_first_fails(self, _mock_logger):
        """场景：首个 loader 失败、第二个成功。预期：使用第二个 loader 的数据。"""
        mock_dataset_loader = MagicMock()
        mock_dataset_loader.get_dataset_by_name.side_effect = InvalidDatasetError("not a text jsonl")
        mock_vlm_loader = MagicMock()
        mock_vlm_loader.get_dataset_by_name.return_value = [{"pixel_values": 1}]
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ns = MagicMock()
        mock_ns.debug = {
            "layer_scores": [{"name": "layer1", "score": 2.0}],
            "method": "kurtosis",
            "patterns": ["*"],
        }
        mock_ctx.__getitem__ = lambda _self, k: mock_ns if k == "layer_analysis" else mock_ctx
        mock_context_factory = MagicMock()
        mock_context_factory.create.return_value = mock_ctx
        mock_builder = MagicMock()
        mock_builder.template_modules.return_value = mock_builder
        mock_builder.create.return_value = []
        mock_pipeline_loader = MagicMock()
        mock_pipeline_loader.get_pipeline_builder.return_value = mock_builder

        service = PipelineAnalysisService(
            mock_dataset_loader,
            mock_context_factory,
            mock_pipeline_loader,
            vlm_dataset_loader=mock_vlm_loader,
        )
        config = AnalysisConfig(
            scope=AnalysisScope.LINEAR,
            metrics="kurtosis",
            calib_dataset="calibImages",
            linear_pattern=["*"],
        )
        with patch("msmodelslim.core.analysis_service.pipeline_analysis.service.LayerWiseRunner"):
            result = service.analyze(
                model_adapter=MagicMock(spec=PipelineInterface),
                analysis_config=config,
                device=DeviceType.CPU,
            )

        mock_dataset_loader.get_dataset_by_name.assert_called_once_with("calibImages")
        mock_vlm_loader.get_dataset_by_name.assert_called_once_with("calibImages")
        assert result.layer_scores == [{"name": "layer1", "score": 2.0}]

    @patch("msmodelslim.core.analysis_service.pipeline_analysis.service.get_logger")
    def test_analyze_falls_back_when_first_loader_raises_security_error(self, _mock_logger):
        """场景：FileDatasetLoader 因扩展名校验抛 SecurityError。预期：回退到 VLM loader。"""
        mock_dataset_loader = MagicMock()
        mock_dataset_loader.get_dataset_by_name.side_effect = SecurityError(
            'The filename calib_image_data doesn\'t endswith "jsonl".'
        )
        mock_vlm_loader = MagicMock()
        mock_vlm_loader.get_dataset_by_name.return_value = [{"pixel_values": 1}]
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ns = MagicMock()
        mock_ns.debug = {
            "layer_scores": [{"name": "layer1", "score": 2.0}],
            "method": "kurtosis",
            "patterns": ["*"],
        }
        mock_ctx.__getitem__ = lambda _self, k: mock_ns if k == "layer_analysis" else mock_ctx
        mock_context_factory = MagicMock()
        mock_context_factory.create.return_value = mock_ctx
        mock_builder = MagicMock()
        mock_builder.template_modules.return_value = mock_builder
        mock_builder.create.return_value = []
        mock_pipeline_loader = MagicMock()
        mock_pipeline_loader.get_pipeline_builder.return_value = mock_builder

        service = PipelineAnalysisService(
            mock_dataset_loader,
            mock_context_factory,
            mock_pipeline_loader,
            vlm_dataset_loader=mock_vlm_loader,
        )
        config = AnalysisConfig(
            scope=AnalysisScope.LINEAR,
            metrics="kurtosis",
            calib_dataset="calib_image_data",
            linear_pattern=["*"],
        )
        with patch("msmodelslim.core.analysis_service.pipeline_analysis.service.LayerWiseRunner"):
            result = service.analyze(
                model_adapter=MagicMock(spec=PipelineInterface),
                analysis_config=config,
                device=DeviceType.CPU,
            )

        mock_vlm_loader.get_dataset_by_name.assert_called_once_with("calib_image_data")
        assert result.layer_scores == [{"name": "layer1", "score": 2.0}]

    @patch("msmodelslim.core.analysis_service.pipeline_analysis.service.get_logger")
    def test_analyze_uses_first_loader_when_it_succeeds(self, _mock_logger):
        """场景：首个 loader 已成功。预期：不再调用后续 loader。"""
        mock_dataset_loader = MagicMock()
        mock_dataset_loader.get_dataset_by_name.return_value = ["hello"]
        mock_vlm_loader = MagicMock()
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
        mock_context_factory = MagicMock()
        mock_context_factory.create.return_value = mock_ctx
        mock_builder = MagicMock()
        mock_builder.template_modules.return_value = mock_builder
        mock_builder.create.return_value = []
        mock_pipeline_loader = MagicMock()
        mock_pipeline_loader.get_pipeline_builder.return_value = mock_builder

        service = PipelineAnalysisService(
            mock_dataset_loader,
            mock_context_factory,
            mock_pipeline_loader,
            vlm_dataset_loader=mock_vlm_loader,
        )
        config = AnalysisConfig(
            scope=AnalysisScope.LINEAR,
            metrics="std",
            calib_dataset="mix_calib.jsonl",
            linear_pattern=["*"],
        )
        with patch("msmodelslim.core.analysis_service.pipeline_analysis.service.LayerWiseRunner"):
            result = service.analyze(
                model_adapter=MagicMock(spec=PipelineInterface),
                analysis_config=config,
                device=DeviceType.CPU,
            )

        mock_dataset_loader.get_dataset_by_name.assert_called_once_with("mix_calib.jsonl")
        mock_vlm_loader.get_dataset_by_name.assert_not_called()
        assert result.layer_scores == [{"name": "layer1", "score": 1.0}]
