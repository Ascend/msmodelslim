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

import pickle
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import torch

from msmodelslim.core.analysis_service import (
    AnalysisConfig,
    AnalysisResult,
    AnalysisScope,
    PipelineAnalysisService,
)
from msmodelslim.processor.analysis.unary_operator.metrics.ra_compress import (
    DUMMY_INPUT_LENGTH,
    PREFIX_TOKEN_ID,
    PLACEHOLDER_DATASET,
    REPET_TIMES,
    RaCompressAnalysisInterface,
)
from msmodelslim.processor.analysis.unary_operator.metrics.ra_compress.calib_input import _RandomTokenForward
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


class _ForwardRecordingAdapter:
    """最小适配器：只实现 runner 用到的模型前向入口，并记录真正进入模型的输入。"""

    def __init__(self):
        self.forward_inputs = []

    def generate_model_forward(self, model, inputs):
        """runner 的模型前向入口：记录本次进入模型的 token_id 输入。"""
        _ = model
        self.forward_inputs.append(inputs)
        return iter(())


class _FixedIdTokenizer:
    """最小 tokenizer 替身：对任意输入都返回同一 token id，用于覆盖 V0 首 token 解析路径。"""

    def __init__(self, token_id):
        self.token_id = token_id

    def __call__(self, text, return_tensors=None):
        _ = (text, return_tensors)
        return {"input_ids": torch.tensor([[self.token_id]])}


class _TokenizerAdapter(_ForwardRecordingAdapter, RaCompressAnalysisInterface):
    """实现 ra_compress 可选钩子的适配器；钩子取值可为 token id、None（无 tokenizer）或待抛出的异常。"""

    def __init__(self, prefix_token_id=None):
        super().__init__()
        self._prefix_token_id = prefix_token_id

    def get_proj_names(self):
        return {}

    def get_tokenizer(self):
        if isinstance(self._prefix_token_id, Exception):
            raise self._prefix_token_id
        if self._prefix_token_id is None:
            return None
        return _FixedIdTokenizer(self._prefix_token_id)


class _RecordingRunner:
    """记录 run() 收到的校准数据，并在运行期观测模型前向真正拿到的 token_id 输入。"""

    def __init__(self, adapter=None):
        self.adapter = adapter
        self.calib_data = None

    def add_processor(self, _cfg):
        pass

    def run(self, calib_data=None, device=None, device_indices=None):
        _ = (device, device_indices)
        self.calib_data = calib_data
        if self.adapter is not None:
            # 运行期观测：runner 交出的是占位样本，真正进入前向的是被顶替的随机 token 段
            placeholder = [torch.ones((1, 2), dtype=torch.long)] * 2
            list(self.adapter.generate_model_forward(None, placeholder))


class TestRandomTokenDataset:
    """ra_compress：token_id 输入由分析服务侧构造，并在模型前向入口注入。"""

    @staticmethod
    def _mock_context_factory():
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ns = MagicMock()
        mock_ns.debug = {
            "layer_scores": [{"name": "layer1", "score": 1.0}],
            "method": "ra_compress",
            "patterns": ["*"],
        }
        mock_ctx.__getitem__ = lambda _self, k: mock_ns if k == "layer_analysis" else mock_ctx
        factory = MagicMock()
        factory.create.return_value = mock_ctx
        return factory

    @classmethod
    def _service(cls, calib_data):
        mock_dataset_loader = MagicMock()
        mock_dataset_loader.get_dataset_by_name.return_value = calib_data
        mock_pipeline_loader = MagicMock()
        builder = MagicMock()
        builder.template_modules.return_value = builder
        builder.create.return_value = []
        mock_pipeline_loader.get_pipeline_builder.return_value = builder
        return (
            PipelineAnalysisService(mock_dataset_loader, cls._mock_context_factory(), mock_pipeline_loader),
            mock_dataset_loader,
        )

    def _run_analyze(self, service: PipelineAnalysisService, model_adapter: Any) -> _RecordingRunner:
        """以 ra_compress 配置执行一次 analyze，返回记录校准数据的 runner。"""
        config = AnalysisConfig(
            scope=AnalysisScope.ATTN_HEAD,
            metrics="ra_compress",
            calib_dataset="mix_calib.jsonl",
        )
        runner = _RecordingRunner()

        def _build_runner(adapter=None, **_kwargs):
            runner.adapter = adapter
            return runner

        with patch(
            "msmodelslim.core.analysis_service.pipeline_analysis.service.LayerWiseRunner",
            side_effect=_build_runner,
        ):
            service.analyze(
                model_adapter=model_adapter,
                analysis_config=config,
                device=DeviceType.CPU,
            )
        return runner

    @patch("msmodelslim.core.analysis_service.pipeline_analysis.service.get_logger")
    def test_analyze_builds_random_recipe_when_ra_compress(self, _mock_logger):
        """场景：metrics=ra_compress。预期：校准输入由服务侧按 V0 配方自造，不询问数据集加载器。"""
        service, mock_dataset_loader = self._service([{"unused": 1}])
        model_adapter = _TokenizerAdapter(prefix_token_id=32)

        runner = self._run_analyze(service, model_adapter)

        mock_dataset_loader.get_dataset_by_name.assert_not_called()
        # runner 侧只消费占位样本，真实随机段在运行期于前向口注入
        assert runner.calib_data == PLACEHOLDER_DATASET

    @patch("msmodelslim.core.analysis_service.pipeline_analysis.service.get_logger")
    def test_analyze_injects_random_tokens_and_restores_forward_when_ra_compress(self, _mock_logger):
        """场景：metrics=ra_compress。预期：运行期以随机重复段顶替前向入口，运行结束后还原。"""
        service, _ = self._service([{"unused": 1}])
        model_adapter = _TokenizerAdapter(prefix_token_id=32)
        original_forward = model_adapter.generate_model_forward

        self._run_analyze(service, model_adapter)

        # 顶替生效：进入前向的是随机重复段（长度 1 + 段长 × 段数），而非 runner 交出的占位样本
        input_ids, attention_mask = model_adapter.forward_inputs[0]
        assert input_ids.shape == (1, 1 + DUMMY_INPUT_LENGTH * REPET_TIMES)
        assert attention_mask.shape == input_ids.shape
        # 首 token 取自适配器 get_tokenizer 按 V0 口径解析的 token id
        assert int(input_ids[0, 0]) == 32
        # 运行结束还原，避免影响同一适配器上的其它流程
        assert model_adapter.generate_model_forward == original_forward

    def test_random_token_forward_survives_pickle_roundtrip(self):
        """场景：适配器可能被序列化后在其他进程使用。预期：顶替对象可 pickle 往返，还原后不递归。"""
        model_adapter = _TokenizerAdapter()
        token_inputs = [torch.ones((1, 4), dtype=torch.long)] * 2
        model_adapter.generate_model_forward = _RandomTokenForward(model_adapter, token_inputs)

        restored = pickle.loads(pickle.dumps(model_adapter))

        # 顶替对象按类取原实现：若持有原绑定方法，反序列化后该属性会指回自身而无限递归
        list(restored.generate_model_forward(None, [torch.ones((1, 2), dtype=torch.long)]))
        assert len(restored.forward_inputs) == 1
        assert torch.equal(restored.forward_inputs[0][0], token_inputs[0])

    @patch("msmodelslim.core.analysis_service.pipeline_analysis.service.get_logger")
    def test_analyze_falls_back_to_default_prefix_when_adapter_has_no_hook(self, _mock_logger):
        """异常：适配器未实现 RaCompressAnalysisInterface。预期：首 token 回退为兜底值，不报错。"""
        service, _ = self._service([{"unused": 1}])
        model_adapter = _ForwardRecordingAdapter()

        self._run_analyze(service, model_adapter)

        assert int(model_adapter.forward_inputs[0][0][0, 0]) == PREFIX_TOKEN_ID

    @patch("msmodelslim.core.analysis_service.pipeline_analysis.service.get_logger")
    def test_analyze_falls_back_to_default_prefix_when_tokenizer_missing(self, _mock_logger):
        """异常：适配器 get_tokenizer 返回 None（如 tokenizer 未加载）。预期：首 token 回退为兜底值。"""
        service, _ = self._service([{"unused": 1}])
        model_adapter = _TokenizerAdapter(prefix_token_id=None)

        self._run_analyze(service, model_adapter)

        assert int(model_adapter.forward_inputs[0][0][0, 0]) == PREFIX_TOKEN_ID

    @patch("msmodelslim.core.analysis_service.pipeline_analysis.service.get_logger")
    def test_analyze_falls_back_to_default_prefix_when_tokenizer_raises(self, _mock_logger):
        """异常：适配器 get_tokenizer 抛异常。预期：首 token 回退为兜底值，不中断分析。"""
        service, _ = self._service([{"unused": 1}])
        model_adapter = _TokenizerAdapter(prefix_token_id=RuntimeError("tokenizer unavailable"))

        self._run_analyze(service, model_adapter)

        assert int(model_adapter.forward_inputs[0][0][0, 0]) == PREFIX_TOKEN_ID

    @patch("msmodelslim.core.analysis_service.pipeline_analysis.service.get_logger")
    def test_analyze_loads_calib_dataset_when_metric_is_not_ra_compress(self, _mock_logger):
        """场景：metrics=std。预期：走常规数据集加载，不构造随机段校准集。"""
        calib_data = [{"input_ids": torch.tensor([[1, 2]])}]
        service, mock_dataset_loader = self._service(calib_data)
        config = AnalysisConfig(
            scope=AnalysisScope.LINEAR,
            metrics="std",
            calib_dataset="mix_calib.jsonl",
            linear_pattern=["*"],
        )

        runner = _RecordingRunner()
        with patch(
            "msmodelslim.core.analysis_service.pipeline_analysis.service.LayerWiseRunner",
            return_value=runner,
        ):
            service.analyze(
                model_adapter=MagicMock(spec=PipelineInterface),
                analysis_config=config,
                device=DeviceType.CPU,
            )

        mock_dataset_loader.get_dataset_by_name.assert_called_once_with("mix_calib.jsonl")
        assert runner.calib_data is calib_data
