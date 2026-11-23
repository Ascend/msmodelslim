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

import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest
import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.quantization_schema.auto import AutoFakeQuantDynamicCache
from msmodelslim.core.quantization_schema.qal import QDType, QScope
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantization_algorithm.quant.kvcache.processor import (
    HOOK_TARGET,
    DynamicCacheProcessorConfig,
    DynamicCacheQuantProcessor,
    DynamicCacheDetails,
)
from msmodelslim.utils import function_hijacker, hook_utils
from msmodelslim.utils.exception import VersionError, UnsupportedError


def create_basic_qconfig(dtype=QDType.INT8, scope=QScope.PER_CHANNEL, symmetric=True) -> QConfig:
    return QConfig(dtype=dtype, scope=scope, symmetric=symmetric, method="minmax")


def create_processor_config(include=None, exclude=None) -> DynamicCacheProcessorConfig:
    if include is None:
        include = ["*"]
    if exclude is None:
        exclude = []
    return DynamicCacheProcessorConfig(qconfig=create_basic_qconfig(), include=include, exclude=exclude)


def create_simple_model():
    class SimpleAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(64, 64)
            self.key = nn.Linear(64, 64)
            self.value = nn.Linear(64, 64)

        def forward(self, x):
            return self.query(x) + self.key(x) + self.value(x)

    class SimpleDecoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = SimpleAttention()
            self.mlp = nn.Linear(64, 64)

        def forward(self, x):
            return self.mlp(self.self_attn(x))

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([SimpleDecoderLayer() for _ in range(2)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    return SimpleModel()


class TestDynamicCacheProcessorConfig(unittest.TestCase):
    def test_config_creation(self):
        config = create_processor_config()
        self.assertIsInstance(config, DynamicCacheProcessorConfig)
        self.assertEqual(config.qconfig.dtype, QDType.INT8)
        self.assertEqual(config.qconfig.scope, QScope.PER_CHANNEL)
        self.assertTrue(config.qconfig.symmetric)

    def test_config_with_include_exclude(self):
        config = create_processor_config(include=["layer1"], exclude=["layer2"])
        self.assertEqual(config.include, ["layer1"])
        self.assertEqual(config.exclude, ["layer2"])

    def test_config_default_fallback(self):
        config = DynamicCacheProcessorConfig()
        self.assertEqual(config.qconfig.dtype, QDType.INT8)
        self.assertEqual(config.qconfig.scope, QScope.PER_CHANNEL)

    def test_details_config_kv_independent(self):
        details = DynamicCacheDetails(
            k_qconfig=QConfig(dtype=QDType.MXFP8, scope=QScope.PER_TOKEN, symmetric=True, method="minmax"),
            v_qconfig=QConfig(dtype=QDType.FP8_E4M3, scope=QScope.PER_CHANNEL, symmetric=True, method="minmax"),
        )
        config = DynamicCacheProcessorConfig(details=details)
        self.assertIsNone(config.qconfig)
        self.assertEqual(config.branch_qconfig("k_qconfig").scope, QScope.PER_TOKEN)
        self.assertEqual(config.branch_qconfig("v_qconfig").scope, QScope.PER_CHANNEL)

    def test_details_config_k_only(self):
        details = DynamicCacheDetails(
            k_qconfig=QConfig(dtype=QDType.FP8_E4M3, scope=QScope.PER_CHANNEL, symmetric=True, method="minmax"),
        )
        config = DynamicCacheProcessorConfig(details=details)
        self.assertIsNotNone(config.branch_qconfig("k_qconfig"))
        self.assertIsNone(config.branch_qconfig("v_qconfig"))

    def test_qconfig_details_exclusive(self):
        from msmodelslim.utils.exception import SchemaValidateError

        details = DynamicCacheDetails(
            k_qconfig=QConfig(dtype=QDType.MXFP8, scope=QScope.PER_TOKEN, symmetric=True, method="minmax"),
        )
        with self.assertRaises(SchemaValidateError):
            DynamicCacheProcessorConfig(qconfig=create_basic_qconfig(), details=details)

    def test_type_equals_dynamic_cache(self):
        config = create_processor_config()
        self.assertEqual(config.type, "dynamic_cache")


class TestDynamicCacheQuantProcessor(unittest.TestCase):
    def setUp(self):
        self.model = create_simple_model()

    @patch("msmodelslim.core.quantization_algorithm.quant.kvcache.processor.DYNAMIC_AVAILABLE", True)
    def test_processor_initialization(self):
        config = create_processor_config()
        processor = DynamicCacheQuantProcessor(self.model, config)
        self.assertEqual(processor.config, config)

    @patch("msmodelslim.core.quantization_algorithm.quant.kvcache.processor.DYNAMIC_AVAILABLE", False)
    def test_processor_without_dynamic_cache(self):
        config = create_processor_config()
        with self.assertRaises(VersionError):
            DynamicCacheQuantProcessor(self.model, config)

    @patch("msmodelslim.core.quantization_algorithm.quant.kvcache.processor.DYNAMIC_AVAILABLE", True)
    def test_invalid_qconfig_scope(self):
        """PER_TENSOR is neither static nor dynamic: it must fail loudly instead of quantizing nothing."""
        config = DynamicCacheProcessorConfig(qconfig=create_basic_qconfig(scope=QScope.PER_TENSOR))
        with self.assertRaisesRegex(ValueError, "does not support scope per_tensor"):
            DynamicCacheQuantProcessor(self.model, config)

    @patch("msmodelslim.core.quantization_algorithm.quant.kvcache.processor.DYNAMIC_AVAILABLE", True)
    def test_support_distributed(self):
        config = create_processor_config()
        processor = DynamicCacheQuantProcessor(self.model, config)
        self.assertTrue(processor.support_distributed())

    @patch("msmodelslim.core.quantization_algorithm.quant.kvcache.processor.DYNAMIC_AVAILABLE", True)
    def test_need_kv_cache(self):
        config = create_processor_config()
        processor = DynamicCacheQuantProcessor(self.model, config)
        self.assertTrue(processor.need_kv_cache())

    @patch("msmodelslim.core.quantization_algorithm.quant.kvcache.processor.DYNAMIC_AVAILABLE", True)
    def test_adapter_accepts_any_object(self):
        config = create_processor_config()
        processor = DynamicCacheQuantProcessor(self.model, config, adapter="any_object")
        self.assertEqual(processor.adapter, "any_object")


class TestBranchDetection(unittest.TestCase):
    def setUp(self):
        self.model = create_simple_model()

    @patch("msmodelslim.core.quantization_algorithm.quant.kvcache.processor.DYNAMIC_AVAILABLE", True)
    def test_static_dynamic_branch_detection(self):
        details = DynamicCacheDetails(
            k_qconfig=QConfig(dtype=QDType.MXFP8, scope=QScope.PER_TOKEN, symmetric=True, method="minmax"),
            v_qconfig=QConfig(dtype=QDType.FP8_E4M3, scope=QScope.PER_CHANNEL, symmetric=True, method="minmax"),
        )
        config = DynamicCacheProcessorConfig(details=details)
        processor = DynamicCacheQuantProcessor(self.model, config)
        self.assertTrue(processor._is_dynamic_branch("key_states"))
        self.assertFalse(processor._is_static_branch("key_states"))
        self.assertFalse(processor._is_dynamic_branch("value_states"))
        self.assertTrue(processor._is_static_branch("value_states"))

    @patch("msmodelslim.core.quantization_algorithm.quant.kvcache.processor.DYNAMIC_AVAILABLE", True)
    def test_is_data_free_all_dynamic(self):
        details = DynamicCacheDetails(
            k_qconfig=QConfig(dtype=QDType.MXFP8, scope=QScope.PER_TOKEN, symmetric=True, method="minmax"),
            v_qconfig=QConfig(dtype=QDType.MXFP8, scope=QScope.PER_TOKEN, symmetric=True, method="minmax"),
        )
        config = DynamicCacheProcessorConfig(details=details)
        processor = DynamicCacheQuantProcessor(self.model, config)
        self.assertTrue(processor.is_data_free())

    @patch("msmodelslim.core.quantization_algorithm.quant.kvcache.processor.DYNAMIC_AVAILABLE", True)
    def test_is_data_free_mixed_not_data_free(self):
        details = DynamicCacheDetails(
            k_qconfig=QConfig(dtype=QDType.MXFP8, scope=QScope.PER_TOKEN, symmetric=True, method="minmax"),
            v_qconfig=QConfig(dtype=QDType.FP8_E4M3, scope=QScope.PER_CHANNEL, symmetric=True, method="minmax"),
        )
        config = DynamicCacheProcessorConfig(details=details)
        processor = DynamicCacheQuantProcessor(self.model, config)
        self.assertFalse(processor.is_data_free())

    @patch("msmodelslim.core.quantization_algorithm.quant.kvcache.processor.DYNAMIC_AVAILABLE", True)
    def test_is_data_free_all_static(self):
        config = create_processor_config()
        processor = DynamicCacheQuantProcessor(self.model, config)
        self.assertFalse(processor.is_data_free())


class TestAttentionLayerDetection(unittest.TestCase):
    def setUp(self):
        self.model = create_simple_model()

    @patch("msmodelslim.core.quantization_algorithm.quant.kvcache.processor.DYNAMIC_AVAILABLE", True)
    def test_detect_attention_layers(self):
        config = create_processor_config()
        processor = DynamicCacheQuantProcessor(self.model, config)
        result = processor._detect_attention_layers("model", self.model)
        self.assertGreater(len(result), 0)
        for full_name in result:
            has_attn = "attention" in full_name.lower() or "attn" in full_name.lower()
            self.assertTrue(has_attn)

    @patch("msmodelslim.core.quantization_algorithm.quant.kvcache.processor.DYNAMIC_AVAILABLE", True)
    def test_detect_attention_layers_with_exclude(self):
        config = create_processor_config(include=["*"], exclude=["*layers.0*"])
        processor = DynamicCacheQuantProcessor(self.model, config)
        result = processor._detect_attention_layers("model", self.model)
        for full_name in result:
            self.assertNotIn("layers.0", full_name)


class TestAttentionLayersMap(unittest.TestCase):
    def setUp(self):
        self.model = create_simple_model()

    def _make_processor(self):
        config = create_processor_config()
        processor = DynamicCacheQuantProcessor(self.model, config)
        mock_adapter = MagicMock()
        mock_adapter.inject_kvcache_placeholders.return_value = {}
        processor.adapter = mock_adapter
        return processor

    @patch("msmodelslim.core.quantization_algorithm.quant.kvcache.processor.DYNAMIC_AVAILABLE", True)
    def test_builds_in_preprocess(self):
        processor = self._make_processor()
        processor.adapter.inject_kvcache_placeholders.return_value = {
            "model.layers.0.self_attn": self.model.layers[0].self_attn,
            "model.layers.1.self_attn": self.model.layers[1].self_attn,
        }
        request = MagicMock()
        request.name = "model"
        request.module = self.model
        processor.preprocess(request)
        self.assertIn(0, processor._attention_layers_map)
        self.assertIn(1, processor._attention_layers_map)
        self.assertEqual(processor._attention_layers_map[0], "model.layers.0.self_attn")
        self.assertEqual(processor._attention_layers_map[1], "model.layers.1.self_attn")

    @patch("msmodelslim.core.quantization_algorithm.quant.kvcache.processor.DYNAMIC_AVAILABLE", True)
    def test_does_not_overwrite_existing(self):
        processor = self._make_processor()
        processor._attention_layers_map[0] = "model.visual.0.self_attn"
        processor.adapter.inject_kvcache_placeholders.return_value = {
            "model.language.0.self_attn": self.model.layers[0].self_attn,
        }
        request = MagicMock()
        request.name = "model.language"
        request.module = self.model
        processor.preprocess(request)
        self.assertEqual(processor._attention_layers_map[0], "model.visual.0.self_attn")

    @patch("msmodelslim.core.quantization_algorithm.quant.kvcache.processor.DYNAMIC_AVAILABLE", True)
    def test_get_full_name_by_layer_idx(self):
        processor = self._make_processor()
        processor._attention_layers_map[0] = "model.layers.0.self_attn"
        self.assertEqual(processor._get_full_name_by_layer_idx(0), "model.layers.0.self_attn")
        self.assertIsNone(processor._get_full_name_by_layer_idx(99))

    @patch("msmodelslim.core.quantization_algorithm.quant.kvcache.processor.DYNAMIC_AVAILABLE", True)
    def test_fallback_on_unsupported_error(self):
        processor = self._make_processor()
        processor.adapter.inject_kvcache_placeholders.side_effect = UnsupportedError("not supported")
        request = MagicMock()
        request.name = "model"
        request.module = self.model
        processor.preprocess(request)
        self.assertGreater(len(processor.attention_layers), 0)

    @patch("msmodelslim.core.quantization_algorithm.quant.kvcache.processor.DYNAMIC_AVAILABLE", True)
    def test_fallback_on_adapter_without_method(self):
        config = create_processor_config()
        processor = DynamicCacheQuantProcessor(self.model, config, adapter=object())
        request = MagicMock()
        request.name = "model"
        request.module = self.model
        processor.preprocess(request)
        self.assertGreater(len(processor.attention_layers), 0)


class TestDistributedSupport(unittest.TestCase):
    def setUp(self):
        self.model = create_simple_model()

    @patch("msmodelslim.core.quantization_algorithm.quant.kvcache.processor.DYNAMIC_AVAILABLE", True)
    def test_no_dist_helper_when_dist_not_initialized(self):
        config = create_processor_config()
        processor = DynamicCacheQuantProcessor(self.model, config)
        processor.adapter = MagicMock()
        processor.adapter.inject_kvcache_placeholders.return_value = {}
        request = MagicMock()
        request.name = "model"
        request.module = self.model
        processor.preprocess(request)
        self.assertIsNone(processor.dist_helper)


class TestQuantizerCreation(unittest.TestCase):
    def setUp(self):
        self.model = create_simple_model()

    @patch("msmodelslim.core.quantization_algorithm.quant.kvcache.processor.DYNAMIC_AVAILABLE", True)
    def test_only_for_static_branch(self):
        details = DynamicCacheDetails(
            k_qconfig=QConfig(dtype=QDType.MXFP8, scope=QScope.PER_TOKEN, symmetric=True, method="minmax"),
            v_qconfig=QConfig(dtype=QDType.FP8_E4M3, scope=QScope.PER_CHANNEL, symmetric=True, method="minmax"),
        )
        config = DynamicCacheProcessorConfig(details=details)
        processor = DynamicCacheQuantProcessor(self.model, config)
        processor._create_quantizer("model.layers.0.self_attn")
        self.assertIsNone(processor.cache_quantizers["model.layers.0.self_attn"]["key_states"])
        self.assertIsNotNone(processor.cache_quantizers["model.layers.0.self_attn"]["value_states"])

    @patch("msmodelslim.core.quantization_algorithm.quant.kvcache.processor.dist.is_initialized", return_value=True)
    @patch("msmodelslim.core.quantization_algorithm.quant.kvcache.processor.DYNAMIC_AVAILABLE", True)
    def test_enables_sync_for_shared_layer(self, *_):
        mock_dist = MagicMock()
        mock_dist.is_shared.return_value = True
        details = DynamicCacheDetails(
            v_qconfig=QConfig(dtype=QDType.FP8_E4M3, scope=QScope.PER_CHANNEL, symmetric=True, method="minmax"),
        )
        config = DynamicCacheProcessorConfig(details=details)
        processor = DynamicCacheQuantProcessor(self.model, config)
        processor.dist_helper = mock_dist
        processor._create_quantizer("model.layers.0.self_attn")
        quantizer = processor.cache_quantizers["model.layers.0.self_attn"]["value_states"]
        self.assertIsNotNone(quantizer)
        self.assertTrue(quantizer.input_quantizer.sync)

    @patch("msmodelslim.core.quantization_algorithm.quant.kvcache.processor.dist.is_initialized", return_value=True)
    @patch("msmodelslim.core.quantization_algorithm.quant.kvcache.processor.DYNAMIC_AVAILABLE", True)
    def test_no_sync_for_local_only_layer(self, *_):
        mock_dist = MagicMock()
        mock_dist.is_shared.return_value = False
        details = DynamicCacheDetails(
            v_qconfig=QConfig(dtype=QDType.FP8_E4M3, scope=QScope.PER_CHANNEL, symmetric=True, method="minmax"),
        )
        config = DynamicCacheProcessorConfig(details=details)
        processor = DynamicCacheQuantProcessor(self.model, config)
        processor.dist_helper = mock_dist
        processor._create_quantizer("model.layers.0.self_attn")
        quantizer = processor.cache_quantizers["model.layers.0.self_attn"]["value_states"]
        self.assertIsNotNone(quantizer)
        self.assertFalse(quantizer.input_quantizer.sync)


@pytest.fixture(name="calibration_case")
def make_calibration_case(monkeypatch):
    """隔离 hook 注册表，并提供两层模型及真实 KV 输入。"""
    manager = hook_utils.HookManager()
    hijacker = function_hijacker.FunctionHijacker()
    monkeypatch.setattr(hook_utils, "_manager", manager)
    monkeypatch.setattr(function_hijacker, "_hijacker", hijacker)
    cache_type, update_name = HOOK_TARGET
    original_update = getattr(cache_type, update_name)
    monkeypatch.setattr(cache_type, update_name, original_update)

    model = create_simple_model()
    attention_layers = {f"layers.{idx}.self_attn": layer.self_attn for idx, layer in enumerate(model.layers)}
    adapter = MagicMock()
    adapter.inject_kvcache_placeholders.side_effect = lambda _name, _module, selected, _factory: {
        name: module for name, module in attention_layers.items() if selected(name)
    }
    request = BatchProcessRequest(name="", module=model)

    def make_processor(config=None):
        if config is None:
            config = DynamicCacheProcessorConfig(qconfig=create_basic_qconfig())
        return DynamicCacheQuantProcessor(model, config, adapter=adapter)

    keys = torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 4) / 8
    try:
        yield SimpleNamespace(
            make_processor=make_processor,
            model=model,
            request=request,
            cache_type=cache_type,
            original_update=original_update,
            keys=keys,
            values=-keys - 0.125,
        )
    finally:
        manager.restore_all()
        hijacker.restore_all()


class TestDynamicCacheCalibration:
    """验证真实 cache update 首次调用的统计、部署和恢复行为。"""

    @pytest.mark.parametrize(
        "dtype,symmetric",
        [(QDType.INT8, True), (QDType.INT8, False), (QDType.FP8_E4M3, True), (QDType.FP4_E2M1, True)],
    )
    @pytest.mark.parametrize("keyword_args", [False, True], ids=["positional", "keyword"])
    def test_update_deploys_layer0_scales_when_first_call_is_static(
        self, calibration_case, dtype, symmetric, keyword_args
    ):
        """场景：静态首层首次通过类级替换调用。预期：原张量返回、cache 为空且可部署有效 scale。"""
        case = calibration_case
        processor = case.make_processor(
            DynamicCacheProcessorConfig(qconfig=create_basic_qconfig(dtype, symmetric=symmetric))
        )
        processor.pre_run()
        processor.preprocess(case.request)
        cache = case.cache_type()
        if keyword_args:
            result = cache.update(key_states=case.keys, value_states=case.values, layer_idx=0)
        else:
            result = cache.update(case.keys, case.values, 0)
        assert cache.get_seq_length(0) == 0
        torch.testing.assert_close(result[0], case.keys, rtol=0, atol=0)
        torch.testing.assert_close(result[1], case.values, rtol=0, atol=0)
        processor.postprocess(case.request)

        for branch in ("key_states", "value_states"):
            ir = getattr(case.model.layers[0].self_attn, f"{branch}_quantizer")
            assert isinstance(ir, AutoFakeQuantDynamicCache)
            assert ir.x_q_scheme == processor.branch_qconfig(branch).to_scheme()
            assert ir.kv_cache_scale.numel() == 8
            assert torch.isfinite(ir.kv_cache_scale).all()
            assert (ir.kv_cache_scale > 0).all()
            assert not torch.equal(ir.kv_cache_scale, torch.ones_like(ir.kv_cache_scale))

    def test_update_collects_each_layer_when_cache_instances_change(self, calibration_case):
        """场景：多个 cache 实例各调用两层。预期：每次均不累积 cache，两层均部署真实 IR。"""
        case = calibration_case
        processor = case.make_processor()
        processor.pre_run()
        processor.preprocess(case.request)
        for _ in range(2):
            cache = case.cache_type()
            for layer_idx in range(2):
                cache.update(case.keys, case.values, layer_idx)
                assert cache.get_seq_length(layer_idx) == 0
        processor.postprocess(case.request)
        for layer in case.model.layers:
            assert isinstance(layer.self_attn.key_states_quantizer, AutoFakeQuantDynamicCache)
            assert isinstance(layer.self_attn.value_states_quantizer, AutoFakeQuantDynamicCache)

    @pytest.mark.parametrize(
        "dtype,scope",
        [
            (QDType.INT8, QScope.PER_TOKEN),
            (QDType.INT8, QScope.PER_TILE),
            (QDType.FP8_E4M3, QScope.PER_TOKEN),
            (QDType.FP8_E4M3, QScope.PER_TILE),
            (QDType.FP4_E2M1, QScope.PER_TOKEN),
            (QDType.FP4_E2M1, QScope.PER_BLOCK),
            (QDType.MXFP8, QScope.PER_TOKEN),
            (QDType.MXFP8, QScope.PER_TILE),
            (QDType.MXFP8, QScope.PER_BLOCK),
            (QDType.MXFP4, QScope.PER_BLOCK),
            (QDType.HiF4, QScope.PER_BLOCK),
        ],
    )
    def test_preprocess_preserves_layer0_scheme_when_quantization_is_data_free(self, calibration_case, dtype, scope):
        """场景：动态配置不执行校准前向。预期：仍部署各层 IR 并保留 dtype/scope，不创建统计器。"""
        case = calibration_case
        config = DynamicCacheProcessorConfig(qconfig=create_basic_qconfig(dtype, scope))
        processor = case.make_processor(config)
        assert processor.is_data_free()
        processor.pre_run()
        processor.preprocess(case.request)
        processor.postprocess(case.request)
        for idx, layer in enumerate(case.model.layers):
            for branch in ("key_states", "value_states"):
                ir = getattr(layer.self_attn, f"{branch}_quantizer")
                assert isinstance(ir, AutoFakeQuantDynamicCache)
                assert ir.x_q_scheme == config.qconfig.to_scheme()
                assert processor.cache_quantizers[f"layers.{idx}.self_attn"][branch] is None

    @pytest.mark.parametrize("static_k", [True, False], ids=["static-k", "static-v"])
    def test_update_deploys_both_branches_when_static_and_dynamic_are_mixed(self, calibration_case, static_k):
        """场景：K/V 一静一动态。预期：首次调用采集静态 scale，动态分支保留原配置。"""
        case = calibration_case
        static = create_basic_qconfig()
        dynamic = create_basic_qconfig(QDType.MXFP8, QScope.PER_BLOCK)
        details = DynamicCacheDetails(
            k_qconfig=static if static_k else dynamic,
            v_qconfig=dynamic if static_k else static,
        )
        processor = case.make_processor(DynamicCacheProcessorConfig(details=details))
        assert not processor.is_data_free()
        processor.pre_run()
        processor.preprocess(case.request)
        case.cache_type().update(case.keys, case.values, 0)
        processor.postprocess(case.request)
        for branch in ("key_states", "value_states"):
            ir = getattr(case.model.layers[0].self_attn, f"{branch}_quantizer")
            assert ir.x_q_scheme == processor.branch_qconfig(branch).to_scheme()
            if processor.branch_qconfig(branch).scope == QScope.PER_CHANNEL:
                assert ir.kv_cache_scale.numel() == 8
            else:
                assert processor.cache_quantizers["layers.0.self_attn"][branch] is None

    def test_update_leaves_unselected_layer_unquantized_when_layer0_is_excluded(self, calibration_case):
        """场景：排除首层。预期：首层透传且无量化 IR，后续层仍采集并部署。"""
        case = calibration_case
        config = DynamicCacheProcessorConfig(qconfig=create_basic_qconfig(), exclude=["layers.0.self_attn"])
        processor = case.make_processor(config)
        processor.pre_run()
        processor.preprocess(case.request)
        cache = case.cache_type()
        for idx in range(2):
            result = cache.update(case.keys, case.values, idx)
            torch.testing.assert_close(result[0], case.keys, rtol=0, atol=0)
            torch.testing.assert_close(result[1], case.values, rtol=0, atol=0)
            assert cache.get_seq_length(idx) == 0
        processor.postprocess(case.request)
        assert not hasattr(case.model.layers[0].self_attn, "key_states_quantizer")
        assert isinstance(case.model.layers[1].self_attn.key_states_quantizer, AutoFakeQuantDynamicCache)

    def test_update_keeps_cache_empty_when_other_processor_runs_before_preprocess(self, calibration_case):
        """场景：其他 Processor 在 KV preprocess 前调用 update。预期：cache 仍为空，后续首层正常部署。"""
        case = calibration_case
        processor = case.make_processor()
        processor.pre_run()
        cache = case.cache_type()
        cache.update(case.keys, case.values, 0)
        assert cache.get_seq_length(0) == 0
        processor.preprocess(case.request)
        cache.update(case.keys, case.values, 0)
        processor.postprocess(case.request)
        assert isinstance(case.model.layers[0].self_attn.key_states_quantizer, AutoFakeQuantDynamicCache)

    @pytest.mark.parametrize(
        "missing,message",
        [("layer_idx", "layer index"), ("key_states", "key_states"), ("value_states", "value_states")],
    )
    def test_update_raises_unsupported_error_when_required_input_is_missing(self, calibration_case, missing, message):
        """场景：归一化参数缺少层号或 K/V。预期：明确报错，不静默生成不完整参数。"""
        case = calibration_case
        processor = case.make_processor()
        processor.pre_run()
        processor.preprocess(case.request)
        kwargs = {"key_states": case.keys, "value_states": case.values, "layer_idx": 0}
        kwargs[missing] = None
        with pytest.raises(UnsupportedError, match=message):
            case.cache_type().update(**kwargs)

    def test_post_run_restores_cache_writes_when_calibration_finishes(self, calibration_case):
        """场景：校准结束并移除后续伪量化 hook。预期：原 update 恢复，可正常追加缓存。"""
        case = calibration_case
        processor = case.make_processor()
        processor.pre_run()
        processor.preprocess(case.request)
        case.cache_type().update(case.keys, case.values, 0)
        processor.postprocess(case.request)
        processor.post_run()
        assert function_hijacker.get_original_function(HOOK_TARGET) is None
        hook_utils.restore_target(HOOK_TARGET)
        assert getattr(*HOOK_TARGET) is case.original_update
        cache = case.cache_type()
        cache.update(case.keys, case.values, 0)
        cache.update(case.keys, case.values, 0)
        assert cache.get_seq_length(0) == 4


if __name__ == '__main__':
    unittest.main()
