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

import importlib
import unittest
from pathlib import Path
from types import GeneratorType, SimpleNamespace
from typing import Any, Callable, Optional, cast
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.model.longcat_flash.loader import LongCatFlashAdapterLoader
from msmodelslim.model.longcat_flash.longcat_flash_mtp import (
    LongCatMultiTokenPredictor,
    LongcatFlashRMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)
from msmodelslim.processor.anti_outlier.iter_smooth import IterSmoothProcessorConfig
from msmodelslim.processor.quant.linear import LinearProcessorConfig
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.security import yaml_safe_load


def _load_longcat_flash_adapter_class():
    module_path, class_name = LongCatFlashAdapterLoader.ADAPTER_CLASS_PATH.split(":", 1)
    return getattr(importlib.import_module(module_path), class_name)


LongCatFlashModelAdapter = _load_longcat_flash_adapter_class()
LongCatFlashModelAdapterModule = importlib.import_module(LongCatFlashAdapterLoader.ADAPTER_CLASS_PATH.split(":", 1)[0])


def _invoke_hook(hook: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return hook(*args, **kwargs)


class DummyConfig:
    """Mock configuration object for LongCat-Flash"""

    def __init__(self, num_layers=28):
        self.hidden_size = 7168
        self.num_attention_heads = 128
        self.num_key_value_heads = 128
        self.num_layers = num_layers
        self.num_hidden_layers = 56
        self.q_lora_rank = 1536
        self.kv_lora_rank = 512
        self.qk_nope_head_dim = 128
        self.qk_rope_head_dim = 64
        self.v_head_dim = 128
        self.rms_norm_eps = 1e-6
        self.rope_theta = 10000.0
        self.ffn_hidden_size = 18432
        self.hidden_act = "silu"
        self.n_routed_experts = 512
        self.use_cache = True
        self._attn_implementation = None
        self.vocab_size = 129280


class TinyLongCatConfig(SimpleNamespace):
    def __init__(self, **kwargs):
        defaults = {
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "q_lora_rank": 4,
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "v_head_dim": 2,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "ffn_hidden_size": 16,
            "hidden_act": "silu",
            "vocab_size": 32,
            "attention_bias": False,
            "mla_scale_q_lora": False,
            "mla_scale_kv_lora": False,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


class TestLongCatFlashModelAdapter(unittest.TestCase):
    """
    Unit tests for LongCat-Flash model adapter, and we only test the interfaces here.
    """

    def setUp(self):
        self.model_type = 'LongCat-Flash-Chat'
        self.model_path = Path('.')
        self.repo_root = Path(__file__).resolve().parents[4]
        self.quant_config_path = self.repo_root / 'lab_practice/longcat_flash/longcat_flash_w4a4_mxfp4.yaml'

    def test_should_return_model_type_when_get_model_type_called_given_initialized_adapter(self):
        """Test get_model_type method returns correct model type."""
        # given
        with patch.object(LongCatFlashModelAdapter, '__init__', return_value=None):
            adapter = LongCatFlashModelAdapter()
            adapter.model_type = self.model_type

            # when
            result = adapter.get_model_type()

            # then
            self.assertEqual(result, self.model_type)

    def test_should_return_longcat_flash_when_get_model_pedigree_called_given_adapter(self):
        """Test get_model_pedigree returns 'longcat_flash'"""
        # given
        with patch.object(LongCatFlashModelAdapter, '__init__', return_value=None):
            adapter = LongCatFlashModelAdapter()

            # when
            result = adapter.get_model_pedigree()

            # then
            self.assertEqual(result, 'longcat_flash')

    def test_should_return_tokenized_data_when_handle_dataset_called_given_dataset(self):
        """Test handle_dataset delegates to _get_tokenized_data."""
        # given
        with patch.object(LongCatFlashModelAdapter, '__init__', return_value=None):
            adapter = LongCatFlashModelAdapter()
            mock_data = [[3, 2, 9], [1, 2, 3]]
            adapter._get_tokenized_data = MagicMock(return_value=mock_data)

            # when
            result = adapter.handle_dataset(dataset='test_data', device=DeviceType.CPU)

            # then
            self.assertEqual(result, mock_data)

    def test_should_return_expected_subgraph_count_when_adapter_config_requested_given_longcat_config(self):
        """Test get_adapter_config_for_subgraph returns a list of AdapterConfig."""
        # given
        with patch.object(LongCatFlashModelAdapter, '__init__', return_value=None):
            adapter = LongCatFlashModelAdapter()
            adapter.config = DummyConfig()

            # when
            result = adapter.get_adapter_config_for_subgraph()

            # then
            self.assertIsInstance(result, list)
            # Each layer has: 6 attention/norm configs + 4 dense MLP configs + routed expert up/down configs.
            expect_count = adapter.config.num_layers * (10 + adapter.config.n_routed_experts)
            self.assertEqual(len(result), expect_count)

    def test_should_validate_processors_when_config_loaded_given_longcat_flash_w4a4_mxfp4_yaml(self):
        """Test the LongCat Flash yaml processor config validates through public config APIs."""
        # given
        config = yaml_safe_load(str(self.quant_config_path))
        processor_config_map = {
            'iter_smooth': IterSmoothProcessorConfig,
            'linear_quant': LinearProcessorConfig,
        }

        # when
        processor_configs = [
            processor_config_map[process['type']].model_validate(process) for process in config['spec']['process']
        ]

        # then
        self.assertEqual(len(processor_configs), 3)
        self.assertIsInstance(processor_configs[0], IterSmoothProcessorConfig)
        self.assertIsInstance(processor_configs[1], LinearProcessorConfig)
        self.assertIsInstance(processor_configs[2], LinearProcessorConfig)
        self.assertEqual(processor_configs[1].qconfig.act.dtype.value, 'mxfp4')
        self.assertEqual(processor_configs[1].qconfig.weight.dtype.value, 'mxfp4')
        self.assertEqual(processor_configs[2].qconfig.act.dtype.value, 'mxfp8')
        self.assertEqual(processor_configs[2].qconfig.weight.dtype.value, 'mxfp8')

    def test_should_cover_longcat_smooth_sources_when_adapter_config_checked_given_w4a4_mxfp4_yaml(self):
        """Test IterSmooth yaml patterns cover LongCat adapter sources through public adapter API."""
        # given
        config = yaml_safe_load(str(self.quant_config_path))
        iter_smooth_config = IterSmoothProcessorConfig.model_validate(config['spec']['process'][0])
        with patch.object(LongCatFlashModelAdapter, '__init__', return_value=None):
            adapter = LongCatFlashModelAdapter()
            adapter.config = DummyConfig(num_layers=1)

            # when
            adapter_configs = adapter.get_adapter_config_for_subgraph()

        enabled_adapter_sources = {
            adapter_config.mapping.source
            for adapter_config in adapter_configs
            if adapter_config.subgraph_type in iter_smooth_config.enable_subgraph_type
        }
        expected_sources = {
            'model.layers.0.input_layernorm.0',
            'model.layers.0.self_attn.0.q_a_layernorm',
            'model.layers.0.mlp.experts.0.up_proj',
        }

        # then
        self.assertTrue(expected_sources.issubset(enabled_adapter_sources))
        self.assertIn('norm-linear', iter_smooth_config.enable_subgraph_type)
        self.assertIn('up-down', iter_smooth_config.enable_subgraph_type)
        self.assertIn('*input_layernorm*', iter_smooth_config.include)
        self.assertIn('*q_a_layernorm*', iter_smooth_config.include)
        self.assertIn('*mlp.experts*', iter_smooth_config.include)

    def test_should_yield_decoder_and_mtp_blocks_when_generate_model_visit_called_given_three_layers(self):
        """Test generate_model_visit returns a generator."""
        # given
        with patch.object(LongCatFlashModelAdapter, '__init__', return_value=None):
            adapter = LongCatFlashModelAdapter()
            adapter.config = DummyConfig(num_layers=3)  # Minimal for test
            adapter.model_path = self.model_path
            adapter._get_state_dict = MagicMock(return_value={})

            # Create mock model with proper structure
            mock_model = MagicMock()
            mock_layers = MagicMock()
            mock_model.model.layers = mock_layers

            # Mock layers that would be created/retrieved
            mock_decoder = MagicMock()
            mock_mtp = MagicMock()

            # Setup get_submodule to simulate layers not existing (will be created)
            def get_submodule_side_effect(name):
                if name in ('model.layers.0', 'model.layers.1', 'model.mtp'):
                    raise AttributeError()  # Layer does not exist
                else:
                    return AttributeError(f"unknown module: {name}")

            mock_model.get_submodule = MagicMock(side_effect=get_submodule_side_effect)
            mock_layers.__getitem__ = MagicMock(return_value=mock_decoder)  # Template layer

            # Patch the MTP import
            with patch.object(LongCatFlashModelAdapterModule, 'LongCatMultiTokenPredictor', return_value=mock_mtp):
                with patch.object(
                    LongCatFlashModelAdapterModule, 'generated_decoder_layer_visit_func'
                ) as mock_visit_func:
                    mock_visit_func.return_value = iter([])

                    # when
                    result = adapter.generate_model_visit(mock_model)

                    # then
                    # Verify it returns an iterator
                    self.assertTrue(hasattr(result, '__iter__'))

                    # Verify generate_decoder_layer_visit_func was called
                    mock_visit_func.assert_called_once()
                    call_args, call_kwargs = mock_visit_func.call_args

                    # First arg should be the model
                    self.assertIs(call_args[0], mock_model)

                    # transformer_blocks should be a generator
                    transformer_blocks = call_kwargs['transformer_blocks']
                    self.assertTrue(hasattr(transformer_blocks, '__iter__'))

                    # Consume the generator to verify it yields correct structure
                    blocks = list(transformer_blocks)

                    # Should have 3 blocks (2 layers + 1 mtp)
                    self.assertEqual(len(blocks), 3)

                    # Verify block names
                    self.assertEqual(blocks[0][0], 'model.layers.0')
                    self.assertEqual(blocks[0][1].layer_idx, 0)
                    self.assertEqual(blocks[1][0], 'model.layers.1')
                    self.assertEqual(blocks[1][1].layer_idx, 1)
                    self.assertEqual(blocks[2][0], 'model.mtp')
                    self.assertIs(blocks[2][1], mock_mtp)

    def test_should_generate_decoder_and_mtp_requests_when_generate_model_forward_called_given_inputs(self):
        """Test generate_model_forward returns returns a generator."""
        # given
        with patch.object(LongCatFlashModelAdapter, '__init__', return_value=None):
            adapter = LongCatFlashModelAdapter()
            adapter.config = DummyConfig(num_layers=3)

            mock_model = MagicMock()
            template_decoder = MagicMock()
            mock_layers = MagicMock()
            mock_model.model.layers = mock_layers
            mock_layers.__getitem__ = MagicMock(return_value=template_decoder)

            # Capture the hook function when registered
            captured_hook: Optional[Callable[..., Any]] = None
            mock_handler = MagicMock()

            def capture_hook(hook, **kwargs):
                nonlocal captured_hook
                captured_hook = hook
                return mock_handler

            template_decoder.register_forward_pre_hook.side_effect = capture_hook

            # Make model call trigger the captured hook with TransformersForwardBreak
            mock_hidden_states = torch.randn(1, 10, 7168)
            mock_attention_mask = torch.ones(1, 1, 10, 10)
            mock_position_embeddings = (torch.randn(1, 10, 64), torch.randn(1, 10, 64))

            def model_call(*args, **kwargs):
                if captured_hook:
                    mock_args = (mock_hidden_states,)
                    mock_kwargs = {
                        'attention_mask': mock_attention_mask,
                        'position_embeddings': mock_position_embeddings,
                    }
                    captured_hook(mock_model, mock_args, mock_kwargs)

            mock_model.side_effect = model_call

            # Mock decoder layers that would be created by _load_decoder_if_not exist
            mock_decoder_0 = MagicMock()
            mock_decoder_1 = MagicMock()
            mock_mtp = MagicMock()

            def get_submodule_side_effect(name):
                if name == 'model.layers.0':
                    return mock_decoder_0
                elif name == 'model.layers.1':
                    return mock_decoder_1
                elif name == 'model.mtp':
                    return mock_mtp
                else:
                    raise AttributeError(f"unknown module: {name}")

            mock_model.get_submodule = MagicMock(side_effect=get_submodule_side_effect)

            # Setup inputs
            input_ids = [torch.randn(1, 10)]

            # when
            result = adapter.generate_model_forward(mock_model, input_ids)

            # then
            self.assertIsInstance(result, GeneratorType)

            # Get first ProcessRequest - this triggers hook registration
            request_0 = next(result)

            # Verify hook was registered and removed
            template_decoder.register_forward_pre_hook.assert_called_once()
            mock_handler.remove.assert_called_once()

            # Verify first request is for decoder layer 0
            self.assertIsInstance(request_0, ProcessRequest)
            self.assertEqual(request_0.name, 'model.layers.0')
            self.assertEqual(request_0.module, mock_decoder_0)

            # Send hidden states back and get next request
            request_1 = result.send(mock_hidden_states)

            # Verify second request is for decoder layer 1
            self.assertIsInstance(request_1, ProcessRequest)
            self.assertEqual(request_1.name, 'model.layers.1')
            self.assertEqual(request_1.module, mock_decoder_1)

            # Send hidden states back and get MTP request
            request_mtp = result.send(mock_hidden_states)

            # Verify MTP request
            self.assertIsInstance(request_mtp, ProcessRequest)
            self.assertEqual(request_mtp.name, 'model.mtp')
            self.assertEqual(request_mtp.module, mock_mtp)

            # Verify MTP receives special args: (input_ids, prev_hidden_states, position_embeddings, attention_mask)
            self.assertEqual(len(request_mtp.args), 4)
            torch.testing.assert_close(request_mtp.args[0], input_ids[0])
            torch.testing.assert_close(request_mtp.args[1], mock_hidden_states)
            self.assertEqual(request_mtp.args[2], mock_position_embeddings)
            torch.testing.assert_close(request_mtp.args[3], mock_attention_mask)
            self.assertEqual(request_mtp.kwargs, {})

            # Verify generator exhausts after MTP
            with self.assertRaises(StopIteration):
                result.send(mock_hidden_states)

    def test_should_initialize_one_layer_template_when_init_model_called_given_longcat_config(self):
        """
        Test init_model interface.

        1. test the SafeGenerator.get_model_from_pretrained with 1 layer and CPU device.
        2. test the final config layers is origin num_layers + 1 (for MTP).
        """
        # given
        with patch.object(LongCatFlashModelAdapter, '__init__', return_value=None):
            adapter = LongCatFlashModelAdapter()
            original_num_layers = 28
            adapter.config = DummyConfig(num_layers=original_num_layers)
            adapter.model_path = self.model_path
            adapter.trust_remote_code = True

            mock_model = MagicMock()

            cap_num_layers = None

            def capture_num_layers(*args, **kwargs):
                nonlocal cap_num_layers
                config = kwargs.get('config')
                if config:
                    cap_num_layers = config.num_layers
                return mock_model

            with patch.object(
                LongCatFlashModelAdapterModule.SafeGenerator,
                'get_model_from_pretrained',
                side_effect=capture_num_layers,
            ) as mock_get_model:
                with patch.object(adapter, '_get_state_dict', return_value={}):
                    # when
                    adapter.init_model(device=DeviceType.CPU)

            # then
            mock_get_model.assert_called_once()
            call_kwargs = mock_get_model.call_args.kwargs
            self.assertEqual(call_kwargs.get('model_path'), str(self.model_path))
            self.assertEqual(call_kwargs.get('device_map'), 'cpu')
            self.assertEqual(call_kwargs.get('torch_dtype'), 'auto')
            self.assertNotIn('attn_implementation', call_kwargs)
            self.assertEqual(cap_num_layers, 1)
            self.assertTrue(call_kwargs.get('trust_remote_code'))
            self.assertFalse(adapter.config.use_cache)
            self.assertEqual(adapter.config._attn_implementation, 'sdpa')

            # After loading, num_layers should be original + 1
            self.assertEqual(adapter.config.num_layers, original_num_layers + 1)

            mock_model.load_state_dict.assert_called_once()
            mock_model.eval.assert_called_once()

    def test_should_raise_invalid_model_error_when_init_model_called_given_config_without_num_layers(self):
        with patch.object(LongCatFlashModelAdapter, '__init__', return_value=None):
            adapter = LongCatFlashModelAdapter()
            adapter.config = SimpleNamespace()
            adapter.model_path = self.model_path
            adapter.trust_remote_code = False

            with self.assertRaises(InvalidModelError):
                adapter.init_model(device=DeviceType.CPU)

    def test_should_keep_non_eager_attention_impl_when_init_model_called_given_existing_sdpa(self):
        with patch.object(LongCatFlashModelAdapter, '__init__', return_value=None):
            adapter = LongCatFlashModelAdapter()
            adapter.config = DummyConfig(num_layers=2)
            adapter.config._attn_implementation = 'flash_attention_2'
            adapter.model_path = self.model_path
            adapter.trust_remote_code = False

            mock_model = MagicMock()

            with patch.object(
                LongCatFlashModelAdapterModule.SafeGenerator,
                'get_model_from_pretrained',
                return_value=mock_model,
            ):
                with patch.object(adapter, '_get_state_dict', return_value={}):
                    adapter.init_model(device=DeviceType.CPU)

            self.assertEqual(adapter.config._attn_implementation, 'flash_attention_2')

    def test_should_yield_process_request_when_generate_model_forward_called_given_dict_inputs(self):
        with patch.object(LongCatFlashModelAdapter, '__init__', return_value=None):
            adapter = LongCatFlashModelAdapter()
            adapter.config = DummyConfig(num_layers=1)

            mock_model = MagicMock()
            first_layer = MagicMock()
            mock_model.model.layers = [first_layer]
            mock_handler = MagicMock()
            captured_hook = None

            def capture_hook(hook, **kwargs):
                nonlocal captured_hook
                captured_hook = hook
                return mock_handler

            first_layer.register_forward_pre_hook.side_effect = capture_hook

            hidden_states = torch.randn(1, 3, 8)
            attention_mask = torch.zeros(1, 1, 3, 3)
            position_embeddings = (torch.ones(1, 3, 2), torch.zeros(1, 3, 2))

            def model_call(**kwargs):
                if captured_hook is None:
                    self.fail("forward pre hook was not captured")
                hook = cast(Callable[..., Any], captured_hook)
                _invoke_hook(
                    hook,
                    mock_model,
                    (hidden_states,),
                    {
                        'attention_mask': attention_mask,
                        'position_embeddings': position_embeddings,
                    },
                )

            mock_model.side_effect = model_call
            mock_mtp = MagicMock()
            mock_model.get_submodule.side_effect = lambda name: mock_mtp

            inputs = {'input_ids': torch.tensor([[1, 2, 3]])}

            result = adapter.generate_model_forward(mock_model, inputs)
            request = next(result)

            self.assertEqual(request.name, 'model.mtp')
            torch.testing.assert_close(request.args[0], inputs['input_ids'])
            mock_handler.remove.assert_called_once()

    def test_should_raise_invalid_model_error_when_generate_model_forward_called_without_first_block_input(self):
        with patch.object(LongCatFlashModelAdapter, '__init__', return_value=None):
            adapter = LongCatFlashModelAdapter()
            adapter.config = DummyConfig(num_layers=1)

            mock_model = MagicMock()
            first_layer = MagicMock()
            first_layer.register_forward_pre_hook.return_value = MagicMock()
            mock_model.model.layers = [first_layer]

            with self.assertRaises(InvalidModelError):
                next(adapter.generate_model_forward(mock_model, [torch.tensor([[1, 2]])]))

    def test_should_reraise_runtime_error_when_generate_model_forward_called_given_model_failure(self):
        with patch.object(LongCatFlashModelAdapter, '__init__', return_value=None):
            adapter = LongCatFlashModelAdapter()
            adapter.config = DummyConfig(num_layers=1)

            mock_model = MagicMock()
            first_layer = MagicMock()
            first_layer.register_forward_pre_hook.return_value = MagicMock()
            mock_model.model.layers = [first_layer]
            mock_model.side_effect = RuntimeError('boom')

            with self.assertRaisesRegex(RuntimeError, 'boom'):
                next(adapter.generate_model_forward(mock_model, [torch.tensor([[1, 2]])]))

    def test_should_inject_kv_cache_flag_when_enable_kv_cache_called_given_model(self):
        with patch.object(LongCatFlashModelAdapter, '__init__', return_value=None):
            adapter = LongCatFlashModelAdapter()
            mock_model = MagicMock()
            captured_hook: Optional[Callable[..., Any]] = None

            def capture_hook(hook, **kwargs):
                nonlocal captured_hook
                captured_hook = hook
                return MagicMock()

            mock_model.model.register_forward_pre_hook.side_effect = capture_hook

            adapter.enable_kv_cache(mock_model, need_kv_cache=True)

            if captured_hook is None:
                self.fail("kv cache hook was not captured")
            hook = cast(Callable[..., Any], captured_hook)
            args, kwargs = _invoke_hook(hook, mock_model, (torch.tensor([1]),), {})
            self.assertEqual(args, (torch.tensor([1]),))
            self.assertTrue(kwargs['need_kv_cache'])

    def test_should_return_same_module_when_ascendv1_preprocess_called_given_non_router(self):
        with patch.object(LongCatFlashModelAdapter, '__init__', return_value=None):
            adapter = LongCatFlashModelAdapter()
            module = nn.Linear(2, 2)
            model = MagicMock()

            prefix, returned_module = adapter.ascendv1_save_module_preprocess('model.linear', module, model)

            self.assertEqual(prefix, 'model.linear')
            self.assertIs(returned_module, module)
            model.set_submodule.assert_not_called()

    def test_should_cast_and_register_bias_when_ascendv1_preprocess_called_given_router_module(self):
        class LongcatFlashTopkRouter(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
                self.e_score_correction_bias = torch.ones(2, dtype=torch.bfloat16)

        with patch.object(LongCatFlashModelAdapter, '__init__', return_value=None):
            adapter = LongCatFlashModelAdapter()
            router = LongcatFlashTopkRouter()
            model = MagicMock()

            prefix, new_module = adapter.ascendv1_save_module_preprocess('model.router', router, model)

            self.assertEqual(prefix, 'model.router')
            self.assertIsNot(new_module, router)
            self.assertEqual(new_module.weight.dtype, torch.float32)
            self.assertIn('e_score_correction_bias', dict(new_module.named_parameters(recurse=False)))
            model.set_submodule.assert_called_once_with('model.router', new_module)


class TestLongCatFlashMTPPublicApi(unittest.TestCase):
    def test_should_return_same_dtype_and_shape_when_rmsnorm_forward_called_given_half_precision_input(self):
        module = LongcatFlashRMSNorm(hidden_size=4, eps=1e-5)
        hidden_states = torch.randn(2, 3, 4, dtype=torch.float16)

        output = module(hidden_states)

        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(tuple(output.shape), (2, 3, 4))
        self.assertIn('eps=1e-05', module.extra_repr())

    def test_should_repeat_key_values_when_repeat_kv_called_given_multiple_groups(self):
        hidden_states = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 4)

        repeated = repeat_kv(hidden_states, n_rep=2)
        same = repeat_kv(hidden_states, n_rep=1)

        self.assertEqual(tuple(repeated.shape), (1, 2, 2, 4))
        self.assertTrue(torch.equal(same, hidden_states))

    def test_should_apply_rotary_embedding_when_called_given_mla_and_non_mla_inputs(self):
        q = torch.randn(1, 2, 3, 2)
        k = torch.randn(1, 2, 3, 2)
        cos = torch.ones(1, 3, 2)
        sin = torch.zeros(1, 3, 2)

        q_plain, k_plain = apply_rotary_pos_emb(q, k, cos, sin, use_mla=False)
        q_mla, k_mla = apply_rotary_pos_emb(q, k, cos, sin, use_mla=True)

        torch.testing.assert_close(q_plain, q)
        torch.testing.assert_close(k_plain, k)
        self.assertEqual(q_mla.shape, q.shape)
        self.assertEqual(k_mla.shape, k.shape)

    def test_should_return_attention_output_when_eager_attention_called_given_mask(self):
        module = SimpleNamespace(num_key_value_groups=2)
        query = torch.randn(1, 2, 3, 4)
        key = torch.randn(1, 1, 3, 4)
        value = torch.randn(1, 1, 3, 2)
        attention_mask = torch.zeros(1, 1, 3, 3)

        masked_output, masked_weights = eager_attention_forward(module, query, key, value, attention_mask, scaling=0.5)
        plain_output, plain_weights = eager_attention_forward(module, query, key, value, None, scaling=0.5)

        self.assertEqual(tuple(masked_output.shape), (1, 3, 2, 2))
        self.assertEqual(tuple(masked_weights.shape), (1, 2, 3, 3))
        self.assertEqual(tuple(plain_output.shape), (1, 3, 2, 2))
        self.assertEqual(tuple(plain_weights.shape), (1, 2, 3, 3))

    def test_should_run_predictor_forward_when_called_given_scaled_config_and_mask(self):
        torch.manual_seed(0)
        config = TinyLongCatConfig(
            num_key_value_heads=2,
            mla_scale_q_lora=True,
            mla_scale_kv_lora=True,
        )
        predictor = LongCatMultiTokenPredictor(config)
        input_ids = torch.tensor([[1, 2, 3]])
        previous_hidden_states = torch.randn(1, 3, config.hidden_size)
        position_embeddings = (
            torch.ones(1, 3, config.qk_rope_head_dim),
            torch.zeros(1, 3, config.qk_rope_head_dim),
        )
        attention_mask = torch.zeros(1, 1, 3, 3)

        hidden_states = predictor(
            input_ids=input_ids,
            previous_hidden_states=previous_hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )

        self.assertEqual(tuple(hidden_states.shape), (1, 3, config.hidden_size))
        self.assertEqual(hidden_states.dtype, previous_hidden_states.dtype)

    def test_should_run_predictor_forward_when_called_given_equal_kv_heads_without_mask(self):
        torch.manual_seed(1)
        config = TinyLongCatConfig(num_key_value_heads=2)
        predictor = LongCatMultiTokenPredictor(config)
        input_ids = torch.tensor([[4, 5]])
        previous_hidden_states = torch.randn(1, 2, config.hidden_size)
        position_embeddings = (
            torch.ones(1, 2, config.qk_rope_head_dim),
            torch.zeros(1, 2, config.qk_rope_head_dim),
        )

        hidden_states = predictor(
            input_ids=input_ids,
            previous_hidden_states=previous_hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=None,
        )

        self.assertEqual(tuple(hidden_states.shape), (1, 2, config.hidden_size))
