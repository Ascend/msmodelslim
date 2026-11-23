#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# pylint: disable=too-many-lines

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
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

import pytest
import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.quantization_schema import AutoFakeQuantActivation
from msmodelslim.core.quantization_schema.const import hif8_per_tensor_sym, int8_per_token_sym
from msmodelslim.core.quantization_schema.qal import QParam
from msmodelslim.model.deepseek_v4.model_adapter import DeepSeekV4ModelAdapter
from msmodelslim.model.deepseek_v4.model import Attention, Compressor, Indexer, ModelArgs
from msmodelslim.core.quantization_algorithm.quant.fa3 import (
    FA3QuantPlaceHolder,
    FA3QuantProcessor,
    FA3QuantProcessorConfig,
)


class DummyConfig:
    def __init__(self):
        self.num_hidden_layers = 2
        self.n_routed_experts = 2
        self.n_shared_experts = 1
        self.dim = 128
        self.q_lora_rank = 4
        self.compress_ratios = (1, 1)
        self.norm_eps = 1e-6


class DummyDecoderLayerV4(nn.Module):
    def __init__(self, layer_id=0, args=None):
        super().__init__()
        self.layer_id = layer_id
        self.args = args
        self.linear = nn.Linear(1, 1)
        # attributes used by tests / adapter checks
        self.enorm = None
        # provide a get_submodule helper so tests don't need to monkeypatch it

    def get_submodule(self, name):
        value = getattr(self, name)
        if value is None:
            raise AttributeError(f"Module '{name}' not found")
        return value

    def forward(self, x):
        return self.linear(x)


class DummyModelV4(nn.Module):
    def __init__(self, config=None, num_layers=None):
        super().__init__()
        if isinstance(config, DummyConfig):
            self.config = config
            num_layers = config.num_hidden_layers if num_layers is None else num_layers
        else:
            num_layers = config if isinstance(config, int) else 1
            self.config = None

        self.layers = nn.ModuleList([DummyDecoderLayerV4(layer_id=i, args=self.config) for i in range(num_layers)])
        self.mtp = nn.ModuleList()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TestDeepSeekV4ModelAdapter(unittest.TestCase):
    def setUp(self):
        self.model_path = Path('.')
        self.model_type = 'DeepSeek-V4'
        self.dummy_config = DummyConfig()
        self.adapter_patcher = patch.object(DeepSeekV4ModelAdapter, '__init__', lambda x, model_path, model_type: None)

    def create_adapter(self, **kwargs):
        with self.adapter_patcher:
            adapter = DeepSeekV4ModelAdapter(model_path=self.model_path, model_type=self.model_type)
            adapter.config = self.dummy_config
            adapter.model_path = self.model_path
            for key, value in kwargs.items():
                setattr(adapter, key, value)
            return adapter

    # Verify the adapter reports the expected model pedigree string.
    def test_model_pedigree_returns_deepseek_v4_when_adapter_created(self):
        adapter = self.create_adapter()
        self.assertEqual(adapter.get_model_pedigree(), 'deepseek_v4')

    # Verify the adapter returns the provided model type.
    def test_model_type_returns_model_type_when_adapter_type_set(self):
        adapter = self.create_adapter(model_type=self.model_type)
        self.assertEqual(adapter.get_model_type(), self.model_type)

    # Verify dataset handling delegates to the tokenizer helper.
    def test_handle_dataset_calls_tokenizer_when_handling_data(self):
        adapter = self.create_adapter()
        adapter._get_tokenized_data = Mock(return_value=['tokenized'])
        result = adapter.handle_dataset('ds', device=DeviceType.CPU)
        adapter._get_tokenized_data.assert_called_once_with('ds', DeviceType.CPU)
        self.assertEqual(result, ['tokenized'])

    @patch('msmodelslim.model.common.weight_helper.json_safe_load')
    @patch('msmodelslim.model.common.weight_helper.os.path.join')
    # Verify weight map loading via weight_helper returns the parsed mapping.
    def test_get_weight_map_returns_weight_map_when_index_json_loaded(self, mock_join, mock_json_load):
        mock_json_load.return_value = {'weight_map': {'a': 'file1', 'b': 'file2'}}
        mock_join.return_value = self.model_path / 'model.safetensors.index.json'

        from msmodelslim.model.common.weight_helper import get_weight_map

        result = get_weight_map(self.model_path)

        mock_join.assert_called_once_with(self.model_path, 'model.safetensors.index.json')
        mock_json_load.assert_called_once()
        self.assertEqual(result, {'a': 'file1', 'b': 'file2'})

    # Verify state dict loading via weight_helper reads weights from a single file.
    def test_get_state_dict_reads_single_file_when_single_file_contains_all_parameters(self):
        from msmodelslim.model.common.weight_helper import get_state_dict, get_weight_map

        get_weight_map.cache_clear()

        weight_map = {'layer.weight': 'file1.safetensors', 'layer.bias': 'file1.safetensors'}
        mock_module = Mock(spec=nn.Module)
        mock_module.named_parameters.return_value = [('layer.weight', Mock()), ('layer.bias', Mock())]

        mock_file = MagicMock()
        mock_file.get_tensor.side_effect = lambda name: f'tensor_{name}'

        with (
            patch('msmodelslim.model.common.weight_helper.get_weight_map', return_value=weight_map),
            patch('msmodelslim.model.common.weight_helper.get_valid_read_path', side_effect=lambda x, **kwargs: x),
            patch(
                'msmodelslim.model.common.weight_helper.safe_open',
                return_value=Mock(__enter__=Mock(return_value=mock_file), __exit__=Mock(return_value=False)),
            ),
        ):
            result = get_state_dict(str(self.model_path), mock_module)

        self.assertEqual(result['layer.weight'], 'tensor_layer.weight')
        self.assertEqual(result['layer.bias'], 'tensor_layer.bias')

    # Verify state dict loading works when tensors are spread across multiple files.
    def test_get_state_dict_reads_multiple_files_when_parameters_spread_across_files(self):
        from msmodelslim.model.common.weight_helper import get_state_dict, get_weight_map

        get_weight_map.cache_clear()

        weight_map = {
            'layer1.weight': 'file1.safetensors',
            'layer1.bias': 'file1.safetensors',
            'layer2.weight': 'file2.safetensors',
            'layer2.bias': 'file2.safetensors',
        }
        mock_module = Mock(spec=nn.Module)
        mock_module.named_parameters.return_value = [
            ('layer1.weight', Mock()),
            ('layer1.bias', Mock()),
            ('layer2.weight', Mock()),
            ('layer2.bias', Mock()),
        ]

        def safe_open_side_effect(path, framework, device):
            mock_file = MagicMock()
            mock_file.get_tensor.side_effect = lambda name: f'tensor_{name}'
            mock_context = MagicMock()
            mock_context.__enter__.return_value = mock_file
            mock_context.__exit__.return_value = False
            return mock_context

        with (
            patch('msmodelslim.model.common.weight_helper.get_weight_map', return_value=weight_map),
            patch('msmodelslim.model.common.weight_helper.get_valid_read_path', side_effect=lambda x, **kwargs: x),
            patch('msmodelslim.model.common.weight_helper.safe_open', side_effect=safe_open_side_effect),
        ):
            result = get_state_dict(str(self.model_path), mock_module)

        self.assertEqual(result['layer1.weight'], 'tensor_layer1.weight')
        self.assertEqual(result['layer2.bias'], 'tensor_layer2.bias')

    # Verify a missing weight file raises FileNotFoundError during state dict loading.
    def test_get_state_dict_raises_file_not_found_when_weight_file_missing(self):
        from msmodelslim.model.common.weight_helper import get_state_dict, get_weight_map

        get_weight_map.cache_clear()

        weight_map = {'layer.weight': 'missing.safetensors'}
        mock_module = Mock(spec=nn.Module)
        mock_module.named_parameters.return_value = [('layer.weight', Mock())]

        with (
            patch('msmodelslim.model.common.weight_helper.get_weight_map', return_value=weight_map),
            patch(
                'msmodelslim.model.common.weight_helper.get_valid_read_path',
                side_effect=FileNotFoundError('not found'),
            ),
        ):
            with self.assertRaises(FileNotFoundError):
                get_state_dict(str(self.model_path), mock_module)

    # Verify MTP decoder is returned when model.mtp already has it.
    def test_load_mtp_decoder_if_not_exist_creates_mtp_layer_when_decoder_missing(self):
        adapter = self.create_adapter()
        adapter.config.num_hidden_layers = 1
        adapter.config.n_mtp_layers = 1
        dummy_model = DummyModelV4(num_layers=1, config=self.dummy_config)
        dummy_model.mtp = nn.ModuleList([DummyDecoderLayerV4()])

        mtp_decoder = adapter.load_mtp_decoder_if_not_exist(dummy_model, layer_prefix='mtp.0', mtp_idx=0)

        self.assertIsNotNone(mtp_decoder)

    @patch('msmodelslim.model.deepseek_v4.model_adapter.get_state_dict')
    # Verify existing MTP decoder is not re-created when already loaded.
    def test_load_mtp_decoder_if_not_exist_skips_when_mtp_already_exists(self, mock_get_sd):
        adapter = self.create_adapter()
        adapter.config.n_mtp_layers = 1
        dummy_model = DummyModelV4(num_layers=1, config=self.dummy_config)
        dummy_model.mtp = nn.ModuleList([DummyDecoderLayerV4()])

        result = adapter.load_mtp_decoder_if_not_exist(dummy_model, layer_prefix='mtp.0', mtp_idx=0)
        self.assertIsNotNone(result)

    @patch('msmodelslim.model.deepseek_v4.model_adapter.auto_dequant_state_dict')
    # Verify the existing decoder is returned and no auto-dequant is performed when the decoder exists.
    def test_load_decoder_if_not_exist_returns_existing_decoder_when_decoder_present(self, mock_auto_dequant):
        adapter = self.create_adapter()
        dummy_model = DummyModelV4(num_layers=1, config=self.dummy_config)

        result = adapter.load_decoder_if_not_exist(model=dummy_model, layer_prefix='layers.0', idx=0)

        self.assertEqual(result, dummy_model.get_submodule('layers.0'))
        mock_auto_dequant.assert_not_called()

    # Verify decoder layer generator yields the expected layer names.
    def test_generate_decoder_layer_returns_layer_names_when_layers_generated(self):
        adapter = self.create_adapter()
        adapter.config.num_hidden_layers = 2
        mock_decoders = [DummyDecoderLayerV4(layer_id=i) for i in range(adapter.config.num_hidden_layers)]
        adapter.load_decoder_if_not_exist = Mock(side_effect=mock_decoders)

        layers = list(adapter.generate_decoder_layer(model=Mock()))

        self.assertEqual([name for name, _ in layers], ['layers.0', 'layers.1'])

    # Verify adapter config generation for subgraph uses model configuration values.
    def test_get_adapter_config_for_subgraph_returns_configs_when_config_present(self):
        adapter = self.create_adapter()
        adapter.config.num_hidden_layers = 3
        adapter.config.n_routed_experts = 2
        adapter.config.n_shared_experts = 1

        configs = adapter.get_adapter_config_for_subgraph()

        self.assertEqual(len(configs), 18)
        self.assertEqual(configs[0].mapping.source, 'layers.0.ffn.shared_experts.w3')
        self.assertEqual(configs[3].mapping.source, 'layers.0.attn.wo_a')
        # the 5th config corresponds to layer 0's norm-linear mapping targets
        self.assertIn('layers.0.attn.wq_a', configs[4].mapping.targets)

    # Verify layer norm fuse maps are returned for models with multiple layers.
    def test_get_ln_fuse_map_returns_maps_when_model_has_layers(self):
        adapter = self.create_adapter()
        adapter.config.num_hidden_layers = 2
        adapter.config.n_routed_experts = 2
        adapter.config.n_shared_experts = 1

        pre_ln_map, ln_map = adapter.get_ln_fuse_map()

        self.assertEqual(pre_ln_map, {'norm': ['head']})
        self.assertIn('layers.0.ffn_norm', ln_map)

    @patch('msmodelslim.model.deepseek_v4.model_adapter.QuaRotInterface.get_rotate_command')
    # Verify rotate map generation returns the pre-run list and rotation pairs.
    def test_get_rotate_map_returns_rotation_pairs_when_block_size_specified(self, mock_rotate_cmd):
        adapter = self.create_adapter()
        adapter.config.num_hidden_layers = 2
        mock_rotate_cmd.side_effect = ['rot_a', 'rot_b']

        pre_run_list, rot_pairs = adapter.get_rotate_map(block_size=128)

        self.assertEqual(len(pre_run_list), 1)
        self.assertEqual(len(rot_pairs), 2)
        self.assertTrue(any('layers.0.attn.wq_a' in pair.right_rot for pair in rot_pairs))
        self.assertEqual(mock_rotate_cmd.call_count, 2)

    # Verify Ascend v1 save preprocessing preserves the original prefix (MTP is now in model.mtp).
    def test_ascendv1_save_module_preprocess_replaces_mtp_prefix_when_layer_prefix_given(self):
        adapter = self.create_adapter()
        adapter.config.num_hidden_layers = 3
        prefix, module = adapter.ascendv1_save_module_preprocess('layers.2.some', nn.Linear(1, 1), Mock())
        self.assertEqual(prefix, 'layers.2.some')

    def test_ascendv1_save_module_preprocess_unchanged_when_not_mtp_prefix(self):
        """验证非 MTP 层前缀不被修改"""
        adapter = self.create_adapter()
        adapter.config.num_hidden_layers = 3
        prefix, module = adapter.ascendv1_save_module_preprocess('layers.0.some', nn.Linear(1, 1), Mock())
        self.assertEqual(prefix, 'layers.0.some')

    def test_enable_kv_cache_does_nothing_when_called(self):
        """验证 enable_kv_cache 是空操作"""
        adapter = self.create_adapter()
        result = adapter.enable_kv_cache(Mock(), True)
        self.assertIsNone(result)

    def test_get_bake_names_returns_empty_lists_when_called(self):
        """验证 get_bake_names 返回两个空列表"""
        adapter = self.create_adapter()
        left, right = adapter.get_bake_names()
        self.assertEqual(left, [])
        self.assertEqual(right, [])

    def test_generate_model_visit_delegates_to_decoder_visit_func_when_called(self):
        """验证 generate_model_visit 委托给 generated_decoder_layer_visit_func"""
        adapter = self.create_adapter()
        with patch('msmodelslim.model.deepseek_v4.model_adapter.generated_decoder_layer_visit_func') as mock_visit:
            adapter.generate_model_visit(Mock())
        mock_visit.assert_called_once()

    @patch('msmodelslim.model.deepseek_v4.model_adapter.auto_dequant_state_dict')
    def test_load_decoder_if_not_exist_creates_new_decoder_when_missing(self, mock_auto_dequant):
        """验证 decoder 不存在时创建新 decoder 并加载权重"""
        adapter = self.create_adapter()
        dummy_model = DummyModelV4(num_layers=1, config=self.dummy_config)

        state_dict = {'linear.weight': torch.ones(1, 1), 'linear.bias': torch.zeros(1)}
        with patch.object(nn.Linear, 'reset_parameters', lambda _self: None):
            with patch('msmodelslim.model.deepseek_v4.model_adapter.get_state_dict', return_value=state_dict):
                decoder = adapter.load_decoder_if_not_exist(model=dummy_model, layer_prefix='layers.1', idx=1)

        self.assertIsNotNone(decoder)

    @patch('msmodelslim.model.deepseek_v4.model_adapter.auto_dequant_state_dict')
    def test_load_decoder_if_not_exist_creates_with_mtp_prefix_when_mtp_layer(self, mock_auto_dequant):
        """验证创建 decoder 并加载权重"""
        adapter = self.create_adapter()
        adapter.config.num_hidden_layers = 1
        dummy_model = DummyModelV4(num_layers=1, config=self.dummy_config)

        with patch.object(nn.Linear, 'reset_parameters', lambda _self: None):
            with patch(
                'msmodelslim.model.deepseek_v4.model_adapter.get_state_dict',
                return_value={'linear.weight': torch.ones(1, 1)},
            ):
                decoder = adapter.load_decoder_if_not_exist(model=dummy_model, layer_prefix='layers.0', idx=0)

                self.assertIsNotNone(decoder)

    def test_get_adapter_config_for_subgraph_three_layers_when_multi_layer(self):
        """验证 3 层时每层生成正确数量的子图配置"""
        adapter = self.create_adapter()
        adapter.config.num_hidden_layers = 3
        adapter.config.n_routed_experts = 2
        adapter.config.n_shared_experts = 1

        configs = adapter.get_adapter_config_for_subgraph()

        # 每层: 1 shared up-down + 2 routed up-down + 1 linear-linear + 2 norm-linear = 6
        # 3层: 18
        self.assertEqual(len(configs), 18)

    def test_get_adapter_config_for_subgraph_differs_by_layer_type(self):
        """验证不同 layer_idx 的 norm-linear 目标数量不同（layer 0 少, layer 1 多）"""
        adapter = self.create_adapter()
        adapter.config.num_hidden_layers = 4
        adapter.config.n_routed_experts = 1
        adapter.config.n_shared_experts = 1

        configs = adapter.get_adapter_config_for_subgraph()

        # layer0: norm-linear targets 少（无 compressor/indexer）
        layer0_configs = [c for c in configs if 'layers.0.attn_norm' in (c.mapping.source or '')]
        self.assertEqual(len(layer0_configs), 1)
        # layer0 只有 wq_a + wkv
        self.assertEqual(len(layer0_configs[0].mapping.targets), 2)

    def test_ln_fuse_map_includes_mtp_mappings_when_mtp_present(self):
        """验证 ln_fuse_map 包含 MTP 层专用映射"""
        adapter = self.create_adapter()
        adapter.config.num_hidden_layers = 1
        adapter.config.n_mtp_layers = 1
        adapter.config.n_routed_experts = 1
        adapter.config.n_shared_experts = 1

        pre_ln, ln_map = adapter.get_ln_fuse_map()

        self.assertIn('mtp.0.enorm', ln_map)
        self.assertIn('mtp.0.hnorm', ln_map)
        self.assertIn('mtp.0.norm', ln_map)

    def test_rotate_map_includes_mtp_rotations_when_mtp_layer(self):
        """验证 rotate_map 含 MTP 层专用旋转对"""
        adapter = self.create_adapter()
        adapter.config.num_hidden_layers = 1
        adapter.config.n_mtp_layers = 1
        adapter.config.n_routed_experts = 1
        adapter.config.n_shared_experts = 1

        with patch(
            'msmodelslim.model.deepseek_v4.model_adapter.QuaRotInterface.get_rotate_command', return_value='rot'
        ):
            pre_run, rot_pairs = adapter.get_rotate_map(block_size=128)

        self.assertEqual(len(rot_pairs), 2)
        # MTP 层 h_proj 在 mtp.0 的右旋中
        mtp_right = rot_pairs[0].right_rot
        self.assertIn('mtp.0.h_proj', mtp_right)

    def test_load_config_returns_model_args_when_called(self):
        """验证 _load_config 返回 ModelArgs 实例"""
        adapter = self.create_adapter()
        fake_config = {
            'hidden_size': 128,
            'moe_intermediate_size': 64,
            'index_topk': 1,
            'n_routed_experts': 2,
            'num_attention_heads': 8,
            'num_hidden_layers': 2,
            'o_groups': 1,
            'q_lora_rank': 8,
            'routed_scaling_factor': 1.0,
            'compress_ratios': [1, 1],
        }
        with (
            patch('msmodelslim.model.deepseek_v4.model_adapter.json_safe_load', return_value=fake_config),
            patch('msmodelslim.model.deepseek_v4.model_adapter.os.path.join', return_value='ignored'),
        ):
            config = adapter._load_config()
        self.assertIsNotNone(config)


class MockCompressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.kv_cache = None

    def forward(self, x, start_pos, freqs_cis):
        pass


def _apply_rotary_emb(x, freqs_cis):
    return x


class DummyIndexer(Indexer):
    """Use the native Indexer type without allocating model weights."""

    def __init__(self, n_heads=4, head_dim=64, compress_ratio=4):  # pylint: disable=super-init-not-called
        nn.Module.__init__(self)
        self.compress_ratio = compress_ratio
        self.rope_head_dim = head_dim // 2
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.index_topk = 2
        self.softmax_scale = 0.1
        self.wq_b = nn.Linear(head_dim, n_heads * head_dim)
        self.weights_proj = nn.Linear(head_dim, 1)
        self.compressor = MockCompressor()
        self.kv_cache = torch.zeros(1, 10, n_heads * head_dim)

    def forward(self, x, qr, start_pos, freqs_cis, offset):
        raise NotImplementedError("Original forward should be replaced")


class TestInjectFA3Placeholders(unittest.TestCase):
    """测试 DeepSeekV4ModelAdapter.inject_fa3_placeholders 的逻辑"""

    def _make_adapter_with_inject(self):
        adapter = DeepSeekV4ModelAdapter.__new__(DeepSeekV4ModelAdapter)
        return adapter

    def _build_model_with_indexer(self, module_name="indexer"):
        model = nn.Module()
        indexer = DummyIndexer()
        model.set_submodule(module_name, indexer)
        return model, indexer

    def _inject_with_mock_deepseek_module(self, adapter, root_name, model, should_inject):
        with patch('msmodelslim.model.deepseek_v4.model_adapter.import_module') as mock_import:
            mock_deepseek_module = MagicMock()
            mock_deepseek_module.apply_rotary_emb = _apply_rotary_emb
            mock_import.return_value = mock_deepseek_module
            adapter.inject_fa3_placeholders(root_name, model, should_inject)

    def test_should_inject_fa_q_fa_k_fa_v_placeholders(self):
        adapter = self._make_adapter_with_inject()
        model, indexer = self._build_model_with_indexer()
        self.assertFalse(hasattr(indexer, 'fa_q'))

        self._inject_with_mock_deepseek_module(adapter, "", model, lambda name: True)

        self.assertTrue(hasattr(indexer, 'fa_q'))
        self.assertTrue(hasattr(indexer, 'fa_k'))
        self.assertTrue(hasattr(indexer, 'fa_v'))
        self.assertIsInstance(indexer.fa_q, FA3QuantPlaceHolder)
        self.assertIsInstance(indexer.fa_k, FA3QuantPlaceHolder)
        self.assertIsInstance(indexer.fa_v, FA3QuantPlaceHolder)

    def test_should_inject_with_root_name(self):
        adapter = self._make_adapter_with_inject()
        model = nn.Module()
        root = nn.Module()
        indexer = DummyIndexer()
        root.set_submodule("indexer", indexer)
        model.set_submodule("attn", root)

        self._inject_with_mock_deepseek_module(adapter, "attn", root, lambda name: True)

        self.assertTrue(hasattr(indexer, 'fa_q'))
        self.assertTrue(hasattr(indexer, 'fa_k'))
        self.assertTrue(hasattr(indexer, 'fa_v'))

    def test_should_not_inject_when_should_inject_returns_false(self):
        adapter = self._make_adapter_with_inject()
        model, indexer = self._build_model_with_indexer()

        self._inject_with_mock_deepseek_module(adapter, "", model, lambda name: False)

        self.assertFalse(hasattr(indexer, 'fa_q'))

    def test_should_wrap_forward_method(self):
        adapter = self._make_adapter_with_inject()
        model, indexer = self._build_model_with_indexer()

        original_forward = indexer.forward
        self._inject_with_mock_deepseek_module(adapter, "", model, lambda name: True)

        self.assertIsNot(original_forward, indexer.forward.__wrapped__)  # pylint: disable=no-member

    def test_should_not_inject_non_indexer_modules(self):
        adapter = self._make_adapter_with_inject()
        model = nn.Module()
        attn = nn.Module()
        model.set_submodule("attn", attn)

        self._inject_with_mock_deepseek_module(adapter, "", model, lambda name: True)

        self.assertFalse(hasattr(attn, 'fa_q'))

    def test_inject_only_matching_indexer(self):
        adapter = self._make_adapter_with_inject()
        model = nn.Module()
        idx1 = DummyIndexer()
        idx2 = DummyIndexer()
        model.set_submodule("indexer_0", idx1)
        model.set_submodule("indexer_1", idx2)

        self._inject_with_mock_deepseek_module(adapter, "", model, lambda name: "indexer_0" in name)

        self.assertTrue(hasattr(idx1, 'fa_q'))
        self.assertFalse(hasattr(idx2, 'fa_q'))


def test_load_config_reads_checkpoint_yarn_when_dspark_is_configured():
    from msmodelslim.model.deepseek_v4_dspark.model_adapter import DeepSeekV4DSparkModelAdapter

    defaults = ModelArgs()
    checkpoint = {key: getattr(defaults, field) for field, key in DeepSeekV4ModelAdapter._get_args_mapping().items()}
    checkpoint.update(n_mtp_layers=1, dspark_block_size=5, rope_scaling={"type": "yarn", "factor": 16})

    dspark = DeepSeekV4DSparkModelAdapter.__new__(DeepSeekV4DSparkModelAdapter)
    legacy = DeepSeekV4ModelAdapter.__new__(DeepSeekV4ModelAdapter)
    dspark.model_path = legacy.model_path = Path(".")
    with (
        patch("msmodelslim.model.deepseek_v4.model_adapter.json_safe_load", return_value=checkpoint),
        patch("msmodelslim.model.deepseek_v4_dspark.model_adapter.json_safe_load", return_value=checkpoint),
    ):
        assert dspark._load_config().rope_factor == 16
        assert legacy._load_config().rope_factor == defaults.rope_factor


def _tiny_dspark_mtp_config():
    return ModelArgs(
        max_batch_size=1,
        max_seq_len=8,
        vocab_size=6,
        dim=2,
        moe_inter_dim=4,
        num_hidden_layers=1,
        n_hash_layers=0,
        n_heads=1,
        n_routed_experts=2,
        n_shared_experts=1,
        n_activated_experts=1,
        score_func="softmax",
        q_lora_rank=4,
        head_dim=8,
        rope_head_dim=4,
        o_groups=1,
        o_lora_rank=4,
        window_size=4,
        compress_ratios=(1, 1),
        original_seq_len=8,
        rope_factor=1.0,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        n_mtp_layers=1,
    )


def test_fake_quant_model_generates_two_greedy_tokens_when_built_on_meta():
    adapter = DeepSeekV4ModelAdapter.__new__(DeepSeekV4ModelAdapter)
    adapter.config = _tiny_dspark_mtp_config()
    adapter.config.compress_ratios = (4, 4)
    adapter.config.index_n_heads = 1
    adapter.config.index_head_dim = 8
    adapter.config.index_topk = 2
    adapter.tokenizer = Mock(eos_token_id=1)
    model = adapter.build_meta_model()

    parameters = dict(model.named_parameters())
    buffers = tuple(model.buffers())
    assert parameters and all(parameter.is_meta for parameter in parameters.values())
    assert buffers and all(buffer.device.type == "cpu" for buffer in buffers)
    assert model.eos_token_id == 1

    # Load deterministic weights without replacing the initialized runtime buffers.
    weights = {name: torch.zeros_like(parameter, device="cpu") for name, parameter in parameters.items()}
    weights["norm.weight"].fill_(1)
    weights["embed.weight"][0, 0] = weights["embed.weight"][1, 1] = 1
    weights["head.weight"][1, 0] = weights["head.weight"][2, 1] = 1
    model.load_state_dict(weights, assign=True)

    prefix_lengths = []

    def check_prefill(module, inputs):
        assert module.layers[0].attn.compressor.kv_cache is None
        prefix_lengths.append(inputs[0].shape[1])

    prompt = torch.tensor([[0, 0, 0, 0]])
    with model.register_forward_pre_hook(check_prefill):
        generated = model.generate(prompt, max_new_tokens=2, eos_token_id=None)

    assert prefix_lengths == [4, 5]
    assert generated.tolist() == [[0, 0, 0, 0, 1, 2]]


@pytest.mark.parametrize("method", ["minmax", "static"])
@pytest.mark.parametrize("split_branches", [False, True])
def test_dspark_forward_stays_native_when_backbone_fa3_is_enabled(method, split_branches):
    from msmodelslim.core.quantization_schema.hif8_activation_static import HIF8FakeQuantActivationPerHead
    from msmodelslim.model.deepseek_v4_dspark.model import DSparkBlock
    from msmodelslim.model.deepseek_v4_dspark.model_adapter import DeepSeekV4DSparkModelAdapter

    config = _tiny_dspark_mtp_config()
    config.dspark_block_size = 2
    config.dspark_target_layer_ids = (0,)
    config.dspark_markov_rank = 2
    model = nn.Module()
    model.attn = Attention(0, config).float()
    mtp = DSparkBlock(config.num_hidden_layers, config).float()
    model.mtp = nn.ModuleList([mtp])
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.05)
    reference = deepcopy(mtp)
    original_forward = mtp.attn.forward
    adapter = DeepSeekV4DSparkModelAdapter.__new__(DeepSeekV4DSparkModelAdapter)
    adapter.config = config
    qconfig = {"dtype": "hif8", "scope": "per_head", "symmetric": True, "method": method}
    if method == "static":
        qconfig["ext"] = {"static_scale": 15}
    branches = ("fa_q", "fa_k")
    configs = (
        [FA3QuantProcessorConfig(details={branch: qconfig}, exclude=["*mtp*"]) for branch in branches]
        if split_branches
        else [FA3QuantProcessorConfig(details=dict.fromkeys(branches, qconfig), exclude=["*mtp*"])]
    )
    draft = torch.arange(1, 9, dtype=torch.float32).reshape(1, 2, 2, 2)
    main = torch.arange(1, 7, dtype=torch.float32).reshape(1, 3, 2) / 10
    tokens = torch.tensor([[0, 1]])
    request = BatchProcessRequest(name="", module=model, datas=[])
    for cfg in configs:
        processor = FA3QuantProcessor(model, cfg, adapter)
        processor.preprocess(request)
        if not processor.is_data_free():
            with torch.no_grad():
                model.attn(main, 0)
        processor.postprocess(request)
        processor.post_run()

    assert mtp.attn.forward == original_forward
    assert not any(isinstance(module, (FA3QuantPlaceHolder, AutoFakeQuantActivation)) for module in mtp.modules())
    for branch in branches:
        assert isinstance(getattr(model.attn, branch), HIF8FakeQuantActivationPerHead)
    # Both DSpark prefill and decode retain their original main_x and cache behavior.
    with torch.no_grad():
        for start_pos, main_x in ((0, main), (2, main[:, -1:])):
            output = mtp(draft, start_pos, tokens, main_x=main_x)
            expected = reference(draft, start_pos, tokens, main_x=main_x)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(mtp.attn.kv_cache, reference.attn.kv_cache, rtol=0, atol=0)
            assert output.shape == draft.shape and bool(torch.isfinite(output).all())


def _attention_shell():
    def empty_module(cls):
        module = cls.__new__(cls)
        # Initialize only nn.Module, without constructing the full attention model.
        nn.Module.__init__(module)  # pylint: disable=unnecessary-dunder-call
        return module

    model = nn.Module()
    model.attn = empty_module(Attention)
    model.attn.n_local_heads = 2
    model.attn.head_dim = 8
    model.attn.compressor = empty_module(Compressor)
    model.attn.compressor.head_dim = 8
    model.attn.indexer = empty_module(Indexer)
    model.attn.indexer.n_heads = 3
    model.attn.indexer.head_dim = 4
    model.attn.indexer.compressor = empty_module(Compressor)
    return model


def test_inject_fa3_placeholders_declares_real_boundaries_when_backbone_supports_fa3():
    adapter = DeepSeekV4ModelAdapter.__new__(DeepSeekV4ModelAdapter)
    model = _attention_shell()
    original_forward = model.attn.indexer.forward
    original_compressor_forward = model.attn.indexer.compressor.forward
    adapter.inject_fa3_placeholders("", model, lambda _: True)
    assert {name: node.num_heads for name, node in model.named_modules() if isinstance(node, FA3QuantPlaceHolder)} == {
        "attn.fa_q": 2,
        "attn.fa_k": 1,
        "attn.indexer.fa_q": 3,
        "attn.indexer.fa_k": 1,
        "attn.indexer.fa_v": 1,
    }
    wrapped_forward = model.attn.indexer.forward
    assert wrapped_forward != original_forward
    installed_nodes = dict(model.named_modules())
    adapter.inject_fa3_placeholders("", model, lambda _: True)
    assert dict(model.named_modules()) == installed_nodes
    assert model.attn.indexer.compressor.forward == original_compressor_forward
    assert not hasattr(model.attn, "fa_v")


@pytest.mark.parametrize(
    "selected,branches",
    [
        ("layers.0.attn", ("fa_q", "fa_k")),
        ("layers.0.attn.compressor", ()),
    ],
)
def test_inject_fa3_placeholders_respects_module_filter_when_root_name_is_given(selected, branches):
    model = _attention_shell()
    adapter = DeepSeekV4ModelAdapter.__new__(DeepSeekV4ModelAdapter)
    original_modules = dict(model.named_modules())
    original_compressor_forward = model.attn.compressor.forward
    adapter.inject_fa3_placeholders("layers.0", model, lambda name: name == selected)
    assert {"layers.0." + name for name, node in model.named_modules() if isinstance(node, FA3QuantPlaceHolder)} == {
        f"{selected}.{branch}" for branch in branches
    }
    assert model.attn.compressor.forward == original_compressor_forward
    if not branches:
        assert dict(model.named_modules()) == original_modules


def test_inject_fa3_placeholders_keeps_nodes_when_another_processor_already_installed_ir():
    model = _attention_shell()
    adapter = DeepSeekV4ModelAdapter.__new__(DeepSeekV4ModelAdapter)
    fixed = AutoFakeQuantActivation.create(QParam(hif8_per_tensor_sym, ext={"scale": torch.ones(1)}))
    dynamic = AutoFakeQuantActivation.create(QParam(int8_per_token_sym))
    model.attn.fa_q = fixed
    model.attn.indexer.fa_k = dynamic
    adapter.inject_fa3_placeholders("", model, lambda _: True)
    assert model.attn.fa_q is fixed
    assert model.attn.indexer.fa_k is dynamic
    nodes = dict(model.named_modules())
    adapter.inject_fa3_placeholders("", model, lambda _: False)
    assert dict(model.named_modules()) == nodes


def _make_test_attention(ratio=4, *, max_seq_len=None, model=None):
    args = _tiny_dspark_mtp_config()
    if max_seq_len is not None:
        args.max_seq_len = args.original_seq_len = max_seq_len
    args.compress_ratios = (ratio, ratio)
    args.index_n_heads, args.index_head_dim, args.index_topk = 1, 8, 2
    attention = Attention(0, args).float()
    if model is not None:
        model.attn = attention
    with torch.no_grad():
        for parameter in (model if model is not None else attention).parameters():
            parameter.fill_(0.05)
    return args, attention


@pytest.mark.parametrize("ratio", [1, 4, 128])
def test_attention_forward_keeps_outputs_when_only_transparent_fa3_placeholders_are_installed(ratio):
    args, attention = _make_test_attention(ratio, max_seq_len=max(8, ratio))
    reference = deepcopy(attention)
    adapter = DeepSeekV4ModelAdapter.__new__(DeepSeekV4ModelAdapter)
    adapter.inject_fa3_placeholders("attn", attention, lambda _: True)
    values = torch.arange(1, 1 + max(4, ratio) * args.dim, dtype=torch.float32).reshape(1, -1, args.dim) / 100
    with torch.no_grad():
        torch.testing.assert_close(attention(values, 0), reference(values, 0), rtol=0, atol=0)


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("replace_parent_cache", [False, True])
def test_full_prefix_cache_maintenance_matches_baseline_when_identity_quantizers_are_installed(
    ratio,
    replace_parent_cache,
):
    args, reference = _make_test_attention(ratio, max_seq_len=ratio + 8)
    quantized = deepcopy(reference)
    adapter = DeepSeekV4ModelAdapter.__new__(DeepSeekV4ModelAdapter)
    adapter.inject_fa3_placeholders("attn", quantized, lambda _: True)
    values = torch.arange(1, 1 + (ratio + 5) * args.dim, dtype=torch.float32).reshape(1, -1, args.dim) / 100
    for length in (ratio + 5, 3, ratio):
        if replace_parent_cache:
            for attention in (reference, quantized):
                attention.kv_cache = torch.zeros_like(attention.kv_cache)
                if attention.indexer is not None:
                    attention.indexer.kv_cache = torch.zeros_like(attention.indexer.kv_cache)
                # The complete-prefix evaluator invalidates views before each round.
                for module in attention.modules():
                    if isinstance(module, Compressor):
                        module.kv_cache = None
        with torch.no_grad():
            torch.testing.assert_close(
                quantized(values[:, :length], 0), reference(values[:, :length], 0), rtol=0, atol=0
            )
        expected_buffers = dict(reference.named_buffers())
        actual_buffers = dict(quantized.named_buffers())
        assert actual_buffers.keys() == expected_buffers.keys()
        for name, expected in expected_buffers.items():
            torch.testing.assert_close(actual_buffers[name], expected, rtol=0, atol=0)
        for attention in (reference, quantized):
            assert attention.compressor.kv_cache.data_ptr() == attention.kv_cache[:, args.window_size :].data_ptr()
            if attention.indexer is not None:
                assert attention.indexer.compressor.kv_cache is attention.indexer.kv_cache


@pytest.mark.parametrize(
    "dtype,scope,method",
    [
        ("hif8", "per_tensor", "static"),
        ("hif8", "per_head", "static"),
        ("hif8", "per_tensor", "minmax"),
        ("hif8", "per_head", "minmax"),
        ("int8", "per_token", "minmax"),
        ("fp8_e4m3", "per_token", "minmax"),
        ("int8", "per_head", "minmax"),
        ("fp8_e4m3", "per_head", "minmax"),
    ],
)
@pytest.mark.parametrize("ratio", [4, 128])
def test_attention_forward_consumes_quantized_q_and_complete_kv_when_fa3_is_enabled(
    dtype,
    scope,
    method,
    ratio,
    monkeypatch,
):
    model = nn.Module()
    _make_test_attention(ratio, max_seq_len=ratio, model=model)
    reference = deepcopy(model.attn)
    from msmodelslim.model.deepseek_v4 import model as model_module

    kernel_inputs = []
    original_sparse_attn = model_module.sparse_attn

    def record_kernel(q, kv, *args):
        kernel_inputs.append((q.detach().clone(), kv.detach().clone()))
        return original_sparse_attn(q, kv, *args)

    monkeypatch.setattr(model_module, "sparse_attn", record_kernel)
    adapter = DeepSeekV4ModelAdapter.__new__(DeepSeekV4ModelAdapter)
    qconfig = dict(dtype=dtype, scope=scope, symmetric=True, method=method)
    if method == "static":
        qconfig["ext"] = {"static_scale": 15}
    details = {"fa_q": qconfig, "fa_k": qconfig}
    configs = [
        FA3QuantProcessorConfig(details=details, include=["*attn.indexer"]),
        FA3QuantProcessorConfig(details=details, include=["*attn"], exclude=["*indexer*"]),
    ]
    values = torch.arange(1, 2 * ratio + 1, dtype=torch.float32).reshape(1, ratio, 2) / (2 * ratio)
    request = BatchProcessRequest(name="", module=model, datas=[])
    for config in configs:
        processor = FA3QuantProcessor(model, config, adapter)
        processor.preprocess(request)
        if not processor.is_data_free():
            with torch.no_grad():
                model.attn(values, 0)
        processor.postprocess(request)
    with torch.no_grad():
        reference(values, 0)
    kernel_inputs.clear()
    indexer_inputs = []
    original_einsum = torch.einsum

    def record_einsum(equation, *operands):
        if equation == "bshd,btd->bsht":
            indexer_inputs.append(tuple(tensor.detach().clone() for tensor in operands))
        return original_einsum(equation, *operands)

    monkeypatch.setattr(torch, "einsum", record_einsum)
    captured = {}
    quantizer_inputs = {}
    handles = []
    for name, node in model.named_modules():
        if isinstance(node, AutoFakeQuantActivation):

            def record(_node, inputs, output, name=name):
                quantizer_inputs[name] = inputs[0].detach().clone()
                captured[name] = output.detach().clone()

            handles.append(node.register_forward_hook(record))
    try:
        with torch.no_grad():
            output = model.attn(values, 0)
        assert bool(torch.isfinite(output).all())
        expected = {"attn.fa_q", "attn.fa_k"}
        if ratio == 4:
            expected.update(("attn.indexer.fa_q", "attn.indexer.fa_k"))
            # Indexer scores use quantized K while the persistent cache stays unchanged.
            raw_indexer_k = quantizer_inputs["attn.indexer.fa_k"].squeeze(1)
            assert len(indexer_inputs) == 1
            torch.testing.assert_close(indexer_inputs[0][0], captured["attn.indexer.fa_q"].transpose(1, 2))
            torch.testing.assert_close(indexer_inputs[0][1], captured["attn.indexer.fa_k"].squeeze(1), rtol=0, atol=0)
            torch.testing.assert_close(model.attn.indexer.kv_cache[:, :1], raw_indexer_k, rtol=0, atol=0)
        else:
            assert not indexer_inputs
        assert set(captured) == expected
        assert len(kernel_inputs) == 1
        kernel_q, kernel_kv = kernel_inputs[0]
        assert kernel_q.shape == (1, ratio, model.attn.n_local_heads, model.attn.head_dim)
        assert kernel_kv.shape == (1, ratio + 1, model.attn.head_dim)
        torch.testing.assert_close(kernel_q, captured["attn.fa_q"].transpose(1, 2), rtol=0, atol=0)
        torch.testing.assert_close(kernel_kv, captured["attn.fa_k"].squeeze(1), rtol=0, atol=0)
        torch.testing.assert_close(model.attn.kv_cache, reference.kv_cache, rtol=0, atol=0)
        torch.testing.assert_close(model.attn.compressor.kv_cache, reference.compressor.kv_cache, rtol=0, atol=0)
        assert not hasattr(model.attn.compressor, "fa_k")
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("prefill_length", [3, 4])
@pytest.mark.parametrize("quantize_k", [False, True])
def test_indexer_uses_selected_k_and_preserves_cache_when_prefill_and_decode(
    prefill_length,
    quantize_k,
    monkeypatch,
):
    model = nn.Module()
    args, _ = _make_test_attention(model=model)
    reference = deepcopy(model.attn.indexer)
    adapter = DeepSeekV4ModelAdapter.__new__(DeepSeekV4ModelAdapter)
    adapter.inject_fa3_placeholders("", model, lambda name: name == "attn.indexer")
    indexer = model.attn.indexer
    original_einsum = torch.einsum
    captured = {}

    def record_einsum(equation, *operands):
        if equation == "bshd,btd->bsht":
            captured["kernel_k"] = operands[1].detach().clone()
        return original_einsum(equation, *operands)

    def zero_k(_module, inputs, _output):
        captured["raw_k"] = inputs[0].detach().clone().squeeze(1)
        return torch.zeros_like(inputs[0])

    monkeypatch.setattr(torch, "einsum", record_einsum)
    if quantize_k:
        handle = indexer.fa_k.register_forward_hook(zero_k)
    else:
        del indexer.fa_k
        handle = None
    values = torch.arange(1, 17, dtype=torch.float32).reshape(1, 8, 2) / 8
    # Exercise empty K, new compressed K and decode steps that reuse existing K.
    steps = [(0, prefill_length)] + [(pos, 1) for pos in range(prefill_length, 8)]
    try:
        with torch.no_grad():
            for start_pos, length in steps:
                end_pos = start_pos + length
                x = values[:, start_pos:end_pos]
                qr = model.attn.q_norm(model.attn.wq_a(x))
                freqs = model.attn.freqs_cis[start_pos:end_pos]
                expected = reference(x, qr, start_pos, freqs, args.window_size)
                captured.clear()
                actual = indexer(x, qr, start_pos, freqs, args.window_size)
                kv_len = end_pos // indexer.compress_ratio
                raw_k = reference.kv_cache[:, :kv_len]
                expected_k = torch.zeros_like(raw_k) if quantize_k else raw_k
                torch.testing.assert_close(captured["kernel_k"], expected_k, rtol=0, atol=0)
                if quantize_k and kv_len:
                    assert bool(torch.count_nonzero(raw_k))
                    torch.testing.assert_close(captured["raw_k"], raw_k, rtol=0, atol=0)
                else:
                    assert "raw_k" not in captured
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for name, buffer in reference.named_buffers():
                    torch.testing.assert_close(indexer.get_buffer(name), buffer, rtol=0, atol=0)
    finally:
        if handle is not None:
            handle.remove()
    assert indexer.compressor.forward.__func__ is Compressor.forward


@pytest.mark.parametrize("scope", ["per_tensor", "per_head"])
@pytest.mark.parametrize("dynamic_first", [False, True])
def test_fa3_processors_preserve_nodes_when_static_and_dynamic_branches_are_configured_separately(scope, dynamic_first):
    model = nn.Module()
    _make_test_attention(model=model)
    values = torch.arange(1, 9, dtype=torch.float32).reshape(1, 4, 2) / 8
    adapter = DeepSeekV4ModelAdapter.__new__(DeepSeekV4ModelAdapter)
    request = BatchProcessRequest(name="", module=model, datas=[])
    fixed = FA3QuantProcessor(
        model,
        FA3QuantProcessorConfig(
            details={
                "fa_q": dict(dtype="hif8", scope=scope, symmetric=True, method="static", ext={"static_scale": 15}),
            }
        ),
        adapter,
    )
    dynamic = FA3QuantProcessor(
        model,
        FA3QuantProcessorConfig(
            details={
                "fa_k": dict(dtype="int8", scope="per_token", symmetric=True, method="minmax"),
            }
        ),
        adapter,
    )
    first, second = (dynamic, fixed) if dynamic_first else (fixed, dynamic)

    def install(processor):
        processor.preprocess(request)
        if not processor.is_data_free():
            with torch.no_grad():
                model.attn(values, 0)
        processor.postprocess(request)

    install(first)
    installed = {name: node for name, node in model.named_modules() if isinstance(node, AutoFakeQuantActivation)}
    install(second)
    assert installed
    assert all(model.get_submodule(name) is node for name, node in installed.items())
    assert all(
        isinstance(getattr(owner, "fa_q"), AutoFakeQuantActivation) for owner in (model.attn, model.attn.indexer)
    )
    assert all(
        isinstance(getattr(owner, "fa_k"), AutoFakeQuantActivation) for owner in (model.attn, model.attn.indexer)
    )


@pytest.mark.parametrize(
    "include,exclude,expected",
    [
        (["attn.fa_q"], [], {"attn.fa_q"}),
        (["attn.fa_*"], ["attn.fa_k"], {"attn.fa_q"}),
        (["*"], ["*indexer*", "*compressor*"], {"attn.fa_q", "attn.fa_k"}),
        (["attn.fa_q"], ["attn.fa_q"], set()),
    ],
)
def test_fa3_processor_honors_filters_when_installing_hif8(include, exclude, expected):
    model = nn.Module()
    _make_test_attention(model=model)
    adapter = DeepSeekV4ModelAdapter.__new__(DeepSeekV4ModelAdapter)
    request = BatchProcessRequest(name="", module=model, datas=[])
    qconfig = dict(dtype="hif8", scope="per_head", symmetric=True, method="static")
    config = FA3QuantProcessorConfig(
        include=include,
        exclude=exclude,
        details={"fa_q": qconfig, "fa_k": qconfig},
    )
    processor = FA3QuantProcessor(model, config, adapter)
    processor.preprocess(request)
    with torch.no_grad():
        model.attn(torch.arange(1, 9, dtype=torch.float32).reshape(1, 4, 2) / 8, 0)
    processor.postprocess(request)

    assert {name for name, module in model.named_modules() if isinstance(module, AutoFakeQuantActivation)} == expected


@pytest.mark.parametrize("method", ["static", "minmax"])
@pytest.mark.parametrize("scope", ["per_tensor", "per_head"])
@pytest.mark.parametrize("legacy_first", [False, True])
def test_indexer_group_keeps_legacy_v_when_hif8_and_fp8_share_a_layer(method, scope, legacy_first):
    from msmodelslim.core.devkit.group import GroupProcessor, GroupProcessorConfig

    model = nn.Module()
    _make_test_attention(model=model)
    adapter = DeepSeekV4ModelAdapter.__new__(DeepSeekV4ModelAdapter)
    qconfig = dict(dtype="hif8", scope=scope, symmetric=True, method=method)
    hif8 = FA3QuantProcessorConfig(
        include=["attn.indexer"],
        details={"fa_q": qconfig, "fa_k": qconfig},
    )
    legacy = FA3QuantProcessorConfig(
        include=["attn.indexer"],
        details={"fa_v": dict(dtype="fp8_e4m3", scope="per_head", symmetric=True, method="minmax")},
    )
    stages = [legacy, hif8] if legacy_first else [hif8, legacy]
    group = GroupProcessor(model, GroupProcessorConfig(type="group", configs=stages), adapter)
    request = BatchProcessRequest(name="", module=model, datas=[])
    values = torch.arange(1, 9, dtype=torch.float32).reshape(1, 4, 2) / 8

    group.preprocess(request)
    with torch.no_grad():
        model.attn(values, 0)
    raw_latent = model.attn.indexer.kv_cache[:, :1].clone()
    group.postprocess(request)

    indexer = model.attn.indexer
    assert indexer.fa_q.x_q_scheme == indexer.fa_k.x_q_scheme == hif8.details.fa_q.to_scheme()
    assert indexer.fa_v.x_q_scheme == legacy.details.fa_v.to_scheme()
    expected_scale = raw_latent.abs().amax().reshape_as(indexer.fa_v.input_scale) / 448
    torch.testing.assert_close(indexer.fa_v.input_scale, expected_scale)


@pytest.mark.parametrize("scope", ["per_tensor", "per_head"])
@pytest.mark.parametrize("minmax_first", [False, True])
@pytest.mark.parametrize("method", ["static", "minmax"])
def test_fa3_keeps_separate_scales_when_nodes_share_branch_name(scope, minmax_first, method):
    from msmodelslim.core.devkit.group import GroupProcessor, GroupProcessorConfig

    model = nn.Module()
    _make_test_attention(model=model)
    adapter = DeepSeekV4ModelAdapter.__new__(DeepSeekV4ModelAdapter)
    hif8_qconfig = dict(dtype="hif8", scope=scope, symmetric=True, method=method)
    if method == "static":
        hif8_qconfig["ext"] = {"static_scale": 15}
    fp8_qconfig = dict(dtype="fp8_e4m3", scope="per_head", symmetric=True, method="minmax")
    hif8 = FA3QuantProcessorConfig(include=["attn"], details={"fa_k": hif8_qconfig})
    fp8 = FA3QuantProcessorConfig(include=["attn.indexer"], details={"fa_k": fp8_qconfig})
    configs = [fp8, hif8] if minmax_first else [hif8, fp8]
    if method == "static":
        processors = [GroupProcessor(model, GroupProcessorConfig(type="group", configs=configs), adapter)]
    else:
        processors = [FA3QuantProcessor(model, config, adapter) for config in configs]
    request = BatchProcessRequest(name="", module=model, datas=[])
    values = torch.arange(1, 9, dtype=torch.float32).reshape(1, 4, 2) / 8

    for processor in processors:
        processor.preprocess(request)
        with torch.no_grad():
            model.attn(values, 0)
        processor.postprocess(request)

    assert model.attn.fa_k.x_q_scheme == hif8.details.fa_k.to_scheme()
    assert model.attn.indexer.fa_k.x_q_scheme == fp8.details.fa_k.to_scheme()
    if method == "static":
        torch.testing.assert_close(model.attn.fa_k.input_scale, torch.tensor([15.0]))


@pytest.mark.parametrize(
    "dtype,scope,method,branches,exclude,expected",
    [
        ("int8", "per_token", "minmax", (), [], "INT8_DYNAMIC"),
        ("int8", "per_token", "minmax", ("fa_q", "fa_k"), [], "QK_INT8_DYNAMIC"),
        ("int8", "per_token", "minmax", (), ["attn.indexer.fa_v"], "QK_INT8_DYNAMIC"),
        ("int8", "per_token", "minmax", ("fa_k",), [], "K_INT8_DYNAMIC"),
        ("int8", "per_token", "minmax", ("fa_v",), [], "V_INT8_DYNAMIC"),
        ("fp8_e4m3", "per_head", "minmax", (), [], "FP8"),
        ("hif8", "per_tensor", "static", ("fa_q", "fa_k"), [], "QK_HIF8_PER_TENSOR"),
    ],
)
def test_indexer_export_reloads_expected_nodes_when_legacy_or_hif8_is_configured(
    tmp_path,
    dtype,
    scope,
    method,
    branches,
    exclude,
    expected,
):
    import json
    from msmodelslim.core.quant_service.modelslim_v1.save.ascendv1 import AscendV1Config, AscendV1Saver
    from msmodelslim.core.quantization_format.ascendV1_format.format import AscendV1Format

    model = nn.Module()
    args, _ = _make_test_attention(model=model)
    source_path = tmp_path / "original"
    source_path.mkdir()
    (source_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v4"}), encoding="utf-8")
    adapter = DeepSeekV4ModelAdapter.__new__(DeepSeekV4ModelAdapter)
    adapter.config, adapter.model_path = args, source_path
    adapter.model_type = "DeepSeek-V4"
    request = BatchProcessRequest(name="", module=model, datas=[])
    qconfig = dict(dtype=dtype, scope=scope, symmetric=True, method=method)
    scheme = {"details": {branch: qconfig for branch in branches}} if branches else {"qconfig": qconfig}
    include = ["attn.indexer.fa_*"] if exclude else ["attn.indexer"]
    config = FA3QuantProcessorConfig(include=include, exclude=exclude, **scheme)
    processor = FA3QuantProcessor(model, config, adapter)
    processor.preprocess(request)
    values = torch.arange(1, 9, dtype=torch.float32).reshape(1, 4, 2) / 8
    raw_latent = None
    if not processor.is_data_free():
        with torch.no_grad():
            model.attn(values, 0)
        raw_latent = model.attn.indexer.kv_cache[:, :1].clone()
    processor.postprocess(request)
    if dtype == "fp8_e4m3":
        assert raw_latent is not None
        expected_scale = raw_latent.abs().amax().reshape_as(model.attn.indexer.fa_v.input_scale) / 448
        torch.testing.assert_close(model.attn.indexer.fa_v.input_scale, expected_scale)
    with torch.no_grad():
        expected_output = model.attn(values, 0)
    expected_cache = model.attn.indexer.kv_cache[:, :1].clone()
    output_path = tmp_path / "quantized"
    output_path.mkdir()
    saver = AscendV1Saver(model, AscendV1Config(save_directory=str(output_path), part_file_size=0), adapter)
    saver.postprocess(request)
    saver.post_run()
    description = json.loads((output_path / "quant_model_description.json").read_text(encoding="utf-8"))
    assert description["attn.indexer.quant_type"] == expected
    assert ("attn.indexer.fa_v.scale" in description) == (dtype == "fp8_e4m3")
    restored = nn.Module()
    restored.attn = Attention(0, args).float()
    loader = AscendV1Format(str(output_path))
    loader.bind_adapter(adapter)
    loader.apply_ir(restored)
    loader.hydrate(restored)
    for branch in ("fa_q", "fa_k", "fa_v"):
        original = getattr(model.attn.indexer, branch, None)
        reloaded = getattr(restored.attn.indexer, branch, None)
        if isinstance(original, AutoFakeQuantActivation):
            assert type(reloaded) is type(original)
        else:
            assert not isinstance(reloaded, AutoFakeQuantActivation)
    assert not hasattr(restored.attn.indexer.compressor, "fa_k")
    with torch.no_grad():
        torch.testing.assert_close(restored.attn(values, 0), expected_output)
    torch.testing.assert_close(restored.attn.indexer.kv_cache[:, :1], expected_cache)


@pytest.mark.parametrize("dtype,divisor", [("int8", 127), ("fp8_e4m3", 448), ("hif8", 32768)])
def test_indexer_static_scale_preserves_dtype_range_when_calibration_has_an_outlier(dtype, divisor):
    model = _attention_shell()
    adapter = DeepSeekV4ModelAdapter.__new__(DeepSeekV4ModelAdapter)
    qconfig = dict(dtype=dtype, scope="per_head", symmetric=True, method="minmax")
    branches = {"details": {"fa_q": qconfig, "fa_k": qconfig}} if dtype == "hif8" else {"qconfig": qconfig}
    config = FA3QuantProcessorConfig(include=["*indexer*"], **branches)
    processor = FA3QuantProcessor(model, config, adapter)
    request = BatchProcessRequest(name="", module=model, datas=[])
    processor.preprocess(request)
    assert not hasattr(model.attn, "fa_q") and not hasattr(model.attn.compressor, "fa_k")
    values = torch.tensor([1.0, 1.0, 1.0, 1000.0]).reshape(1, 1, 1, 4)
    model.attn.indexer.fa_q(values.expand(1, 3, 1, 4))
    model.attn.indexer.fa_k(values)
    model.attn.indexer.fa_v(values)
    processor.postprocess(request)
    for branch, count in (("fa_q", 3), ("fa_k", 1)):
        node = getattr(model.attn.indexer, branch)
        torch.testing.assert_close(node.input_scale.reshape(-1), torch.full((count,), 1 / divisor))
        if dtype == "hif8":
            output = node(values.expand(1, count, 1, 4))
            assert torch.isfinite(output).all()
            torch.testing.assert_close(output[..., -1], torch.ones((1, count, 1)))
    if dtype == "hif8":
        assert isinstance(model.attn.indexer.fa_v, FA3QuantPlaceHolder)
    else:
        torch.testing.assert_close(model.attn.indexer.fa_v.input_scale.reshape(-1), torch.tensor([1000 / divisor]))
