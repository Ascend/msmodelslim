#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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
from pathlib import Path
from unittest.mock import MagicMock, patch

from torch import nn

from msmodelslim.core.const import DeviceType
from msmodelslim.model.qwen2_5.model_adapter import Qwen25ModelAdapter
from msmodelslim.processor.kv_smooth import KVSmoothFusedType, KVSmoothFusedUnit
from msmodelslim.utils.exception import InvalidModelError


class DummyConfig:
    """模拟配置对象"""

    def __init__(self):
        self.hidden_size = 128
        self.num_attention_heads = 8
        self.num_key_value_heads = 4
        self.num_hidden_layers = 3


class TestQwen25ModelAdapter(unittest.TestCase):
    def setUp(self):
        self.model_type = 'Qwen2.5-7B-Instruct'
        self.model_path = Path('.')

    def test_get_model_type(self):
        """测试get_model_type方法"""
        with patch('msmodelslim.model.qwen2_5.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen25ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.model_type = self.model_type

            result = adapter.get_model_type()
            self.assertEqual(result, self.model_type)

    def test_get_model_pedigree(self):
        """测试get_model_pedigree方法"""
        with patch('msmodelslim.model.qwen2_5.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen25ModelAdapter(model_type=self.model_type, model_path=self.model_path)

            result = adapter.get_model_pedigree()
            self.assertEqual(result, 'qwen2_5')

    def test_load_model(self):
        """测试load_model方法"""
        with patch('msmodelslim.model.qwen2_5.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen25ModelAdapter(model_type=self.model_type, model_path=self.model_path)

            mock_model = nn.Linear(10, 10)
            adapter._load_model = MagicMock(return_value=mock_model)

            result = adapter.load_model(device=DeviceType.NPU)

            self.assertIs(result, mock_model)
            adapter._load_model.assert_called_once_with(DeviceType.NPU)

    def test_handle_dataset(self):
        """测试handle_dataset方法"""
        with patch('msmodelslim.model.qwen2_5.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen25ModelAdapter(model_type=self.model_type, model_path=self.model_path)

            mock_dataset = ['data1', 'data2']
            adapter._get_tokenized_data = MagicMock(return_value=mock_dataset)

            result = adapter.handle_dataset(dataset='test_data', device=DeviceType.CPU)

            self.assertEqual(result, mock_dataset)
            adapter._get_tokenized_data.assert_called_once_with('test_data', DeviceType.CPU)

    def test_handle_dataset_by_batch(self):
        """测试handle_dataset_by_batch方法"""
        with patch('msmodelslim.model.qwen2_5.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen25ModelAdapter(model_type=self.model_type, model_path=self.model_path)

            mock_batch_dataset = [['batch1'], ['batch2']]
            adapter._get_batch_tokenized_data = MagicMock(return_value=mock_batch_dataset)

            result = adapter.handle_dataset_by_batch(dataset='test_data', batch_size=2, device=DeviceType.CPU)

            self.assertEqual(result, mock_batch_dataset)
            adapter._get_batch_tokenized_data.assert_called_once_with(
                calib_list='test_data', batch_size=2, device=DeviceType.CPU
            )

    def test_init_model(self):
        """测试init_model方法"""
        with patch('msmodelslim.model.qwen2_5.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen25ModelAdapter(model_type=self.model_type, model_path=self.model_path)

            mock_model = nn.Linear(10, 10)
            adapter._load_model = MagicMock(return_value=mock_model)

            result = adapter.init_model(device=DeviceType.NPU)

            self.assertIs(result, mock_model)
            adapter._load_model.assert_called_once_with(DeviceType.NPU)

    def test_enable_kv_cache(self):
        """测试enable_kv_cache方法"""
        with patch('msmodelslim.model.qwen2_5.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen25ModelAdapter(model_type=self.model_type, model_path=self.model_path)

            mock_model = nn.Linear(10, 10)
            adapter._enable_kv_cache = MagicMock(return_value=None)

            adapter.enable_kv_cache(model=mock_model, need_kv_cache=True)

            adapter._enable_kv_cache.assert_called_once_with(mock_model, True)

    def test_get_kvcache_smooth_fused_subgraph(self):
        """测试get_kvcache_smooth_fused_subgraph方法"""
        with patch('msmodelslim.model.qwen2_5.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen25ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = DummyConfig()

            result = adapter.get_kvcache_smooth_fused_subgraph()

            # 验证返回列表
            self.assertIsInstance(result, list)
            # 每一层应该有一个KVSmoothFusedUnit
            self.assertEqual(len(result), adapter.config.num_hidden_layers)

            # 验证第一个单元的配置
            first_unit = result[0]
            self.assertIsInstance(first_unit, KVSmoothFusedUnit)
            self.assertEqual(first_unit.attention_name, "model.layers.0.self_attn")
            self.assertEqual(first_unit.layer_idx, 0)
            self.assertEqual(first_unit.fused_from_query_states_name, "q_proj")
            self.assertEqual(first_unit.fused_from_key_states_name, "k_proj")
            self.assertEqual(first_unit.fused_type, KVSmoothFusedType.StateViaRopeToLinear)

    def test_get_head_dim_success(self):
        """测试get_head_dim方法成功情况"""
        with patch('msmodelslim.model.qwen2_5.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen25ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = DummyConfig()

            result = adapter.get_head_dim()

            # hidden_size=128, num_attention_heads=8, head_dim=16
            expected = adapter.config.hidden_size // adapter.config.num_attention_heads
            self.assertEqual(result, expected)
            self.assertEqual(result, 16)

    def test_get_head_dim_missing_hidden_size(self):
        """测试get_head_dim方法缺少hidden_size时抛出异常"""
        with patch('msmodelslim.model.qwen2_5.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen25ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            # 创建一个没有hidden_size的config
            adapter.config = type('Config', (), {})()

            with self.assertRaises(InvalidModelError) as context:
                adapter.get_head_dim()

            self.assertIn("hidden_size is not found", str(context.exception))

    def test_get_head_dim_missing_num_attention_heads(self):
        """测试get_head_dim方法缺少num_attention_heads时抛出异常"""
        with patch('msmodelslim.model.qwen2_5.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen25ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            # 创建一个有hidden_size但没有num_attention_heads的config
            adapter.config = type('Config', (), {'hidden_size': 128})()

            with self.assertRaises(InvalidModelError) as context:
                adapter.get_head_dim()

            self.assertIn("num_attention_heads is not found", str(context.exception))

    def test_get_head_dim_zero_num_attention_heads(self):
        """测试get_head_dim方法num_attention_heads为0时抛出异常"""
        with patch('msmodelslim.model.qwen2_5.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen25ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = type('Config', (), {'hidden_size': 128, 'num_attention_heads': 0})()

            with self.assertRaises(InvalidModelError) as context:
                adapter.get_head_dim()

            self.assertIn("num_attention_heads is 0", str(context.exception))

    def test_get_num_key_value_groups_success(self):
        """测试get_num_key_value_groups方法成功情况"""
        with patch('msmodelslim.model.qwen2_5.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen25ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = DummyConfig()

            result = adapter.get_num_key_value_groups()

            # num_attention_heads=8, num_key_value_heads=4, groups=2
            expected = adapter.config.num_attention_heads // adapter.config.num_key_value_heads
            self.assertEqual(result, expected)
            self.assertEqual(result, 2)

    def test_get_num_key_value_groups_missing_num_attention_heads(self):
        """测试get_num_key_value_groups缺少num_attention_heads时抛出异常"""
        with patch('msmodelslim.model.qwen2_5.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen25ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = type('Config', (), {})()

            with self.assertRaises(InvalidModelError) as context:
                adapter.get_num_key_value_groups()

            self.assertIn("num_attention_heads is not found", str(context.exception))

    def test_get_num_key_value_groups_missing_num_key_value_heads(self):
        """测试get_num_key_value_groups缺少num_key_value_heads时抛出异常"""
        with patch('msmodelslim.model.qwen2_5.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen25ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = type('Config', (), {'num_attention_heads': 8})()

            with self.assertRaises(InvalidModelError) as context:
                adapter.get_num_key_value_groups()

            self.assertIn("num_key_value_heads is not found", str(context.exception))

    def test_get_num_key_value_groups_zero_num_key_value_heads(self):
        """测试get_num_key_value_groups的num_key_value_heads为0时抛出异常"""
        with patch('msmodelslim.model.qwen2_5.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen25ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = type('Config', (), {'num_attention_heads': 8, 'num_key_value_heads': 0})()

            with self.assertRaises(InvalidModelError) as context:
                adapter.get_num_key_value_groups()

            self.assertIn("num_key_value_heads is 0", str(context.exception))

    def test_get_num_key_value_heads_success(self):
        """测试get_num_key_value_heads方法成功情况"""
        with patch('msmodelslim.model.qwen2_5.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen25ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = DummyConfig()

            result = adapter.get_num_key_value_heads()

            self.assertEqual(result, adapter.config.num_key_value_heads)
            self.assertEqual(result, 4)

    def test_get_num_key_value_heads_missing(self):
        """测试get_num_key_value_heads缺少num_key_value_heads时抛出异常"""
        with patch('msmodelslim.model.qwen2_5.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen25ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = type('Config', (), {})()

            with self.assertRaises(InvalidModelError) as context:
                adapter.get_num_key_value_heads()

            self.assertIn("num_key_value_heads is not found", str(context.exception))

    def test_load_tokenizer(self):
        """测试_load_tokenizer方法"""
        with patch('msmodelslim.model.qwen2_5.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen25ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.model_path = self.model_path

            with patch(
                'msmodelslim.model.qwen2_5.model_adapter.SafeGenerator.get_tokenizer_from_pretrained'
            ) as mock_get_tokenizer:
                mock_tokenizer = MagicMock()
                mock_get_tokenizer.return_value = mock_tokenizer

                result = adapter._load_tokenizer(trust_remote_code=True)

                self.assertIs(result, mock_tokenizer)
                mock_get_tokenizer.assert_called_once_with(
                    model_path=str(self.model_path),
                    use_fast=False,
                    legacy=False,
                    padding_side='left',
                    pad_token='<|extra_0|>',
                    eos_token='<|endoftext|>',
                    trust_remote_code=True,
                )
