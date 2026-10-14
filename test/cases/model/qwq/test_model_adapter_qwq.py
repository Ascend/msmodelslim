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
# pylint: disable=consider-using-from-import

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch.nn as nn

from msmodelslim.core.const import DeviceType
from msmodelslim.model.qwq.model_adapter import QwqModelAdapter


class TestQwqModelAdapter(unittest.TestCase):
    """测试QwqModelAdapter的功能"""

    def setUp(self):
        """测试前的准备工作"""
        self.model_type = 'QwQ-32B'
        self.model_path = Path('.')
        self.trust_remote_code = False

    def test_get_model_type_when_initialized_then_return_model_type(self):
        """测试get_model_type方法：初始化后应返回正确的模型类型"""
        with patch('msmodelslim.model.qwq.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = QwqModelAdapter(
                model_type=self.model_type, model_path=self.model_path, trust_remote_code=self.trust_remote_code
            )
            adapter.model_type = self.model_type

            result = adapter.get_model_type()

            self.assertEqual(result, self.model_type)

    def test_get_model_pedigree_when_called_then_return_qwq(self):
        """测试get_model_pedigree方法：应返回'qwq'"""
        with patch('msmodelslim.model.qwq.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = QwqModelAdapter(model_type=self.model_type, model_path=self.model_path)

            result = adapter.get_model_pedigree()

            self.assertEqual(result, 'qwq')

    def test_load_model_with_npu_device_when_called_then_delegate_to_load_model(self):
        """测试load_model方法：使用NPU设备时应委托给_load_model方法"""
        with patch('msmodelslim.model.qwq.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = QwqModelAdapter(model_type=self.model_type, model_path=self.model_path)

            mock_model = nn.Linear(10, 10)
            adapter._load_model = MagicMock(return_value=mock_model)

            result = adapter.load_model(device=DeviceType.NPU)

            self.assertIs(result, mock_model)
            adapter._load_model.assert_called_once_with(DeviceType.NPU)

    def test_handle_dataset_when_called_then_return_tokenized_data(self):
        """测试handle_dataset方法：应返回tokenized数据"""
        with patch('msmodelslim.model.qwq.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = QwqModelAdapter(model_type=self.model_type, model_path=self.model_path)

            mock_dataset = ['data1', 'data2']
            adapter._get_tokenized_data = MagicMock(return_value=mock_dataset)

            result = adapter.handle_dataset(dataset='test_data', device=DeviceType.CPU)

            self.assertEqual(result, mock_dataset)
            adapter._get_tokenized_data.assert_called_once_with('test_data', DeviceType.CPU)

    def test_handle_dataset_by_batch_when_called_then_return_batch_tokenized_data(self):
        """测试handle_dataset_by_batch方法：应返回批量tokenized数据"""
        with patch('msmodelslim.model.qwq.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = QwqModelAdapter(model_type=self.model_type, model_path=self.model_path)

            mock_batch_dataset = [['batch1'], ['batch2']]
            adapter._get_batch_tokenized_data = MagicMock(return_value=mock_batch_dataset)

            result = adapter.handle_dataset_by_batch(dataset='test_data', batch_size=2, device=DeviceType.CPU)

            self.assertEqual(result, mock_batch_dataset)
            adapter._get_batch_tokenized_data.assert_called_once_with(
                calib_list='test_data', batch_size=2, device=DeviceType.CPU
            )

    def test_init_model_when_called_then_delegate_to_load_model(self):
        """测试init_model方法：应委托给_load_model方法"""
        with patch('msmodelslim.model.qwq.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = QwqModelAdapter(model_type=self.model_type, model_path=self.model_path)

            mock_model = nn.Linear(10, 10)
            adapter._load_model = MagicMock(return_value=mock_model)

            result = adapter.init_model(device=DeviceType.NPU)

            self.assertIs(result, mock_model)
            adapter._load_model.assert_called_once_with(DeviceType.NPU)

    def test_enable_kv_cache_when_called_with_true_then_enable_cache(self):
        """测试enable_kv_cache方法：传入True时应启用缓存"""
        with patch('msmodelslim.model.qwq.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = QwqModelAdapter(model_type=self.model_type, model_path=self.model_path)

            mock_model = nn.Linear(10, 10)
            adapter._enable_kv_cache = MagicMock(return_value=None)

            result = adapter.enable_kv_cache(model=mock_model, need_kv_cache=True)  # noqa: F841

            # 验证_enable_kv_cache被调用
            adapter._enable_kv_cache.assert_called_once_with(mock_model, True)

    def test_enable_kv_cache_when_called_with_false_then_disable_cache(self):
        """测试enable_kv_cache方法：传入False时应禁用缓存"""
        with patch('msmodelslim.model.qwq.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = QwqModelAdapter(model_type=self.model_type, model_path=self.model_path)

            mock_model = nn.Linear(10, 10)
            adapter._enable_kv_cache = MagicMock(return_value=None)

            adapter.enable_kv_cache(model=mock_model, need_kv_cache=False)

            # 验证参数正确传递
            adapter._enable_kv_cache.assert_called_once_with(mock_model, False)

    def test_handle_dataset_with_empty_dataset_when_called_then_return_empty_result(self):
        """测试handle_dataset方法：空数据集时应返回空结果"""
        with patch('msmodelslim.model.qwq.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = QwqModelAdapter(model_type=self.model_type, model_path=self.model_path)

            mock_dataset = []
            adapter._get_tokenized_data = MagicMock(return_value=mock_dataset)

            result = adapter.handle_dataset(dataset='', device=DeviceType.CPU)

            self.assertEqual(result, [])

    def test_get_hidden_dim_when_config_has_hidden_size_then_return_hidden_size(self):
        """测试get_hidden_dim方法：应返回hidden_size"""
        with patch(
            "msmodelslim.model.qwq.model_adapter.DefaultModelAdapter.__init__",
            return_value=None,
        ):
            adapter = QwqModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = type("Config", (), {"hidden_size": 4096})()

            result = adapter.get_hidden_dim()

            self.assertEqual(result, 4096)

    def test_get_head_dim_when_config_has_head_dim_then_return_head_dim(self):
        """测试get_head_dim方法：配置中有head_dim时直接返回"""
        with patch(
            "msmodelslim.model.qwq.model_adapter.DefaultModelAdapter.__init__",
            return_value=None,
        ):
            adapter = QwqModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = type("Config", (), {"head_dim": 128})()

            result = adapter.get_head_dim()

            self.assertEqual(result, 128)

    def test_get_head_dim_when_head_dim_missing_then_return_derived_value(self):
        """测试get_head_dim方法：head_dim缺失时回退到hidden_size/num_attention_heads"""
        with patch(
            "msmodelslim.model.qwq.model_adapter.DefaultModelAdapter.__init__",
            return_value=None,
        ):
            adapter = QwqModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = type("Config", (), {"hidden_size": 4096, "num_attention_heads": 32})()

            result = adapter.get_head_dim()

            self.assertEqual(result, 128)

    def test_get_num_attention_heads_when_config_has_heads_then_return_heads(self):
        """测试get_num_attention_heads方法：应返回注意力头数"""
        with patch(
            "msmodelslim.model.qwq.model_adapter.DefaultModelAdapter.__init__",
            return_value=None,
        ):
            adapter = QwqModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = type("Config", (), {"num_attention_heads": 40})()
            adapter.model_path = self.model_path

            result = adapter.get_num_attention_heads()

            self.assertEqual(result, 40)

    def test_load_tokenizer_when_called_then_use_qwen_tokenizer_settings(self):
        """测试_load_tokenizer方法：应使用Qwen系tokenizer参数"""
        with (
            patch(
                "msmodelslim.model.qwq.model_adapter.DefaultModelAdapter.__init__",
                return_value=None,
            ),
            patch(
                "msmodelslim.model.qwq.model_adapter.SafeGenerator.get_tokenizer_from_pretrained",
                return_value=MagicMock(name="tokenizer"),
            ) as mock_get_tokenizer,
        ):
            adapter = QwqModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.model_path = self.model_path

            tokenizer = adapter._load_tokenizer(trust_remote_code=True)

            self.assertIsNotNone(tokenizer)
            mock_get_tokenizer.assert_called_once_with(
                model_path=str(self.model_path),
                use_fast=False,
                legacy=False,
                padding_side="left",
                pad_token="<|extra_0|>",
                eos_token="<|endoftext|>",
                trust_remote_code=True,
            )

    def test_get_ln_fuse_map_when_called_then_return_expected_mapping(self):
        """测试get_ln_fuse_map方法：应返回QwQ的LayerNorm融合映射"""
        with patch(
            "msmodelslim.model.qwq.model_adapter.DefaultModelAdapter.__init__",
            return_value=None,
        ):
            adapter = QwqModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = type("Config", (), {"num_hidden_layers": 2})()

            pre_run, fused_map = adapter.get_ln_fuse_map()

            self.assertEqual(pre_run, {})
            self.assertIn("model.layers.0.input_layernorm", fused_map)
            self.assertIn("model.layers.0.post_attention_layernorm", fused_map)
            self.assertIn("model.norm", fused_map)
            self.assertEqual(
                fused_map["model.layers.0.input_layernorm"],
                [
                    "model.layers.0.self_attn.q_proj",
                    "model.layers.0.self_attn.k_proj",
                    "model.layers.0.self_attn.v_proj",
                ],
            )
            self.assertEqual(fused_map["model.norm"], ["lm_head"])

    def test_get_adapter_config_for_subgraph_when_called_then_return_expected_configs(
        self,
    ):
        """测试get_adapter_config_for_subgraph方法：应返回iter_smooth需要的子图配置"""
        with patch(
            "msmodelslim.model.qwq.model_adapter.DefaultModelAdapter.__init__",
            return_value=None,
        ):
            adapter = QwqModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = type("Config", (), {"num_hidden_layers": 2})()

            adapter_configs = adapter.get_adapter_config_for_subgraph()

            self.assertEqual(len(adapter_configs), 6)
            self.assertEqual(adapter_configs[0].subgraph_type, "norm-linear")
            self.assertEqual(adapter_configs[1].subgraph_type, "norm-linear")
            self.assertEqual(adapter_configs[2].subgraph_type, "up-down")
            self.assertEqual(adapter_configs[0].mapping.source, "model.layers.0.input_layernorm")
            self.assertEqual(adapter_configs[2].mapping.targets, ["model.layers.0.mlp.down_proj"])

    def test_get_rotate_map_when_called_then_return_pre_run_and_rotate_pairs(self):
        """测试get_rotate_map方法：应返回预旋转和层内旋转映射"""
        with patch(
            "msmodelslim.model.qwq.model_adapter.DefaultModelAdapter.__init__",
            return_value=None,
        ):
            adapter = QwqModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = type(
                "Config",
                (),
                {
                    "num_hidden_layers": 2,
                    "hidden_size": 128,
                    "num_attention_heads": 8,
                    "head_dim": 16,
                },
            )()

            pre_run_list, rot_pairs_list = adapter.get_rotate_map(block_size=8)

            self.assertIsInstance(pre_run_list, list)
            self.assertEqual(len(pre_run_list), 1)
            self.assertIn("model.embed_tokens", pre_run_list[0].right_rot)

            self.assertIsInstance(rot_pairs_list, list)
            self.assertEqual(len(rot_pairs_list), 2)
            self.assertIn("lm_head", rot_pairs_list[0].right_rot)
            self.assertIn("model.layers.0.self_attn.o_proj", rot_pairs_list[0].left_rot)
            self.assertIn("model.layers.0.self_attn.o_proj", rot_pairs_list[1].right_rot)
