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
from msmodelslim.core.graph import AdapterConfig
from msmodelslim.model.glm4_moe.model_adapter import GLM4MoeModelAdapter


class DummyConfig:
    """模拟配置对象"""

    def __init__(self):
        self.num_hidden_layers = 3


class TestGLM4MoeModelAdapter(unittest.TestCase):
    """测试GLM4MoeModelAdapter的功能"""

    def setUp(self):
        """测试前的准备工作"""
        self.model_type = 'GLM-4.7'
        self.model_path = Path('.')
        self.trust_remote_code = False

    def test_get_model_type_when_initialized_then_return_model_type(self):
        """测试get_model_type方法：初始化后应返回正确的模型类型"""
        with patch('msmodelslim.model.glm4_moe.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = GLM4MoeModelAdapter(
                model_type=self.model_type, model_path=self.model_path, trust_remote_code=self.trust_remote_code
            )
            adapter.model_type = self.model_type

            result = adapter.get_model_type()

            self.assertEqual(result, self.model_type)

    def test_get_model_pedigree_when_called_then_return_glm4_moe(self):
        """测试get_model_pedigree方法：应返回'glm4_moe'"""
        with patch('msmodelslim.model.glm4_moe.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = GLM4MoeModelAdapter(
                model_type=self.model_type, model_path=self.model_path, trust_remote_code=self.trust_remote_code
            )

            result = adapter.get_model_pedigree()

            self.assertEqual(result, 'glm4_moe')

    def test_handle_dataset_when_called_then_return_tokenized_data(self):
        """测试handle_dataset方法：应返回tokenized数据"""
        with patch('msmodelslim.model.glm4_moe.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = GLM4MoeModelAdapter(model_type=self.model_type, model_path=self.model_path)

            mock_dataset = ['data1', 'data2']
            adapter._get_tokenized_data = MagicMock(return_value=mock_dataset)

            result = adapter.handle_dataset(dataset='test_data', device=DeviceType.CPU)

            self.assertEqual(result, mock_dataset)
            adapter._get_tokenized_data.assert_called_once_with('test_data', DeviceType.CPU)

    def test_init_model_when_called_then_delegate_to_load_model(self):
        """测试init_model方法：应委托给_load_model方法"""
        with patch('msmodelslim.model.glm4_moe.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = GLM4MoeModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = DummyConfig()

            # 模拟一个具有完整模型结构的对象，包含 model.layers (nn.ModuleList)
            class MockDecoderLayer(nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__()

                def forward(self, x):
                    return x

            mock_layer = MockDecoderLayer()
            mock_model = MagicMock(spec=nn.Module)
            mock_model.model = MagicMock(spec=nn.Module)
            mock_model.model.layers = nn.ModuleList([mock_layer])
            adapter._load_model = MagicMock(return_value=mock_model)

            result = adapter.init_model(device=DeviceType.NPU)

            self.assertIs(result, mock_model)
            adapter._load_model.assert_called_once_with(DeviceType.NPU)

    def test_enable_kv_cache_when_called_then_delegate_to_enable_kv_cache(self):
        """测试enable_kv_cache方法：应委托给_enable_kv_cache方法"""
        with patch('msmodelslim.model.glm4_moe.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = GLM4MoeModelAdapter(model_type=self.model_type, model_path=self.model_path)

            mock_model = nn.Linear(10, 10)
            adapter._enable_kv_cache = MagicMock(return_value=None)

            adapter.enable_kv_cache(model=mock_model, need_kv_cache=True)

            adapter._enable_kv_cache.assert_called_once_with(mock_model, True)

    def test_enable_kv_cache_with_false_then_disable_cache(self):
        """测试enable_kv_cache方法：传入False时应禁用缓存"""
        with patch('msmodelslim.model.glm4_moe.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = GLM4MoeModelAdapter(model_type=self.model_type, model_path=self.model_path)

            mock_model = nn.Linear(10, 10)
            adapter._enable_kv_cache = MagicMock(return_value=None)

            adapter.enable_kv_cache(model=mock_model, need_kv_cache=False)

            # 验证参数正确传递
            adapter._enable_kv_cache.assert_called_once_with(mock_model, False)

    def test_get_adapter_config_for_subgraph_when_called_then_return_adapter_configs(self):
        """测试get_adapter_config_for_subgraph方法：应返回适配器配置列表"""
        with patch('msmodelslim.model.glm4_moe.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = GLM4MoeModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = DummyConfig()

            result = adapter.get_adapter_config_for_subgraph()

            # 验证返回列表
            self.assertIsInstance(result, list)

            # 每层应该有2个配置（Norm-Linear融合 + OV融合）
            expected_configs = adapter.config.num_hidden_layers * 2
            self.assertEqual(len(result), expected_configs)

            # 验证第一个配置的类型
            self.assertIsInstance(result[0], AdapterConfig)

    def test_get_adapter_config_for_subgraph_when_zero_layers_then_return_empty_list(self):
        """测试get_adapter_config_for_subgraph方法：0层时应返回空列表"""
        with patch('msmodelslim.model.glm4_moe.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = GLM4MoeModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = DummyConfig()
            adapter.config.num_hidden_layers = 0

            result = adapter.get_adapter_config_for_subgraph()

            # 验证返回空列表
            self.assertEqual(len(result), 0)
            self.assertIsInstance(result, list)


class TestGLM4MoeModuleFunctions(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        self.model_type = 'GLM-4.7'
        self.model_path = Path('.')
        self.config = DummyConfig()
        self.config.num_experts = 4
        self.config.hidden_size = 128
        self.config.head_dim = 64

    def test_glm4_moe_get_ln_fuse_map_when_called_then_return_correct_mapping(self):
        """测试glm4_moe_get_ln_fuse_map方法：调用时应返回正确的融合映射"""
        from msmodelslim.model.glm4_moe.model_adapter import glm4_moe_get_ln_fuse_map

        result = glm4_moe_get_ln_fuse_map(self.config)

        # 验证返回字典
        self.assertIsInstance(result, dict)

        # 验证包含必要的键
        self.assertIn("model.norm", result)
        self.assertIn("model.layers.0.input_layernorm", result)
        self.assertIn("model.layers.0.post_attention_layernorm", result)

        # 验证input_layernorm的映射
        input_ln_targets = result["model.layers.0.input_layernorm"]
        expected_input_targets = [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
        ]
        for target in expected_input_targets:
            self.assertIn(target, input_ln_targets)

        # 验证post_attention_layernorm的映射包含专家和gate
        post_ln_targets = result["model.layers.0.post_attention_layernorm"]
        self.assertIn("model.layers.0.mlp.gate", post_ln_targets)

        # 验证专家数量正确
        expert_targets = [t for t in post_ln_targets if "experts" in t]
        self.assertEqual(len(expert_targets), self.config.num_experts * 2)  # gate_proj + up_proj

    def test_glm4_moe_get_rotate_map_when_called_then_return_correct_rotate_commands(self):
        """测试glm4_moe_get_rotate_map方法：调用时应返回正确的旋转命令"""
        from msmodelslim.model.glm4_moe.model_adapter import glm4_moe_get_rotate_map
        from msmodelslim.processor.quarot import QuaRotInterface

        block_size = 64
        pre_run, rot_pairs, rot, rot_uv = glm4_moe_get_rotate_map(self.config, block_size)

        # 验证返回类型
        self.assertIsInstance(pre_run, QuaRotInterface.RotatePair)
        self.assertIsInstance(rot_pairs, dict)

        # 验证pre_run包含embed_tokens
        self.assertIn("model.embed_tokens", pre_run.right_rot)

        # 验证rot_pairs包含rot和rot_uv
        self.assertIn("rot", rot_pairs)
        self.assertIn("rot_uv", rot_pairs)

        # 验证rot包含正确的层数
        rot_pair = rot_pairs["rot"]
        self.assertIn("lm_head", rot_pair.right_rot)

        # 验证包含self_attn层
        for layer_idx in range(self.config.num_hidden_layers):
            for proj in ["q_proj", "k_proj", "v_proj"]:
                key = f"model.layers.{layer_idx}.self_attn.{proj}"
                self.assertIn(key, rot_pair.right_rot)

            # 验证o_proj在left_rot中
            o_proj_key = f"model.layers.{layer_idx}.self_attn.o_proj"
            self.assertIn(o_proj_key, rot_pair.left_rot)

            # 验证专家层
            for i in range(self.config.num_experts):
                for proj in ["gate_proj", "up_proj"]:
                    key = f"model.layers.{layer_idx}.mlp.experts.{i}.{proj}"
                    self.assertIn(key, rot_pair.right_rot)

                down_proj_key = f"model.layers.{layer_idx}.mlp.experts.{i}.down_proj"
                self.assertIn(down_proj_key, rot_pair.left_rot)

            # 验证gate
            gate_key = f"model.layers.{layer_idx}.mlp.gate"
            self.assertIn(gate_key, rot_pair.right_rot)

    def test_get_adapter_config_for_subgraph_with_mtp_then_include_mtp_layer(self):
        """测试get_adapter_config_for_subgraph：含MTP层时应多2个配置"""
        with patch('msmodelslim.model.glm4_moe.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = GLM4MoeModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = DummyConfig()

            result = adapter.get_adapter_config_for_subgraph()

            self.assertIsInstance(result, list)
            expected_configs = adapter.config.num_hidden_layers * 2
            self.assertEqual(len(result), expected_configs)
            self.assertIsInstance(result[0], AdapterConfig)


if __name__ == '__main__':
    unittest.main()
