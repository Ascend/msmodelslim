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
from unittest.mock import patch

import torch

from msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp import (
    Qwen3_5MtpDecoderLayer,
    Qwen3_5MultiTokenPredictor,
    Qwen3_5MtpForCausalLM,
)


class DummyMoeConfig:
    def __init__(self):
        self.hidden_size = 64
        self.num_experts = 2
        self.moe_intermediate_size = 16
        self.shared_expert_intermediate_size = 16
        self.hidden_act = "silu"
        self.num_experts_per_tok = 2
        self.num_attention_heads = 4
        self.num_key_value_heads = 2
        self.rms_norm_eps = 1e-6
        self.intermediate_size = 32
        self.max_position_embeddings = 512
        self.rope_theta = 10000.0
        self.mtp_num_hidden_layers = 1
        self._attn_implementation = 'eager'


class TestQwen3_5MtpDecoderLayer(unittest.TestCase):
    """测试Qwen3_5MtpDecoderLayer的功能"""

    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5Attention', autospec=True)
    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5RMSNorm', autospec=True)
    def test_init_assert_has_self_attn_when_created(self, mock_norm, mock_attn):
        """测试初始化后：应包含self_attn属性"""
        config = DummyMoeConfig()
        layer = Qwen3_5MtpDecoderLayer(config, layer_idx=0)
        self.assertTrue(hasattr(layer, 'self_attn'))

    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5Attention', autospec=True)
    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5RMSNorm', autospec=True)
    def test_init_assert_has_mlp_when_created(self, mock_norm, mock_attn):
        """测试初始化后：应包含mlp属性"""
        config = DummyMoeConfig()
        layer = Qwen3_5MtpDecoderLayer(config, layer_idx=0)
        self.assertTrue(hasattr(layer, 'mlp'))

    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5Attention', autospec=True)
    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5RMSNorm', autospec=True)
    def test_init_assert_has_norms_when_created(self, mock_norm, mock_attn):
        """测试初始化后：应包含input_layernorm和post_attention_layernorm"""
        config = DummyMoeConfig()
        layer = Qwen3_5MtpDecoderLayer(config, layer_idx=0)
        self.assertTrue(hasattr(layer, 'input_layernorm'))
        self.assertTrue(hasattr(layer, 'post_attention_layernorm'))

    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5Attention', autospec=True)
    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5RMSNorm', autospec=True)
    def test_init_assert_mlp_is_moe_when_config_has_experts(self, mock_norm, mock_attn):
        """测试配置包含experts时：mlp应为Qwen3_5MoeSparseMoeBlockWithMLP实例"""
        from msmodelslim.model.qwen3_5_moe.moe_utils import Qwen3_5MoeSparseMoeBlockWithMLP

        config = DummyMoeConfig()
        layer = Qwen3_5MtpDecoderLayer(config, layer_idx=0)
        self.assertIsInstance(layer.mlp, Qwen3_5MoeSparseMoeBlockWithMLP)


class TestQwen3_5MultiTokenPredictor(unittest.TestCase):
    """测试Qwen3_5MultiTokenPredictor的功能"""

    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5Attention', autospec=True)
    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5RMSNorm', autospec=True)
    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5TextRotaryEmbedding', autospec=True)
    def test_init_assert_has_fc_when_created(self, mock_rotary, mock_norm, mock_attn):
        """测试初始化后：应包含fc层且输入输出维度正确"""
        config = DummyMoeConfig()
        predictor = Qwen3_5MultiTokenPredictor(config)
        self.assertTrue(hasattr(predictor, 'fc'))
        self.assertEqual(predictor.fc.in_features, config.hidden_size * 2)
        self.assertEqual(predictor.fc.out_features, config.hidden_size)

    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5Attention', autospec=True)
    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5RMSNorm', autospec=True)
    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5TextRotaryEmbedding', autospec=True)
    def test_init_assert_has_layers_when_created(self, mock_rotary, mock_norm, mock_attn):
        """测试初始化后：应包含正确数量的decoder层"""
        config = DummyMoeConfig()
        predictor = Qwen3_5MultiTokenPredictor(config)
        self.assertTrue(hasattr(predictor, 'layers'))
        self.assertEqual(len(predictor.layers), config.mtp_num_hidden_layers)

    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5Attention', autospec=True)
    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5RMSNorm', autospec=True)
    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5TextRotaryEmbedding', autospec=True)
    def test_init_assert_has_norms_when_created(self, mock_rotary, mock_norm, mock_attn):
        """测试初始化后：应包含pre_fc_norm_embedding、pre_fc_norm_hidden和norm"""
        config = DummyMoeConfig()
        predictor = Qwen3_5MultiTokenPredictor(config)
        self.assertTrue(hasattr(predictor, 'pre_fc_norm_embedding'))
        self.assertTrue(hasattr(predictor, 'pre_fc_norm_hidden'))
        self.assertTrue(hasattr(predictor, 'norm'))

    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5Attention', autospec=True)
    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5RMSNorm', autospec=True)
    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5TextRotaryEmbedding', autospec=True)
    def test_assert_raises_value_error_when_inputs_embeds_is_none(self, mock_rotary, mock_norm, mock_attn):
        """测试inputs_embeds为None时：应抛出ValueError"""
        config = DummyMoeConfig()
        predictor = Qwen3_5MultiTokenPredictor(config)
        hidden_states = torch.randn(1, 4, 64)
        with self.assertRaises(ValueError):
            predictor(hidden_states=hidden_states, inputs_embeds=None)


class TestQwen3_5MtpForCausalLM(unittest.TestCase):
    """测试Qwen3_5MtpForCausalLM的功能"""

    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5Attention', autospec=True)
    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5RMSNorm', autospec=True)
    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5TextRotaryEmbedding', autospec=True)
    def test_init_assert_has_mtp_when_created(self, mock_rotary, mock_norm, mock_attn):
        """测试初始化后：应包含mtp属性且为Qwen3_5MultiTokenPredictor实例"""
        config = DummyMoeConfig()
        model = Qwen3_5MtpForCausalLM(config)
        self.assertTrue(hasattr(model, 'mtp'))
        self.assertIsInstance(model.mtp, Qwen3_5MultiTokenPredictor)

    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5Attention', autospec=True)
    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5RMSNorm', autospec=True)
    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5TextRotaryEmbedding', autospec=True)
    def test_remap_weight_name_when_mtp_prefix(self, mock_rotary, mock_norm, mock_attn):
        """测试_remap_weight_name方法：mtp前缀的名称应原样返回"""
        config = DummyMoeConfig()
        model = Qwen3_5MtpForCausalLM(config)
        result = model._remap_weight_name("mtp.fc.weight")
        self.assertEqual(result, "mtp.fc.weight")

    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5Attention', autospec=True)
    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5RMSNorm', autospec=True)
    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5TextRotaryEmbedding', autospec=True)
    def test_remap_weight_name_when_non_mtp_prefix(self, mock_rotary, mock_norm, mock_attn):
        """测试_remap_weight_name方法：非mtp前缀的名称应返回None"""
        config = DummyMoeConfig()
        model = Qwen3_5MtpForCausalLM(config)
        result = model._remap_weight_name("model.layers.0.weight")
        self.assertIsNone(result)

    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5Attention', autospec=True)
    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5RMSNorm', autospec=True)
    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5TextRotaryEmbedding', autospec=True)
    def test_load_weights_when_gate_up_proj_packed(self, mock_rotary, mock_norm, mock_attn):
        """测试load_weights方法：gate_up_proj打包权重应被正确拆分加载"""
        config = DummyMoeConfig()
        model = Qwen3_5MtpForCausalLM(config)
        num_experts = config.num_experts
        gate_up_weight = torch.randn(num_experts, 2 * config.moe_intermediate_size, config.hidden_size)
        weights = {"mtp.layers.0.mlp.experts.gate_up_proj.weight": gate_up_weight}
        loaded = model.load_weights(weights)
        self.assertTrue(len(loaded) > 0)

    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5Attention', autospec=True)
    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5RMSNorm', autospec=True)
    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5TextRotaryEmbedding', autospec=True)
    def test_load_weights_when_down_proj_packed(self, mock_rotary, mock_norm, mock_attn):
        """测试load_weights方法：down_proj打包权重应被正确拆分加载"""
        config = DummyMoeConfig()
        model = Qwen3_5MtpForCausalLM(config)
        num_experts = config.num_experts
        down_weight = torch.randn(num_experts, config.hidden_size, config.moe_intermediate_size)
        weights = {"mtp.layers.0.mlp.experts.down_proj.weight": down_weight}
        loaded = model.load_weights(weights)
        self.assertTrue(len(loaded) > 0)

    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5Attention', autospec=True)
    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5RMSNorm', autospec=True)
    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5TextRotaryEmbedding', autospec=True)
    def test_load_weights_when_direct_weight(self, mock_rotary, mock_norm, mock_attn):
        """测试load_weights方法：直接权重应被正确加载"""
        config = DummyMoeConfig()
        model = Qwen3_5MtpForCausalLM(config)
        fc_weight = torch.randn(config.hidden_size, config.hidden_size * 2)
        weights = {"mtp.fc.weight": fc_weight}
        loaded = model.load_weights(weights)
        self.assertIn("mtp.fc.weight", loaded)

    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5Attention', autospec=True)
    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5RMSNorm', autospec=True)
    @patch('msmodelslim.model.qwen3_5_moe.modeling_qwen3_5_mtp.Qwen3_5TextRotaryEmbedding', autospec=True)
    def test_load_weights_when_rotary_emb_skipped(self, mock_rotary, mock_norm, mock_attn):
        """测试load_weights方法：rotary_emb.inv_freq应被跳过"""
        config = DummyMoeConfig()
        model = Qwen3_5MtpForCausalLM(config)
        weights = {"mtp.rotary_emb.inv_freq": torch.randn(4, 2)}
        loaded = model.load_weights(weights)
        self.assertNotIn("mtp.rotary_emb.inv_freq", loaded)


if __name__ == '__main__':
    unittest.main()
