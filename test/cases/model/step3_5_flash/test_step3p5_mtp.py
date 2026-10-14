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

import torch
from torch import nn

from msmodelslim.model.step3_5_flash.step3p5_mtp import (
    Step3p5RotaryEmbedding,
    Step3p5RMSNorm,
    Step3p5MLP,
    Step3p5Attention,
    Step3p5MTPModule,
    SharedHead,
)


def _make_config(
    hidden_size=64,
    intermediate_size=128,
    num_attention_heads=8,
    num_attention_groups=4,
    max_position_embeddings=128,
    rope_theta=10000.0,
    rms_norm_eps=1e-5,
    sliding_window=None,
    use_head_wise_attn_gate=False,
    use_rope_layers=None,
    layer_types=None,
    rope_scaling=None,
    vocab_size=100,
    partial_rotary_factors=None,
    yarn_only_types=None,
    attention_other_setting=None,
    moe_num_experts=4,
    moe_intermediate_size=32,
    moe_top_k=2,
    use_moe_router_bias=False,
    moe_router_activation="sigmoid",
    need_fp32_gate=False,
):
    if attention_other_setting is None:
        attention_other_setting = {"num_attention_heads": 4, "num_attention_groups": 2}
    # layer_types defaults to empty list so if layer_types: evaluates False (safe for even/odd fallback)
    # yarn_only_types excluded when None to avoid "x not in None" crash in source code
    cfg = SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_attention_groups=num_attention_groups,
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
        rms_norm_eps=rms_norm_eps,
        sliding_window=sliding_window,
        use_head_wise_attn_gate=use_head_wise_attn_gate,
        use_rope_layers=use_rope_layers,
        layer_types=layer_types if layer_types is not None else [],
        rope_scaling=rope_scaling,
        rope_parameters=rope_scaling,
        vocab_size=vocab_size,
        partial_rotary_factors=partial_rotary_factors,
        attention_other_setting=attention_other_setting,
        moe_num_experts=moe_num_experts,
        moe_intermediate_size=moe_intermediate_size,
        moe_top_k=moe_top_k,
        use_moe_router_bias=use_moe_router_bias,
        moe_router_activation=moe_router_activation,
        need_fp32_gate=need_fp32_gate,
    )
    if yarn_only_types is not None:
        cfg.yarn_only_types = yarn_only_types
    return cfg


class TestStep3p5RMSNorm(unittest.TestCase):
    def test_init_when_default_then_weight_is_ones(self):
        """验证 Step3p5RMSNorm 初始化后 weight 为全 1 且形状与 hidden_size 一致"""
        norm = Step3p5RMSNorm(64, eps=1e-5)

        self.assertEqual(norm.weight.shape, (64,))
        self.assertTrue(torch.allclose(norm.weight, torch.ones(64)))
        self.assertEqual(norm.variance_epsilon, 1e-5)

    def test_init_when_different_hidden_size_then_weight_shape_matches(self):
        """验证不同 hidden_size 时 weight 形状正确匹配"""
        norm = Step3p5RMSNorm(128, eps=1e-6)

        self.assertEqual(norm.weight.shape, (128,))
        self.assertEqual(norm.variance_epsilon, 1e-6)


class TestStep3p5MLP(unittest.TestCase):
    def test_init_when_default_then_has_correct_layer_shapes(self):
        """验证 Step3p5MLP 初始化后各投影层形状正确"""
        config = _make_config(hidden_size=64, intermediate_size=128)
        mlp = Step3p5MLP(config)

        self.assertEqual(mlp.gate_proj.weight.shape, (128, 64))
        self.assertEqual(mlp.up_proj.weight.shape, (128, 64))
        self.assertEqual(mlp.down_proj.weight.shape, (64, 128))
        self.assertIsNone(mlp.limit)

    def test_init_when_custom_intermediate_size_then_uses_override(self):
        """验证自定义 intermediate_size 时所有投影层使用该值而非 config 中的值"""
        config = _make_config(hidden_size=64, intermediate_size=128)
        mlp = Step3p5MLP(config, intermediate_size=48)

        self.assertEqual(mlp.gate_proj.weight.shape[0], 48)
        self.assertEqual(mlp.up_proj.weight.shape[0], 48)
        self.assertEqual(mlp.down_proj.weight.shape[1], 48)

    def test_init_when_swiglu_limit_set_then_stored(self):
        """验证设置 swiglu_limit 时该值被正确保存"""
        config = _make_config(hidden_size=64, intermediate_size=128)
        mlp = Step3p5MLP(config, swiglu_limit=1.0)

        self.assertEqual(mlp.limit, 1.0)


class TestStep3p5RotaryEmbedding(unittest.TestCase):
    def test_init_when_rope_theta_float_then_inv_freq_cached(self):
        """验证 rope_theta 为 float 时 Step3p5RotaryEmbedding 有 inv_freq buffer"""
        config = _make_config(rope_theta=10000.0)
        rotary = Step3p5RotaryEmbedding(config, layer_idx=0)

        self.assertTrue(hasattr(rotary, "inv_freq"))
        self.assertIsNotNone(rotary.inv_freq)
        self.assertEqual(rotary.max_seq_len_cached, 128)

    def test_init_when_rope_theta_list_then_selects_by_layer_idx(self):
        """验证 rope_theta 为 list 时 Step3p5RotaryEmbedding 按 layer_idx 使用对应元素，但 self.rope_theta 保存的是完整列表副本"""
        config = _make_config(rope_theta=[10000.0, 20000.0, 30000.0])
        rotary = Step3p5RotaryEmbedding(config, layer_idx=1)

        # self.rope_theta 保存的是 config.rope_theta 的副本（完整列表），而非单个元素
        self.assertEqual(rotary.rope_theta, [10000.0, 20000.0, 30000.0])
        self.assertTrue(hasattr(rotary, "inv_freq"))

    def test_init_when_partial_rotary_factors_set_then_uses_layer_specific_factor(self):
        """验证 partial_rotary_factors 按 layer_idx 选择"""
        config = _make_config(partial_rotary_factors=[0.5, 1.0])
        rotary = Step3p5RotaryEmbedding(config, layer_idx=0)

        self.assertEqual(rotary.config.partial_rotary_factor, 0.5)

    def test_init_when_rope_scaling_dict_then_sets_rope_type_from_scaling(self):
        """验证 rope_scaling 配置了 rope_parameters 时 rope_type 正确从 key 解析"""
        config = _make_config(rope_scaling={"rope_type": "default", "factor": 1.0})
        rotary = Step3p5RotaryEmbedding(config, layer_idx=0)

        self.assertEqual(rotary.rope_type, "default")


class TestStep3p5Attention(unittest.TestCase):
    def test_init_when_default_then_has_all_projections(self):
        """验证 Step3p5Attention 初始化后包含 q/k/v/o 投影层和 Q/K norm"""
        config = _make_config()
        attn = Step3p5Attention(config, layer_idx=0)

        self.assertIsInstance(attn.q_proj, nn.Linear)
        self.assertIsInstance(attn.k_proj, nn.Linear)
        self.assertIsInstance(attn.v_proj, nn.Linear)
        self.assertIsInstance(attn.o_proj, nn.Linear)
        self.assertIsInstance(attn.q_norm, Step3p5RMSNorm)
        self.assertIsInstance(attn.k_norm, Step3p5RMSNorm)
        # head_dim = hidden_size / n_heads = 64 / 8 = 8
        # q_size = n_heads * head_dim = 8 * 8 = 64
        self.assertEqual(attn.q_proj.weight.shape, (64, 64))
        # kv_size = n_kv_heads * head_dim = 4 * 8 = 32
        self.assertEqual(attn.k_proj.weight.shape, (32, 64))
        self.assertEqual(attn.v_proj.weight.shape, (32, 64))

    def test_init_when_sliding_window_enabled_then_uses_other_settings(self):
        """验证 sliding_window 和 enable_sliding_window 时使用 attention_other_setting"""
        config = _make_config(
            sliding_window=128,
            layer_types=["sliding_attention", "full"],
        )
        attn = Step3p5Attention(config, layer_idx=0)

        # Use other setting head counts
        self.assertEqual(attn.num_attention_heads, 4)
        self.assertEqual(attn.num_key_value_heads, 2)
        self.assertIsNotNone(attn.sliding_window)

    def test_init_when_sliding_window_disabled_then_sliding_window_is_none(self):
        """验证非滑动窗口层时 sliding_window 为 None"""
        config = _make_config(
            sliding_window=128,
            layer_types=["sliding_attention", "full"],
        )
        attn = Step3p5Attention(config, layer_idx=1)

        self.assertIsNone(attn.sliding_window)

    def test_init_when_sliding_window_no_layer_types_then_uses_even_odd_logic(self):
        """验证无 layer_types 时按偶/奇层决定 sliding window（偶层开启）"""
        config = _make_config(sliding_window=128)
        attn_even = Step3p5Attention(config, layer_idx=0)  # even → sliding
        attn_odd = Step3p5Attention(config, layer_idx=1)  # odd → no sliding

        self.assertIsNotNone(attn_even.sliding_window)
        self.assertIsNone(attn_odd.sliding_window)

    def test_init_when_use_head_wise_attn_gate_then_has_g_proj(self):
        """验证 use_head_wise_attn_gate=True 时包含 g_proj 层"""
        config = _make_config(use_head_wise_attn_gate=True)
        attn = Step3p5Attention(config, layer_idx=0)

        self.assertIsInstance(attn.g_proj, nn.Linear)
        self.assertIsNotNone(attn.g_proj)

    def test_init_when_use_rope_disabled_by_layer_then_use_rope_is_false(self):
        """验证 use_rope_layers 指定某层不使用 RoPE 时 use_rope=False"""
        config = _make_config(use_rope_layers=[False, True])
        attn = Step3p5Attention(config, layer_idx=0)

        self.assertFalse(attn.use_rope)


class TestSharedHead(unittest.TestCase):
    def test_init_when_default_then_has_norm_and_output(self):
        """验证 SharedHead 包含 norm 和 output 层"""
        config = _make_config(hidden_size=64, vocab_size=100)
        head = SharedHead(config)

        self.assertIsInstance(head.norm, Step3p5RMSNorm)
        self.assertIsInstance(head.output, nn.Linear)
        self.assertEqual(head.output.weight.shape, (100, 64))


class TestStep3p5MTPModule(unittest.TestCase):
    def test_init_when_default_then_has_all_submodules(self):
        """验证 Step3p5MTPModule 初始化后包含所有必要的子模块"""
        config = _make_config()
        mtp = Step3p5MTPModule(config, layer_idx=0)

        self.assertIsInstance(mtp.enorm, Step3p5RMSNorm)
        self.assertIsInstance(mtp.hnorm, Step3p5RMSNorm)
        self.assertIsInstance(mtp.input_layernorm, Step3p5RMSNorm)
        self.assertIsInstance(mtp.eh_proj, nn.Linear)
        self.assertIsInstance(mtp.eh_proj, nn.Linear)
        # eh_proj 输入 channels = hidden_size * 2
        self.assertEqual(mtp.eh_proj.weight.shape[1], 64 * 2)
        self.assertIsInstance(mtp.self_attn, Step3p5Attention)
        self.assertIsInstance(mtp.post_attention_layernorm, Step3p5RMSNorm)
        self.assertTrue(hasattr(mtp.transformer, "shared_head"))
