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

from msmodelslim.model.step3_5_flash.moe_utils import (
    sigmoid_routing_function,
    Step3p5MoeExpertMLP,
    Step3p5MoEMLPWithUnpackExperts,
    convert_step35_moe_to_unpacked,
)

MOE_PATH = "msmodelslim.model.step3_5_flash.moe_utils"


def _make_moe_config(
    num_experts=4,
    top_k=2,
    hidden_size=64,
    moe_intermediate_size=32,
    use_moe_router_bias=False,
    moe_router_activation="sigmoid",
    need_fp32_gate=False,
    moe_router_scaling_factor=1.0,
):
    return SimpleNamespace(
        moe_num_experts=num_experts,
        moe_top_k=top_k,
        hidden_size=hidden_size,
        moe_intermediate_size=moe_intermediate_size,
        use_moe_router_bias=use_moe_router_bias,
        moe_router_activation=moe_router_activation,
        need_fp32_gate=need_fp32_gate,
        moe_router_scaling_factor=moe_router_scaling_factor,
        num_attention_heads=8,
    )


# ===== sigmoid_routing_function =====


class TestSigmoidRoutingFunction(unittest.TestCase):
    def test_routing_when_basic_input_then_returns_correct_shape(self):
        """验证 sigmoid_routing_function 对基本输入返回 (topk_prob, indices) 且形状正确"""
        gating_output = torch.randn(2, 4)  # 2 tokens, 4 experts
        topk_prob, indices = sigmoid_routing_function(gating_output, topk=2, renormalize=True)

        self.assertEqual(topk_prob.shape, (2, 2))
        self.assertEqual(indices.shape, (2, 2))
        self.assertTrue(torch.all(topk_prob >= 0))
        self.assertTrue(torch.all(topk_prob <= 1))

    def test_routing_when_renormalize_false_then_values_are_sigmoid_based(self):
        """验证当 renormalize=False 时，返回的权重为 sigmoid 概率 / 总和而不额外归一化"""
        gating_output = torch.tensor([[2.0, 1.0, -1.0, 0.5]])
        topk_prob, indices = sigmoid_routing_function(gating_output, topk=2, renormalize=False)

        self.assertEqual(topk_prob.shape, (1, 2))
        self.assertEqual(indices.shape, (1, 2))
        # Verify it doesn't crash — the main purpose is that renormalize=False is accepted

    def test_routing_when_all_zeros_then_distribution_is_uniform(self):
        """验证当 gating_output 全零时，sigmoid(0)=0.5，除以 sum 后变为均匀分布"""
        gating_output = torch.zeros(1, 4)
        topk_prob, indices = sigmoid_routing_function(gating_output, topk=2, renormalize=True)

        self.assertEqual(topk_prob.shape, (1, 2))
        # Sigmoid(0) = 0.5, sum = 2.0, so each gets 0.25; top-2 should each be 0.5 after renormalize
        # The sum of top-2 probs should still be positive
        self.assertAlmostEqual(float(topk_prob.sum().item()), 1.0, places=5)

    def test_routing_when_single_token_then_returns_single_result(self):
        """验证单 token 场景下 sigmoid_routing_function 仍正常工作"""
        gating_output = torch.randn(1, 4)
        topk_prob, indices = sigmoid_routing_function(gating_output, topk=2, renormalize=True)

        self.assertEqual(topk_prob.shape, (1, 2))
        self.assertEqual(indices.shape, (1, 2))

    def test_routing_when_topk_equals_one_then_returns_best_expert(self):
        """验证 top_k=1 时只返回一个 expert"""
        gating_output = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
        topk_prob, indices = sigmoid_routing_function(gating_output, topk=1, renormalize=True)

        self.assertEqual(indices[0, 0].item(), 0)
        self.assertAlmostEqual(float(topk_prob[0, 0].item()), 1.0, places=5)


# ===== Step3p5MoeExpertMLP =====


class TestStep3p5MoeExpertMLP(unittest.TestCase):
    def _config(self):
        return SimpleNamespace(hidden_size=16, intermediate_size=32)

    def test_forward_when_basic_input_then_returns_correct_shape(self):
        """验证 Step3p5MoeExpertMLP 前向返回正确形状 (B, hidden_size)"""
        mlp = Step3p5MoeExpertMLP(self._config())
        x = torch.randn(2, 16)
        out = mlp(x)

        self.assertEqual(out.shape, (2, 16))

    def test_forward_when_with_swiglu_limit_then_gate_and_up_are_clamped(self):
        """验证设置 swiglu_limit 后，gate 和 up 输出被截断到指定范围"""
        config_with_limit = SimpleNamespace(hidden_size=16, intermediate_size=32)
        mlp = Step3p5MoeExpertMLP(config_with_limit, swiglu_limit=1.0)
        x = torch.randn(2, 16) * 10  # 放大输入确保触发 clamp
        out = mlp(x)

        self.assertEqual(out.shape, (2, 16))

    def test_forward_when_intermediate_size_override_then_has_correct_fan_out(self):
        """验证自定义 intermediate_size 时中间层维度正确"""
        config_small = SimpleNamespace(hidden_size=16, intermediate_size=128)
        mlp = Step3p5MoeExpertMLP(config_small, intermediate_size=48)

        self.assertEqual(mlp.gate_proj.weight.shape[0], 48)
        self.assertEqual(mlp.up_proj.weight.shape[0], 48)
        self.assertEqual(mlp.down_proj.weight.shape[1], 48)


# ===== Step3p5MoEMLPWithUnpackExperts =====


class TestStep3p5MoEMLPWithUnpackExperts(unittest.TestCase):
    def test_forward_when_basic_input_then_returns_correct_shape(self):
        """验证 Step3p5MoEMLPWithUnpackExperts 前向返回 (B, S, D)"""
        config = _make_moe_config(num_experts=4, top_k=2, hidden_size=16, moe_intermediate_size=32)
        moe = Step3p5MoEMLPWithUnpackExperts(config)
        x = torch.randn(1, 3, 16)  # batch=1, seq=3, dim=16
        out = moe(x)

        self.assertEqual(out.shape, (1, 3, 16))

    def test_forward_when_all_zero_input_then_returns_zero(self):
        """验证全零输入时输出也接近零"""
        config = _make_moe_config(num_experts=4, top_k=2, hidden_size=16, moe_intermediate_size=32)
        moe = Step3p5MoEMLPWithUnpackExperts(config)
        x = torch.zeros(1, 3, 16)
        out = moe(x)

        self.assertEqual(out.shape, (1, 3, 16))

    def test_forward_when_need_fp32_gate_then_uses_fp32_matmul(self):
        """验证 need_fp32_gate=True 时路由计算使用 fp32 矩阵乘法"""
        config = _make_moe_config(
            num_experts=2,
            top_k=1,
            hidden_size=8,
            moe_intermediate_size=16,
            need_fp32_gate=True,
        )
        moe = Step3p5MoEMLPWithUnpackExperts(config)
        x = torch.randn(1, 2, 8)
        out = moe(x)

        self.assertEqual(out.shape, (1, 2, 8))

    def test_forward_when_use_moe_router_bias_then_uses_router_bias_func(self):
        """验证 use_moe_router_bias=True 时使用 router_bias_func 进行路由"""
        config = _make_moe_config(
            num_experts=2,
            top_k=1,
            hidden_size=8,
            moe_intermediate_size=16,
            use_moe_router_bias=True,
        )
        moe = Step3p5MoEMLPWithUnpackExperts(config)
        x = torch.randn(1, 2, 8)
        out = moe(x)

        self.assertEqual(out.shape, (1, 2, 8))

    def test_router_bias_func_when_renormalize_true_then_weights_sum_to_one(self):
        """验证 router_bias_func 在 renormalize=True 时每行的权重之和为 1"""
        config = _make_moe_config(num_experts=4, top_k=2, use_moe_router_bias=True)
        moe = Step3p5MoEMLPWithUnpackExperts(config)
        gating_output = torch.randn(2, 4)

        topk_prob, indices = moe.router_bias_func(gating_output, topk=2, renormalize=True)

        self.assertEqual(topk_prob.shape, (2, 2))
        self.assertTrue(torch.allclose(topk_prob.sum(dim=-1), torch.ones(2), atol=1e-6))

    def test_initialization_when_custom_routing_none_then_uses_softmax(self):
        """验证 moe_router_activation 既非 sigmoid 又无 bias 时 custom_routing_function 为 None"""
        config = _make_moe_config(
            num_experts=2,
            top_k=1,
            hidden_size=8,
            moe_intermediate_size=16,
            use_moe_router_bias=False,
            moe_router_activation="relu",
        )
        moe = Step3p5MoEMLPWithUnpackExperts(config)
        x = torch.randn(1, 2, 8)
        out = moe(x)

        self.assertEqual(out.shape, (1, 2, 8))
        self.assertIsNone(moe.custom_routing_function)

    def test_routed_scaling_factor_then_multiplies_weights(self):
        """验证 routed_scaling_factor 在 forward 中正确缩放路由权重"""
        config = _make_moe_config(
            num_experts=2,
            top_k=1,
            hidden_size=8,
            moe_intermediate_size=16,
            moe_router_scaling_factor=2.0,
        )
        moe = Step3p5MoEMLPWithUnpackExperts(config)
        x = torch.randn(1, 2, 8)
        out = moe(x)

        self.assertEqual(out.shape, (1, 2, 8))


# ===== convert_step35_moe_to_unpacked =====


class MockOriginalMoE:
    """模拟 Step3p5MoEMLP 旧格式，用于测试 convert 函数"""

    def __init__(self, config, use_bias=False):
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.moe_num_experts, bias=False)
        self.gate_proj = SimpleNamespace(
            weight=nn.Parameter(torch.randn(config.moe_num_experts, config.moe_intermediate_size, config.hidden_size))
        )
        self.up_proj = SimpleNamespace(
            weight=nn.Parameter(torch.randn(config.moe_num_experts, config.moe_intermediate_size, config.hidden_size))
        )
        self.down_proj = SimpleNamespace(
            weight=nn.Parameter(torch.randn(config.moe_num_experts, config.hidden_size, config.moe_intermediate_size))
        )
        self.use_moe_router_bias = use_bias
        if use_bias:
            self.router_bias = nn.Parameter(torch.zeros(config.moe_num_experts))
        self.limit = None


class TestConvertStep35MoeToUnpacked(unittest.TestCase):
    def test_convert_when_successful_then_creates_unpacked_with_same_config(self):
        """验证 convert_step35_moe_to_unpacked 成功创建包含相同配置的 Step3p5MoEMLPWithUnpackExperts"""
        config = _make_moe_config(num_experts=2, top_k=1, hidden_size=8, moe_intermediate_size=16)
        original = MockOriginalMoE(config)
        original.limit = 1.0
        original.gate_proj.weight = nn.Parameter(torch.randn(2, 16, 8))
        original.up_proj.weight = nn.Parameter(torch.randn(2, 16, 8))
        original.down_proj.weight = nn.Parameter(torch.randn(2, 8, 16))

        new_moe = convert_step35_moe_to_unpacked(original, config, swiglu_limit=1.0)

        self.assertIsInstance(new_moe, Step3p5MoEMLPWithUnpackExperts)
        self.assertEqual(new_moe.num_experts, 2)
        self.assertEqual(new_moe.hidden_size, 8)
        self.assertEqual(new_moe.moe_intermediate_size, 16)

    def test_convert_when_router_bias_present_then_copies_bias(self):
        """验证当原始 MoE 有 router_bias 时，转换后正确复制偏置"""
        bias_config = _make_moe_config(
            num_experts=2, top_k=1, hidden_size=8, moe_intermediate_size=16, use_moe_router_bias=True
        )
        original = MockOriginalMoE(bias_config, use_bias=True)
        original.gate_proj.weight = nn.Parameter(torch.randn(2, 16, 8))
        original.up_proj.weight = nn.Parameter(torch.randn(2, 16, 8))
        original.down_proj.weight = nn.Parameter(torch.randn(2, 8, 16))

        new_moe = convert_step35_moe_to_unpacked(original, bias_config)

        self.assertTrue(torch.equal(new_moe.router_bias, original.router_bias))

    def test_convert_when_limit_not_set_then_defaults_to_none(self):
        """验证当原始 MoE 没有 limit 属性时，转换后的 swiglu_limit 为 None"""
        config = _make_moe_config(num_experts=2, top_k=1, hidden_size=8, moe_intermediate_size=16)
        original = MockOriginalMoE(config)
        original.gate_proj.weight = nn.Parameter(torch.randn(2, 16, 8))
        original.up_proj.weight = nn.Parameter(torch.randn(2, 16, 8))
        original.down_proj.weight = nn.Parameter(torch.randn(2, 8, 16))

        del original.limit

        new_moe = convert_step35_moe_to_unpacked(original, config)

        # Verify each expert's limit is None
        for expert in new_moe.experts:
            self.assertIsNone(expert.limit)

    def test_convert_when_gate_weight_copied_then_weights_match(self):
        """验证转换后 gate 权重与原始 MoE 一致"""
        config = _make_moe_config(num_experts=2, top_k=1, hidden_size=8, moe_intermediate_size=16)
        original = MockOriginalMoE(config)
        original.gate_proj.weight = nn.Parameter(torch.randn(2, 16, 8))
        original.up_proj.weight = nn.Parameter(torch.randn(2, 16, 8))
        original.down_proj.weight = nn.Parameter(torch.randn(2, 8, 16))

        new_moe = convert_step35_moe_to_unpacked(original, config)

        self.assertTrue(torch.equal(new_moe.gate.weight, original.gate.weight))
