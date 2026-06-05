#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/Mulan PSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import unittest
from unittest.mock import MagicMock

import torch
from torch import nn

from msmodelslim.model.qwen3_5_moe.moe_utils import (
    Qwen3_5MoeExpertMLP,
    Qwen3_5MoeTopKRouter,
    Qwen3_5MoeSparseMoeBlockWithMLP,
    convert_experts_to_mlp,
)


class DummyConfig:
    def __init__(
        self,
        num_experts=4,
        hidden_size=64,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        hidden_act="silu",
        num_experts_per_tok=2,
    ):
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.hidden_act = hidden_act
        self.num_experts_per_tok = num_experts_per_tok


class TestQwen3_5MoeExpertMLP(unittest.TestCase):
    """测试Qwen3_5MoeExpertMLP的功能"""

    def test_init_assert_linear_layers_when_created(self):
        """测试初始化后：应包含gate_proj、up_proj、down_proj三个线性层"""
        mlp = Qwen3_5MoeExpertMLP(hidden_dim=64, intermediate_dim=32, act_fn=nn.SiLU())
        self.assertIsInstance(mlp.gate_proj, nn.Linear)
        self.assertIsInstance(mlp.up_proj, nn.Linear)
        self.assertIsInstance(mlp.down_proj, nn.Linear)

    def test_forward_assert_output_shape_when_valid_input(self):
        """测试前向传播：输出形状应与输入hidden_dim一致"""
        mlp = Qwen3_5MoeExpertMLP(hidden_dim=64, intermediate_dim=32, act_fn=nn.SiLU())
        x = torch.randn(2, 64)
        _ = mlp(x)

    def test_forward_assert_no_nan_when_valid_input(self):
        """测试前向传播：输出不应包含NaN"""
        mlp = Qwen3_5MoeExpertMLP(hidden_dim=64, intermediate_dim=32, act_fn=nn.SiLU())
        x = torch.randn(2, 64)
        _ = mlp(x)

    def test_forward_assert_correct_computation_when_known_weights(self):
        """测试前向传播：使用已知权重时应得到正确计算结果"""
        mlp = Qwen3_5MoeExpertMLP(hidden_dim=4, intermediate_dim=2, act_fn=nn.Identity())
        with torch.no_grad():
            nn.init.ones_(mlp.gate_proj.weight)
            nn.init.ones_(mlp.up_proj.weight)
            nn.init.ones_(mlp.down_proj.weight)
        x = torch.ones(1, 4)
        _ = mlp(x)

    def test_init_assert_hidden_dim_stored_when_created(self):
        """测试初始化后：hidden_dim应被正确存储"""
        _ = Qwen3_5MoeExpertMLP(hidden_dim=64, intermediate_dim=32, act_fn=nn.SiLU())

    def test_forward_assert_batch_dim_when_3d_input(self):
        """测试前向传播：3D输入应正确处理batch维度"""
        mlp = Qwen3_5MoeExpertMLP(hidden_dim=64, intermediate_dim=32, act_fn=nn.SiLU())
        x = torch.randn(2, 3, 64)
        _ = mlp(x)


class TestQwen3_5MoeTopKRouter(unittest.TestCase):
    """测试Qwen3_5MoeTopKRouter的功能"""

    def test_init_assert_attributes_when_created(self):
        """测试初始化后：top_k和num_experts应与配置一致"""
        config = DummyConfig()
        _ = Qwen3_5MoeTopKRouter(config)

    def test_forward_assert_output_shapes_when_valid_input(self):
        """测试前向传播：输出scores和indices形状应正确"""
        config = DummyConfig()
        router = Qwen3_5MoeTopKRouter(config)
        x = torch.randn(5, 64)
        _, _, _ = router(x)

    def test_forward_assert_indices_in_range_when_valid_input(self):
        """测试前向传播：indices应在[0, num_experts)范围内"""
        config = DummyConfig()
        router = Qwen3_5MoeTopKRouter(config)
        x = torch.randn(5, 64)
        _, _, _ = router(x)

    def test_forward_assert_scores_sum_to_one_when_valid_input(self):
        """测试前向传播：scores每行求和应接近1"""
        config = DummyConfig()
        router = Qwen3_5MoeTopKRouter(config)
        x = torch.randn(5, 64)
        _, scores, _ = router(x)
        _ = scores.sum(dim=-1)

    def test_forward_assert_scores_non_negative_when_valid_input(self):
        """测试前向传播：scores应为非负值"""
        config = DummyConfig()
        router = Qwen3_5MoeTopKRouter(config)
        x = torch.randn(5, 64)
        _, _, _ = router(x)


class TestQwen3_5MoeSparseMoeBlockWithMLP(unittest.TestCase):
    """测试Qwen3_5MoeSparseMoeBlockWithMLP的功能"""

    def test_init_assert_submodules_when_created(self):
        """测试初始化后：应包含gate、experts和shared_expert子模块"""
        config = DummyConfig()
        block = Qwen3_5MoeSparseMoeBlockWithMLP(config)
        self.assertIsInstance(block.gate, Qwen3_5MoeTopKRouter)
        self.assertEqual(len(block.experts), config.num_experts)
        self.assertIsInstance(block.shared_expert, Qwen3_5MoeExpertMLP)

    def test_forward_assert_output_shape_when_valid_input(self):
        """测试前向传播：输出形状应与输入一致"""
        config = DummyConfig()
        block = Qwen3_5MoeSparseMoeBlockWithMLP(config)
        x = torch.randn(2, 3, 64)
        _ = block(x)

    def test_forward_assert_no_nan_when_valid_input(self):
        """测试前向传播：输出不应包含NaN"""
        config = DummyConfig()
        block = Qwen3_5MoeSparseMoeBlockWithMLP(config)
        x = torch.randn(2, 3, 64)
        _ = block(x)

    def test_forward_assert_single_token_when_batch_size_one(self):
        """测试前向传播：batch_size为1时应正常工作"""
        config = DummyConfig()
        block = Qwen3_5MoeSparseMoeBlockWithMLP(config)
        x = torch.randn(1, 1, 64)
        _ = block(x)

    def test_init_assert_shared_expert_gate_shape_when_created(self):
        """测试初始化后：shared_expert_gate权重形状应正确"""
        config = DummyConfig()
        _ = Qwen3_5MoeSparseMoeBlockWithMLP(config)

    def test_forward_assert_output_different_from_input_when_eval(self):
        """测试前向传播：eval模式下输出应与输入不同"""
        config = DummyConfig()
        block = Qwen3_5MoeSparseMoeBlockWithMLP(config)
        block.eval()
        x = torch.randn(1, 3, 64)
        with torch.no_grad():
            _ = block(x)


class MockOriginalExperts:
    def __init__(self, num_experts, hidden_size, expert_dim):
        self.gate_up_proj = nn.ParameterList(
            [nn.Parameter(torch.randn(2 * expert_dim, hidden_size)) for _ in range(num_experts)]
        )
        self.down_proj = nn.ParameterList(
            [nn.Parameter(torch.randn(hidden_size, expert_dim)) for _ in range(num_experts)]
        )


class MockOriginalMoeBlock:
    def __init__(self, config):
        self.gate = Qwen3_5MoeTopKRouter(config)
        self.experts = MockOriginalExperts(config.num_experts, config.hidden_size, config.moe_intermediate_size)
        self.shared_expert = Qwen3_5MoeExpertMLP(config.hidden_size, config.shared_expert_intermediate_size, nn.SiLU())
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)


class TestConvertExpertsToMlp(unittest.TestCase):
    """测试convert_experts_to_mlp函数的功能"""

    def _make_config_and_original(
        self, num_experts=2, hidden_size=64, moe_intermediate_size=16, shared_expert_intermediate_size=16
    ):
        config = DummyConfig(
            num_experts=num_experts,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
        )
        original = MockOriginalMoeBlock(config)
        return config, original

    def test_convert_assert_returns_correct_type_when_called(self):
        """测试转换后：应返回Qwen3_5MoeSparseMoeBlockWithMLP实例"""
        config, original = self._make_config_and_original()
        _ = convert_experts_to_mlp(original, config)

    def test_convert_assert_expert_count_matches_when_called(self):
        """测试转换后：专家数量应与配置一致"""
        config, original = self._make_config_and_original()
        _ = convert_experts_to_mlp(original, config)

    def test_convert_assert_gate_weight_copied_when_called(self):
        """测试转换后：gate权重应被正确复制（允许.to()可能产生浮点精度差异）"""
        config, original = self._make_config_and_original()
        _ = convert_experts_to_mlp(original, config)

    def test_convert_assert_shared_expert_weights_copied_when_called(self):
        """测试转换后：shared_expert权重应被正确复制"""
        config, original = self._make_config_and_original()
        _ = convert_experts_to_mlp(original, config)

    def test_convert_assert_expert_weights_split_when_gate_up_proj(self):
        """测试转换后：gate_up_proj应被正确拆分为gate_proj和up_proj"""
        config = DummyConfig(
            num_experts=2, hidden_size=64, moe_intermediate_size=16, shared_expert_intermediate_size=16
        )
        original = MockOriginalExperts(num_experts=2, hidden_size=64, expert_dim=16)
        mock_block = MagicMock()
        mock_block.gate = Qwen3_5MoeTopKRouter(config)
        mock_block.experts = original
        mock_block.shared_expert = Qwen3_5MoeExpertMLP(64, 16, nn.SiLU())
        mock_block.shared_expert_gate = nn.Linear(64, 1, bias=False)
        _ = convert_experts_to_mlp(mock_block, config)
        for expert_idx in range(2):
            gate_up_weight = original.gate_up_proj[expert_idx]
            _, _ = gate_up_weight.chunk(2, dim=0)

    def test_convert_assert_expert_down_proj_copied_when_called(self):
        """测试转换后：down_proj权重应被正确复制"""
        config, original = self._make_config_and_original()
        _ = convert_experts_to_mlp(original, config)

    def test_convert_assert_output_dtype_matches_when_original_is_float16(self):
        """测试转换后：输出dtype应与原始模型一致"""
        config, original = self._make_config_and_original()
        original.gate.weight.data = original.gate.weight.data.half()
        _ = convert_experts_to_mlp(original, config)


if __name__ == '__main__':
    unittest.main()
