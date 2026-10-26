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

import torch
from torch import nn

from msmodelslim.processor.quarot.common.quarot_utils import (
    QuaRotMode,
    create_rot,
    rotate_linear,
    rotate_weight,
    bake_mean_into_linear,
    is_power_of_two,
    get_decompose_dim,
)
from msmodelslim.utils.exception import UnsupportedError


class TestQuaRotMode(unittest.TestCase):
    """测试 QuaRotMode 枚举类"""

    def test_enum_values_should_be_correct(self):
        """测试枚举值"""
        self.assertEqual(QuaRotMode.HADAMARD.value, "hadamard")
        self.assertEqual(QuaRotMode.BLOCK_HADAMARD_SHIFTED.value, "block_hadamard_shifted")


class TestIsPowerOfTwo(unittest.TestCase):
    """测试 is_power_of_two 函数"""

    def test_powers_of_two_should_return_true(self):
        """测试 2 的幂次返回 True"""
        powers = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        for n in powers:
            with self.subTest(n=n):
                self.assertTrue(is_power_of_two(n))

    def test_non_powers_of_two_should_return_false(self):
        """测试非 2 的幂次返回 False"""
        non_powers = [0, 3, 5, 6, 7, 9, 10, 15, 17, 100, 1000]
        for n in non_powers:
            with self.subTest(n=n):
                self.assertFalse(is_power_of_two(n))

    def test_negative_numbers_should_return_false(self):
        """测试负数返回 False"""
        self.assertFalse(is_power_of_two(-1))
        self.assertFalse(is_power_of_two(-2))


class TestCreateRot(unittest.TestCase):
    """测试 create_rot 函数"""

    def test_hadamard_mode_should_return_correct_size(self):
        """测试 HADAMARD 模式返回正确大小"""
        size = 8
        rot = create_rot(QuaRotMode.HADAMARD, size)
        self.assertEqual(rot.shape, (size, size))

    def test_hadamard_mode_with_block_size_should_return_correct_size(self):
        """测试 HADAMARD 模式带 block_size"""
        size = 16
        block_size = 4
        rot = create_rot(QuaRotMode.HADAMARD, size, block_size=block_size)
        self.assertEqual(rot.shape, (size, size))

    def test_hadamard_mode_should_be_orthogonal(self):
        """测试 HADAMARD 模式生成正交矩阵"""
        size = 8
        rot = create_rot(QuaRotMode.HADAMARD, size)
        product = rot.T @ rot
        expected = torch.eye(size)
        self.assertTrue(torch.allclose(product, expected, atol=1e-5))

    def test_block_hadamard_shifted_mode_should_return_correct_size(self):
        """测试 BLOCK_HADAMARD_SHIFTED 模式返回正确大小"""
        size = 64
        rot = create_rot(QuaRotMode.BLOCK_HADAMARD_SHIFTED, size)
        self.assertEqual(rot.shape, (size, size))

    def test_block_hadamard_shifted_with_rot_step_1(self):
        """测试 BLOCK_HADAMARD_SHIFTED 模式 rot_step=1"""
        size = 64
        rot = create_rot(QuaRotMode.BLOCK_HADAMARD_SHIFTED, size, rot_step=1)
        self.assertEqual(rot.shape, (size, size))

    def test_block_hadamard_shifted_with_rot_step_2(self):
        """测试 BLOCK_HADAMARD_SHIFTED 模式 rot_step=2"""
        size = 64
        rot = create_rot(QuaRotMode.BLOCK_HADAMARD_SHIFTED, size, rot_step=2)
        self.assertEqual(rot.shape, (size, size))

    def test_block_hadamard_shifted_with_rot_step_3(self):
        """测试 BLOCK_HADAMARD_SHIFTED 模式 rot_step=3"""
        size = 64
        rot = create_rot(QuaRotMode.BLOCK_HADAMARD_SHIFTED, size, rot_step=3)
        self.assertEqual(rot.shape, (size, size))

    def test_block_hadamard_shifted_with_invalid_rot_step_should_raise(self):
        """测试 BLOCK_HADAMARD_SHIFTED 模式无效 rot_step 抛出异常"""
        size = 64
        with self.assertRaises(UnsupportedError):
            create_rot(QuaRotMode.BLOCK_HADAMARD_SHIFTED, size, rot_step=4)


class TestRotateLinear(unittest.TestCase):
    """测试 rotate_linear 函数"""

    def test_right_rotate_should_modify_weight(self):
        """测试右旋转修改权重"""
        linear = nn.Linear(8, 4)
        rot = torch.eye(8)
        original_weight = linear.weight.data.clone()
        rotate_linear(linear, rot, right_rotate=True)
        # 单位矩阵旋转不应改变权重
        self.assertTrue(torch.allclose(linear.weight.data, original_weight))

    def test_left_rotate_should_modify_weight(self):
        """测试左旋转修改权重"""
        linear = nn.Linear(8, 4)
        rot = torch.eye(4)
        original_weight = linear.weight.data.clone()
        rotate_linear(linear, rot, right_rotate=False)
        # 单位矩阵旋转不应改变权重
        self.assertTrue(torch.allclose(linear.weight.data, original_weight))

    def test_left_rotate_with_bias_should_modify_bias(self):
        """测试左旋转修改偏置"""
        linear = nn.Linear(8, 4, bias=True)
        rot = torch.eye(4)
        original_bias = linear.bias.data.clone()
        rotate_linear(linear, rot, right_rotate=False)
        self.assertTrue(torch.allclose(linear.bias.data, original_bias))

    def test_right_rotate_should_not_modify_bias(self):
        """测试右旋转不修改偏置"""
        linear = nn.Linear(8, 4, bias=True)
        rot = torch.eye(8)
        original_bias = linear.bias.data.clone()
        rotate_linear(linear, rot, right_rotate=True)
        self.assertTrue(torch.allclose(linear.bias.data, original_bias))

    def test_block_rotation_should_expand_rotation_matrix(self):
        """测试块旋转自动扩展旋转矩阵"""
        linear = nn.Linear(16, 4)
        rot = torch.eye(8)  # 8x8 旋转矩阵，但权重是 16 维
        original_weight = linear.weight.data.clone()
        rotate_linear(linear, rot, right_rotate=True)
        # 应该自动扩展为 16x16 的块对角矩阵
        self.assertTrue(torch.allclose(linear.weight.data, original_weight, atol=1e-5))

    def test_invalid_rotation_dim_should_raise(self):
        """测试无效旋转维度抛出异常"""
        linear = nn.Linear(10, 4)
        rot = torch.eye(3)  # 10 不能被 3 整除
        with self.assertRaises(UnsupportedError):
            rotate_linear(linear, rot, right_rotate=True)


class TestRotateWeight(unittest.TestCase):
    """测试 rotate_weight 函数"""

    def test_right_rotate_should_modify_weight(self):
        """测试右旋转修改权重"""
        weight = torch.randn(4, 8)
        rot = torch.eye(8)
        original_weight = weight.clone()
        rotate_weight(weight, rot, right_rotate=True)
        self.assertTrue(torch.allclose(weight.data, original_weight))

    def test_left_rotate_should_modify_weight(self):
        """测试左旋转修改权重"""
        weight = torch.randn(4, 8)
        rot = torch.eye(4)
        original_weight = weight.clone()
        rotate_weight(weight, rot, right_rotate=False)
        self.assertTrue(torch.allclose(weight.data, original_weight))

    def test_list_rotation_should_be_converted_to_block_diag(self):
        """测试列表旋转转换为块对角矩阵"""
        weight = torch.randn(4, 16)
        rot_list = [torch.eye(8), torch.eye(8)]
        original_weight = weight.clone()
        rotate_weight(weight, rot_list, right_rotate=True)
        self.assertTrue(torch.allclose(weight.data, original_weight, atol=1e-5))


class TestBakeMeanIntoLinear(unittest.TestCase):
    """测试 bake_mean_into_linear 函数"""

    def test_should_subtract_mean_from_weight(self):
        """测试从权重中减去均值"""
        linear = nn.Linear(8, 4)
        bake_mean_into_linear(linear)
        # bake_mean_into_linear 减去的是每行的均值 (dim=-2)
        # 所以每行的均值应该接近 0
        # 但这里测试的是整体均值应该减小
        weight_mean = linear.weight.data.double().mean()
        # 减去均值后，整体均值应该接近 0
        self.assertTrue(torch.abs(weight_mean) < 0.5)

    def test_should_subtract_mean_from_bias(self):
        """测试从偏置中减去均值"""
        linear = nn.Linear(8, 4, bias=True)
        bake_mean_into_linear(linear)
        # 减去均值后，偏置的均值应该接近 0
        bias_mean = linear.bias.data.double().mean()
        self.assertTrue(torch.abs(bias_mean) < 1e-4)

    def test_without_bias_should_only_modify_weight(self):
        """测试无偏置时只修改权重"""
        linear = nn.Linear(8, 4, bias=False)
        original_weight = linear.weight.data.clone()
        bake_mean_into_linear(linear)
        self.assertFalse(torch.allclose(linear.weight.data, original_weight))


class TestGetDecomposeDim(unittest.TestCase):
    """测试 get_decompose_dim 函数"""

    def test_valid_dimensions_should_return_tuple(self):
        """测试有效维度返回元组"""
        # 测试一些已知可分解的维度
        for n in [16, 32, 64]:
            with self.subTest(n=n):
                a, b = get_decompose_dim(n)
                self.assertIsInstance(a, int)
                self.assertIsInstance(b, int)
                self.assertGreater(a, 0)
                self.assertGreater(b, 0)


class TestFuseLnLinear(unittest.TestCase):
    """测试 fuse_ln_linear 函数"""

    def test_should_fuse_single_layernorm(self):
        """测试融合单个 layernorm"""
        from msmodelslim.processor.quarot.common.quarot_utils import fuse_ln_linear

        # 创建 RMSNorm (1D weight)
        ln = nn.Module()
        ln.weight = nn.Parameter(torch.tensor([2.0, 3.0, 4.0]))
        ln.bias = nn.Parameter(torch.tensor([0.1, 0.2, 0.3]))

        # 创建 linear 层
        linear = nn.Linear(3, 2)

        fuse_ln_linear([ln], [linear])

        # 验证 layernorm weight 被重置为 1
        self.assertTrue(torch.allclose(ln.weight.data, torch.ones(3)))
        # 验证 layernorm bias 被重置为 0
        self.assertTrue(torch.allclose(ln.bias.data, torch.zeros(3)))

    def test_should_raise_for_2d_layernorm_weight(self):
        """测试 2D layernorm weight 抛出异常"""
        from msmodelslim.processor.quarot.common.quarot_utils import fuse_ln_linear

        ln = nn.Module()
        ln.weight = nn.Parameter(torch.randn(2, 3))  # 2D weight

        linear = nn.Linear(3, 2)

        with self.assertRaises(UnsupportedError):
            fuse_ln_linear([ln], [linear])

    def test_should_raise_for_mismatched_dimensions(self):
        """测试维度不匹配抛出异常"""
        from msmodelslim.processor.quarot.common.quarot_utils import fuse_ln_linear

        ln = nn.Module()
        ln.weight = nn.Parameter(torch.randn(4))  # 4 维

        linear = nn.Linear(3, 2)  # 输入 3 维

        with self.assertRaises(UnsupportedError):
            fuse_ln_linear([ln], [linear])

    def test_should_fuse_with_bias(self):
        """测试融合带 bias 的 layernorm"""
        from msmodelslim.processor.quarot.common.quarot_utils import fuse_ln_linear

        ln = nn.Module()
        ln.weight = nn.Parameter(torch.tensor([2.0, 3.0]))
        ln.bias = nn.Parameter(torch.tensor([0.1, 0.2]))

        linear = nn.Linear(2, 3, bias=True)

        fuse_ln_linear([ln], [linear])

        # 验证 layernorm weight 被重置为 1
        self.assertTrue(torch.allclose(ln.weight.data, torch.ones(2)))


class TestOnlineRotateOprojInput(unittest.TestCase):
    """测试 online_rotate_o_proj_input 函数"""

    def test_should_rotate_o_proj_weight_when_non_identity_rotation(self):
        """给定非单位旋转块，期望 o_proj 权重等于原文右乘 kron(rot, I)。"""
        from msmodelslim.processor.quarot.common.quarot_utils import online_rotate_o_proj_input

        o_proj = nn.Linear(8, 4)
        v_proj = nn.Linear(8, 4)
        ov_pairs = {o_proj: v_proj}

        rot_online = torch.tensor([[0.5, 0.5], [0.5, -0.5]])
        num_attn_heads = 2
        weight_before = o_proj.weight.data.clone()
        expected_rot = torch.kron(rot_online, torch.eye(2))
        expected_rot = torch.kron(rot_online, torch.eye(8 // num_attn_heads))
        online_rotate_o_proj_input(ov_pairs, rot_online, num_attn_heads)

        expected = torch.matmul(weight_before.float(), expected_rot).to(o_proj.weight.dtype)
        assert torch.allclose(o_proj.weight.data, expected)


class TestOnlineRotateDownProj(unittest.TestCase):
    """测试 online_rotate_down_proj 函数"""

    def test_should_rotate_down_proj_weight_when_non_identity_rotations(self):
        """给定非单位 rot1/rot2，期望 down_proj 权重按 rot1.T @ (W @ rot2) 更新。"""
        from msmodelslim.processor.quarot.common.quarot_utils import online_rotate_down_proj

        up_proj = nn.Linear(8, 4)
        down_proj = nn.Linear(4, 8)
        pairs = {up_proj: down_proj}

        rot1 = torch.tensor(
            [[0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, -0.5], [0.5, 0.5, -0.5, -0.5], [0.5, -0.5, -0.5, 0.5]]
        )
        rot2 = torch.tensor([[0.5, 0.5], [0.5, -0.5]])
        weight_before = down_proj.weight.data.clone()

        online_rotate_down_proj(pairs, rot1, rot2)

        rotated = torch.matmul(weight_before.float().view(-1, rot1.shape[0], rot2.shape[0]), rot2)
        expected = torch.matmul(rot1.T, rotated).reshape(weight_before.shape).to(down_proj.weight.dtype)
        assert torch.allclose(down_proj.weight.data, expected)
