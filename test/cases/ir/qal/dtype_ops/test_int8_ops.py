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

from msmodelslim.ir.qal.dtype_ops.int8_ops import Int8


def _eq(actual, expected, msg):
    if not torch.equal(actual, expected):
        raise AssertionError(f"{msg}: got {actual} expected {expected}")


class TestInt8(unittest.TestCase):
    """被测目标：Int8。对称/非对称与三种粒度的公开方法。"""

    def test_calc_qparam_per_tensor_sym_equals_amax_over_127_when_span_is_20(self):
        """场景：对称 per-tensor 参数。预期：scale = 20/127。"""
        scale = Int8.calc_qparam_per_tensor_sym(torch.tensor([-10.0, 3.0, 20.0]))
        self.assertTrue(torch.equal(scale, torch.tensor(20.0 / 127.0)))

    def test_calc_qparam_per_channel_sym_keeps_independent_scale_when_two_rows(self):
        """场景：对称 per-channel，两行不同幅值。预期：两行 scale 分别为 10/127 与 1/127。"""
        x = torch.tensor([[10.0, -10.0], [1.0, -1.0]])
        scale = Int8.calc_qparam_per_channel_sym(x, axis=0)
        expected = torch.tensor([[10.0 / 127.0], [1.0 / 127.0]])
        self.assertTrue(torch.allclose(scale, expected))

    def test_calc_qparam_per_group_sym_returns_two_groups_when_length_8_group_4(self):
        """场景：对称 per-group，长度 8、组 4。预期：scale 形状 (2, 1)。"""
        scale = Int8.calc_qparam_per_group_sym(torch.arange(8, dtype=torch.float32), axis=-1, group_size=4)
        self.assertEqual(tuple(scale.shape), (2, 1))

    def test_calc_qparam_per_group_sym_raises_value_error_when_length_not_divisible(self):
        """场景：对称 per-group，长度 5 不能被 4 整除。预期：ValueError。"""
        with self.assertRaises(ValueError) as ctx:
            Int8.calc_qparam_per_group_sym(torch.arange(5, dtype=torch.float32), axis=-1, group_size=4)
        self.assertIn("not divisible", str(ctx.exception))

    def test_calc_qparam_per_tensor_sym_returns_nan_when_input_has_nan(self):
        """场景：输入含 nan。预期：scale 为 nan。"""
        scale = Int8.calc_qparam_per_tensor_sym(torch.tensor([1.0, float("nan"), 2.0]))
        self.assertTrue(torch.isnan(scale))

    def test_calc_qparam_per_channel_sym_keeps_finite_channel_when_other_is_inf(self):
        """场景：第二列含 inf。预期：第一列 scale 有限，第二列 scale 为 inf。"""
        x = torch.tensor([[1.0, -2.0], [float("inf"), 3.0]])
        scale = Int8.calc_qparam_per_channel_sym(x, axis=0)
        self.assertTrue(torch.isfinite(scale[0]).all())
        self.assertTrue(torch.isinf(scale[1]).all())

    def test_quantize_per_tensor_sym_returns_input_when_values_in_range(self):
        """场景：scale=1，输入在 (-128, 127)。预期：码值等于输入。"""
        x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 4.0, 50.0])
        q, _ = Int8.quantize_per_tensor_sym(x, torch.tensor(1.0))
        _eq(q, x, "int8正常范围异常")

    def test_quantize_per_tensor_sym_returns_float32_when_dtype_omitted(self):
        """场景：省略 dtype。预期：q 为 float32。"""
        q, _ = Int8.quantize_per_tensor_sym(torch.tensor([1.0]), torch.tensor(1.0))
        self.assertEqual(q.dtype, torch.float32)

    def test_quantize_per_tensor_sym_returns_neg128_when_input_is_qmin(self):
        """场景：输入恰好 -128。预期：码值 -128。"""
        q, _ = Int8.quantize_per_tensor_sym(torch.tensor([-128.0]), torch.tensor(1.0))
        _eq(q, torch.tensor([-128.0]), "int8下限异常")

    def test_quantize_per_tensor_sym_returns_127_when_input_is_qmax(self):
        """场景：输入恰好 127。预期：码值 127。"""
        q, _ = Int8.quantize_per_tensor_sym(torch.tensor([127.0]), torch.tensor(1.0))
        _eq(q, torch.tensor([127.0]), "int8上限异常")

    def test_quantize_per_tensor_sym_keeps_neg127_when_input_in_range(self):
        """场景：输入 -127。预期：仍为 -127，不是下限。"""
        q, _ = Int8.quantize_per_tensor_sym(torch.tensor([-127.0]), torch.tensor(1.0))
        _eq(q, torch.tensor([-127.0]), "int8的-127档位异常")

    def test_quantize_per_tensor_sym_clamps_to_127_when_input_is_128(self):
        """场景：输入 128 上溢。预期：饱和到 127。"""
        q, _ = Int8.quantize_per_tensor_sym(torch.tensor([128.0]), torch.tensor(1.0))
        _eq(q, torch.tensor([127.0]), "int8溢出128未饱和到127")

    def test_quantize_per_tensor_sym_clamps_to_neg128_when_input_below_qmin(self):
        """场景：输入 -129 与 -200。预期：都饱和到 -128。"""
        q, _ = Int8.quantize_per_tensor_sym(torch.tensor([-129.0, -200.0]), torch.tensor(1.0))
        _eq(q, torch.tensor([-128.0, -128.0]), "int8下溢未饱和到-128")

    def test_quantize_per_tensor_sym_rounds_to_even_when_half_integer(self):
        """场景：半整数。预期：roundTiesToEven 得到 [-2,0,0,2,2]。"""
        x = torch.tensor([-1.5, -0.5, 0.5, 1.5, 2.5])
        q, _ = Int8.quantize_per_tensor_sym(x, torch.tensor(1.0))
        _eq(q, torch.tensor([-2.0, 0.0, 0.0, 2.0, 2.0]), "int8半整数舍入到偶数异常")

    def test_quantize_per_tensor_sym_returns_int8_grid_when_dtype_is_int8(self):
        """场景：dtype=int8。预期：码值与 float32 网格相同，dtype 为 int8。"""
        x = torch.tensor([0.2, 1.7, -0.4])
        q32, _ = Int8.quantize_per_tensor_sym(x, torch.tensor(1.0))
        q8, _ = Int8.quantize_per_tensor_sym(x, torch.tensor(1.0), dtype=torch.int8)
        self.assertEqual(q8.dtype, torch.int8)
        _eq(q8.to(torch.float32), q32, "int8低精度与float32网格不一致")

    def test_quantize_per_tensor_sym_keeps_input_shape_when_dtype_is_int8(self):
        """场景：dtype=int8。预期：形状与输入相同，不打包。"""
        x = torch.zeros(6)
        q, _ = Int8.quantize_per_tensor_sym(x, torch.tensor(1.0), dtype=torch.int8)
        self.assertEqual(tuple(q.shape), tuple(x.shape))

    def test_quantize_per_tensor_sym_returns_same_scale_object_when_static(self):
        """场景：静态量化。预期：返回的 scale 是传入对象。"""
        scale = torch.tensor(1.0, requires_grad=True)
        _, returned = Int8.quantize_per_tensor_sym(torch.tensor([1.0, 2.0]), scale)
        self.assertIs(returned, scale)

    def test_quantize_per_channel_sym_restores_values_when_channel_scales_differ(self):
        """场景：两行不同 scale。预期：反量化误差不超过 0.5×scale。"""
        x = torch.tensor([[10.0, -10.0], [1.0, -1.0]])
        scale = Int8.calc_qparam_per_channel_sym(x, axis=0)
        q, scale = Int8.quantize_per_channel_sym(x, scale, axis=0)
        y = Int8.dequantize_per_channel_sym(q, scale, axis=0)
        self.assertTrue(torch.all((y - x).abs() <= 0.5 * scale + 1e-5))

    def test_quantize_per_group_sym_restores_length_when_group_size_4(self):
        """场景：长度 8、组 4 对称量化。预期：输出长度仍为 8。"""
        x = torch.arange(8, dtype=torch.float32)
        scale = Int8.calc_qparam_per_group_sym(x, axis=-1, group_size=4)
        q, scale = Int8.quantize_per_group_sym(x, scale, axis=-1, group_size=4)
        self.assertEqual(tuple(q.shape), (8,))

    def test_dequantize_per_tensor_sym_equals_q_times_scale_when_scale_is_two(self):
        """场景：码值 [0,1,-2]，scale=2。预期：反量化 [0,2,-4]。"""
        y = Int8.dequantize_per_tensor_sym(torch.tensor([0.0, 1.0, -2.0]), torch.tensor(2.0))
        _eq(y, torch.tensor([0.0, 2.0, -4.0]), "int8反量化q*scale异常")

    def test_dequantize_per_channel_sym_equals_broadcast_product_when_row_scales(self):
        """场景：两行码值乘各自 scale。预期：按行广播相乘。"""
        q = torch.tensor([[1.0, -1.0], [2.0, 0.0]])
        scale = torch.tensor([[3.0], [5.0]])
        y = Int8.dequantize_per_channel_sym(q, scale, axis=0)
        _eq(y, torch.tensor([[3.0, -3.0], [10.0, 0.0]]), "int8 per-channel反量化异常")

    def test_dequantize_per_group_sym_equals_grouped_product_when_two_groups(self):
        """场景：两组 scale 分别为 1 和 2。预期：后一组码值翻倍。"""
        q = torch.tensor([1.0, 2.0, 3.0, 4.0])
        scale = torch.tensor([[1.0], [2.0]])
        y = Int8.dequantize_per_group_sym(q, scale, axis=-1, group_size=2)
        _eq(y, torch.tensor([1.0, 2.0, 6.0, 8.0]), "int8 per-group反量化异常")

    def test_quantize_dynamic_per_tensor_sym_matches_static_when_same_detached_scale(self):
        """场景：动态量化再与其 scale 做静态量化。预期：码值相同。"""
        x = torch.tensor([1.0, -2.0, 3.0])
        q_d, scale_d = Int8.quantize_dynamic_per_tensor_sym(x)
        q, _ = Int8.quantize_per_tensor_sym(x, scale_d)
        _eq(q, q_d, "int8动态与静态不一致")

    def test_quantize_dynamic_per_tensor_sym_detaches_scale_when_x_requires_grad(self):
        """场景：x 需要梯度。预期：动态 scale 已 detach。"""
        x = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
        _, scale_d = Int8.quantize_dynamic_per_tensor_sym(x)
        self.assertFalse(scale_d.requires_grad)

    def test_quantize_dynamic_per_channel_sym_returns_channel_scale_when_two_rows(self):
        """场景：动态 per-channel。预期：scale 形状 (2, 1)。"""
        x = torch.tensor([[10.0, -10.0], [1.0, -1.0]])
        _, scale = Int8.quantize_dynamic_per_channel_sym(x, axis=0)
        self.assertEqual(tuple(scale.shape), (2, 1))

    def test_quantize_dynamic_per_group_sym_returns_group_scale_when_group_size_4(self):
        """场景：动态 per-group。预期：scale 形状 (2, 1)。"""
        x = torch.arange(8, dtype=torch.float32)
        _, scale = Int8.quantize_dynamic_per_group_sym(x, axis=-1, group_size=4)
        self.assertEqual(tuple(scale.shape), (2, 1))

    def test_quantize_per_tensor_sym_x_grad_equals_one_when_unsaturated(self):
        """场景：默认 ste_clamp，未饱和。预期：x.grad 全为 1。"""
        x = torch.tensor([0.2, 1.7], requires_grad=True)
        scale = torch.tensor(1.0, requires_grad=True)
        q, scale = Int8.quantize_per_tensor_sym(x, scale)
        Int8.dequantize_per_tensor_sym(q, scale).sum().backward()
        _eq(x.grad, torch.ones_like(x), "int8未饱和处输入梯度应为1")

    def test_quantize_per_tensor_sym_scale_grad_equals_sum_residual_when_unsaturated(self):
        """场景：输入 [0.2,1.7]、scale=1。预期：q=[0,2]，scale.grad=0.1。"""
        x = torch.tensor([0.2, 1.7], requires_grad=True)
        scale = torch.tensor(1.0, requires_grad=True)
        q, scale = Int8.quantize_per_tensor_sym(x, scale)
        Int8.dequantize_per_tensor_sym(q, scale).sum().backward()
        _eq(q, torch.tensor([0.0, 2.0]), "int8反传前的码值异常")
        self.assertAlmostEqual(scale.grad.item(), 0.1, places=5)

    def test_quantize_per_group_sym_scale_grad_stays_grouped_when_two_groups(self):
        """场景：两组 scale 都是 1。预期：梯度形状 (2,1)，值为 [[0.1],[0.2]]。"""
        x = torch.tensor([0.2, 1.7, -0.4, 2.2], requires_grad=True)
        scale = torch.ones(2, 1, requires_grad=True)
        q, scale = Int8.quantize_per_group_sym(x, scale, axis=-1, group_size=2)
        Int8.dequantize_per_group_sym(q, scale, axis=-1, group_size=2).sum().backward()
        self.assertTrue(torch.allclose(scale.grad, torch.tensor([[0.1], [0.2]])))

    def test_quantize_per_tensor_sym_cuts_input_grad_when_dtype_is_int8(self):
        """场景：dtype=int8 后反量化。预期：输入无梯度。"""
        x = torch.tensor([0.2, 1.7], requires_grad=True)
        scale = torch.tensor(1.0, requires_grad=True)
        q, scale = Int8.quantize_per_tensor_sym(x, scale, dtype=torch.int8)
        Int8.dequantize_per_tensor_sym(q, scale).sum().backward()
        self.assertIsNone(x.grad)

    def test_dequantize_per_tensor_sym_roundtrips_zero_when_input_all_zero(self):
        """场景：全 0。预期：反量化仍为 0。"""
        x = torch.zeros(4)
        scale = Int8.calc_qparam_per_tensor_sym(x)
        q, scale = Int8.quantize_per_tensor_sym(x, scale)
        y = Int8.dequantize_per_tensor_sym(q, scale)
        _eq(y, x, "int8全零往返异常")

    def test_dequantize_per_channel_sym_keeps_finite_channel_when_other_is_inf(self):
        """场景：第二列 inf。预期：第一列反量化有限，第二列含 nan。"""
        x = torch.tensor([[1.0, -2.0], [float("inf"), 3.0]])
        scale = Int8.calc_qparam_per_channel_sym(x, axis=0)
        q, scale = Int8.quantize_per_channel_sym(x, scale, axis=0)
        y = Int8.dequantize_per_channel_sym(q, scale, axis=0)
        self.assertTrue(torch.isfinite(y[0]).all())
        self.assertTrue(torch.isnan(y[1]).any())

    def test_calc_qparam_per_tensor_asym_equals_one_when_span_is_255(self):
        """场景：非对称 [0, 255]。预期：scale=1。"""
        scale, _offset = Int8.calc_qparam_per_tensor_asym(torch.tensor([0.0, 255.0]))
        self.assertTrue(torch.allclose(scale, torch.tensor(1.0)))

    def test_quantize_per_tensor_asym_stays_inside_int8_when_span_is_255(self):
        """场景：非对称 [0, 255]。预期：码值落在 [-128, 127]。"""
        x = torch.tensor([0.0, 255.0])
        scale, offset = Int8.calc_qparam_per_tensor_asym(x)
        q, _, _ = Int8.quantize_per_tensor_asym(x, scale, offset)
        self.assertGreaterEqual(q.min().item(), -128)
        self.assertLessEqual(q.max().item(), 127)

    def test_quantize_per_tensor_asym_x_grad_is_zero_at_sat_when_ste_clamp_false(self):
        """场景：硬截断，输入 [-200,0,200]。预期：x.grad=[0,1,0]。"""
        x = torch.tensor([-200.0, 0.0, 200.0], requires_grad=True)
        scale = torch.tensor(1.0, requires_grad=True)
        offset = torch.tensor(0.0, requires_grad=True)
        q, scale, offset = Int8.quantize_per_tensor_asym(x, scale, offset, ste_clamp=False)
        Int8.dequantize_per_tensor_asym(q, scale, offset).sum().backward()
        _eq(x.grad, torch.tensor([0.0, 1.0, 0.0]), "int8非对称饱和输入梯度异常")

    def test_quantize_per_tensor_asym_scale_grad_equals_neg1_when_ste_clamp_false(self):
        """场景：硬截断，码值 [-128,0,127]。预期：scale.grad=-1。"""
        x = torch.tensor([-200.0, 0.0, 200.0], requires_grad=True)
        scale = torch.tensor(1.0, requires_grad=True)
        offset = torch.tensor(0.0, requires_grad=True)
        q, scale, offset = Int8.quantize_per_tensor_asym(x, scale, offset, ste_clamp=False)
        Int8.dequantize_per_tensor_asym(q, scale, offset).sum().backward()
        _eq(q, torch.tensor([-128.0, 0.0, 127.0]), "int8非对称硬截断码值异常")
        self.assertAlmostEqual(scale.grad.item(), -1.0, places=5)

    def test_quantize_per_tensor_asym_offset_grad_equals_neg2_when_ste_clamp_false(self):
        """场景：硬截断，两端饱和。预期：offset.grad=-2。"""
        x = torch.tensor([-200.0, 0.0, 200.0], requires_grad=True)
        scale = torch.tensor(1.0, requires_grad=True)
        offset = torch.tensor(0.0, requires_grad=True)
        q, scale, offset = Int8.quantize_per_tensor_asym(x, scale, offset, ste_clamp=False)
        Int8.dequantize_per_tensor_asym(q, scale, offset).sum().backward()
        self.assertAlmostEqual(offset.grad.item(), -2.0, places=5)

    def test_calc_qparam_per_channel_asym_returns_keepdim_scale_when_axis_0(self):
        """场景：非对称 per-channel。预期：scale 形状 (2, 1)。"""
        x = torch.tensor([[0.0, 10.0], [1.0, 2.0]])
        scale, offset = Int8.calc_qparam_per_channel_asym(x, axis=0)
        self.assertEqual(tuple(scale.shape), (2, 1))
        self.assertEqual(tuple(offset.shape), (2, 1))

    def test_calc_qparam_per_group_asym_returns_two_groups_when_length_8_group_4(self):
        """场景：非对称 per-group。预期：scale 形状 (2, 1)。"""
        scale, offset = Int8.calc_qparam_per_group_asym(torch.arange(8, dtype=torch.float32), axis=-1, group_size=4)
        self.assertEqual(tuple(scale.shape), (2, 1))
        self.assertEqual(tuple(offset.shape), (2, 1))

    def test_quantize_per_channel_asym_returns_float32_when_dtype_omitted(self):
        """场景：非对称 per-channel 省略 dtype。预期：q 为 float32。"""
        x = torch.tensor([[0.0, 10.0], [1.0, 2.0]])
        scale, offset = Int8.calc_qparam_per_channel_asym(x, axis=0)
        q, _, _ = Int8.quantize_per_channel_asym(x, scale, offset, axis=0)
        self.assertEqual(q.dtype, torch.float32)

    def test_quantize_per_group_asym_restores_length_when_group_size_4(self):
        """场景：非对称 per-group。预期：q 长度 8。"""
        x = torch.arange(8, dtype=torch.float32)
        scale, offset = Int8.calc_qparam_per_group_asym(x, axis=-1, group_size=4)
        q, _, _ = Int8.quantize_per_group_asym(x, scale, offset, axis=-1, group_size=4)
        self.assertEqual(tuple(q.shape), (8,))

    def test_dequantize_per_tensor_asym_equals_q_minus_offset_times_scale_when_offset_zero(self):
        """场景：offset=0、scale=2。预期：与对称反量化相同。"""
        y = Int8.dequantize_per_tensor_asym(torch.tensor([1.0, -2.0]), torch.tensor(2.0), torch.tensor(0.0))
        _eq(y, torch.tensor([2.0, -4.0]), "int8非对称反量化异常")

    def test_dequantize_per_channel_asym_uses_offset_when_row_params_differ(self):
        """场景：两行不同 offset。预期：(q-offset)*scale。"""
        q = torch.tensor([[0.0], [2.0]])
        scale = torch.tensor([[1.0], [2.0]])
        offset = torch.tensor([[0.0], [1.0]])
        y = Int8.dequantize_per_channel_asym(q, scale, offset, axis=0)
        _eq(y, torch.tensor([[0.0], [2.0]]), "int8 per-channel非对称反量化异常")

    def test_dequantize_per_group_asym_uses_group_offset_when_two_groups(self):
        """场景：两组 offset 为 0 与 1，scale 为 1。预期：第二组减 1。"""
        q = torch.tensor([1.0, 2.0, 3.0, 4.0])
        scale = torch.ones(2, 1)
        offset = torch.tensor([[0.0], [1.0]])
        y = Int8.dequantize_per_group_asym(q, scale, offset, axis=-1, group_size=2)
        _eq(y, torch.tensor([1.0, 2.0, 2.0, 3.0]), "int8 per-group非对称反量化异常")

    def test_quantize_dynamic_per_tensor_asym_detaches_params_when_x_requires_grad(self):
        """场景：非对称动态。预期：scale 与 offset 已 detach。"""
        x = torch.tensor([0.0, 10.0], requires_grad=True)
        _, scale, offset = Int8.quantize_dynamic_per_tensor_asym(x)
        self.assertFalse(scale.requires_grad)
        self.assertFalse(offset.requires_grad)

    def test_quantize_dynamic_per_channel_asym_returns_channel_params_when_two_rows(self):
        """场景：非对称动态 per-channel。预期：参数形状 (2, 1)。"""
        x = torch.tensor([[0.0, 10.0], [1.0, 2.0]])
        _, scale, offset = Int8.quantize_dynamic_per_channel_asym(x, axis=0)
        self.assertEqual(tuple(scale.shape), (2, 1))
        self.assertEqual(tuple(offset.shape), (2, 1))

    def test_quantize_dynamic_per_group_asym_returns_group_params_when_group_size_4(self):
        """场景：非对称动态 per-group。预期：参数形状 (2, 1)。"""
        x = torch.arange(8, dtype=torch.float32)
        _, scale, offset = Int8.quantize_dynamic_per_group_asym(x, axis=-1, group_size=4)
        self.assertEqual(tuple(scale.shape), (2, 1))
        self.assertEqual(tuple(offset.shape), (2, 1))

    def test_quantize_per_tensor_sym_is_identity_when_input_on_int_grid(self):
        """场景：[-127,127] 随机整数网格。预期：量化再反量化是自身。"""
        gen = torch.Generator().manual_seed(7)
        codes = torch.randint(-127, 128, (64,), generator=gen).to(torch.float32)
        q, _ = Int8.quantize_per_tensor_sym(codes, torch.tensor(1.0))
        _eq(q, codes, "int8网格恒等异常")
        y = Int8.dequantize_per_tensor_sym(q, torch.tensor(1.0))
        _eq(y, codes, "int8网格反量化恒等异常")

    def test_quantize_per_tensor_sym_is_odd_when_inside_qmax(self):
        """场景：对称量化，输入在 [-127,127]。预期：q(-x)=-q(x)。"""
        x = torch.tensor([-127.0, -3.2, -1.5, -0.5, 0.0, 0.5, 1.5, 4.6, 127.0])
        q, _ = Int8.quantize_per_tensor_sym(x, torch.tensor(1.0))
        qn, _ = Int8.quantize_per_tensor_sym(-x, torch.tensor(1.0))
        _eq(q, -qn, "int8对称奇函数异常")

    def test_random_nonzero_ratio_equals_one_when_seeds_0_1_2(self):
        """场景：随机非零 x。预期：x/x 全是 1。"""
        for seed in (0, 1, 2):
            gen = torch.Generator().manual_seed(seed)
            x = torch.randn(128, generator=gen)
            x = torch.where(x == 0, torch.ones_like(x), x)
            self.assertTrue(torch.allclose(x / x, torch.ones_like(x)), f"不变量x/x异常 seed={seed}")

    def test_quantize_per_channel_sym_roundtrip_error_at_most_half_scale_when_random(self):
        """场景：随机 seed 0/1/2。预期：|y-x| <= 0.5*scale。"""
        for seed in (0, 1, 2):
            gen = torch.Generator().manual_seed(seed)
            x = torch.randn(16, 8, generator=gen)
            scale = Int8.calc_qparam_per_channel_sym(x, axis=-1)
            q, scale = Int8.quantize_per_channel_sym(x, scale, axis=-1)
            y = Int8.dequantize_per_channel_sym(q, scale, axis=-1)
            self.assertTrue(torch.all((y - x).abs() <= 0.5 * scale + 1e-4), f"int8随机往返超界 seed={seed}")


if __name__ == "__main__":
    unittest.main()
