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

from msmodelslim.ir.qal.dtype_ops.int4_ops import Int4


def _eq(actual, expected, msg):
    if not torch.equal(actual, expected):
        raise AssertionError(f"{msg}: got {actual} expected {expected}")


class TestInt4(unittest.TestCase):
    """被测目标：Int4。默认 float32 网格，dtype=int8 为同形 8bit，不打包。"""

    def test_calc_qparam_per_tensor_sym_equals_amax_over_7_when_span_is_14(self):
        """场景：对称 per-tensor。预期：scale = 14/7。"""
        scale = Int4.calc_qparam_per_tensor_sym(torch.tensor([-3.0, 14.0]))
        self.assertTrue(torch.equal(scale, torch.tensor(14.0 / 7.0)))

    def test_calc_qparam_per_channel_sym_keeps_independent_scale_when_two_rows(self):
        """场景：两行幅值 7 与 3.5。预期：scale 为 1 与 0.5。"""
        x = torch.tensor([[7.0, -7.0], [3.5, -3.5]])
        scale = Int4.calc_qparam_per_channel_sym(x, axis=0)
        self.assertTrue(torch.allclose(scale, torch.tensor([[1.0], [0.5]])))

    def test_calc_qparam_per_group_sym_returns_two_groups_when_length_8_group_4(self):
        """场景：长度 8、组 4。预期：scale 形状 (2, 1)。"""
        scale = Int4.calc_qparam_per_group_sym(torch.arange(8, dtype=torch.float32), axis=-1, group_size=4)
        self.assertEqual(tuple(scale.shape), (2, 1))

    def test_calc_qparam_per_group_sym_raises_value_error_when_length_not_divisible(self):
        """场景：长度 6 不能被 4 整除。预期：ValueError。"""
        with self.assertRaises(ValueError) as ctx:
            Int4.calc_qparam_per_group_sym(torch.arange(6, dtype=torch.float32), axis=-1, group_size=4)
        self.assertIn("not divisible", str(ctx.exception))

    def test_quantize_per_tensor_sym_returns_float32_when_dtype_omitted(self):
        """场景：省略 dtype。预期：q 为 float32。"""
        q, _ = Int4.quantize_per_tensor_sym(torch.tensor([1.0]), torch.tensor(1.0))
        self.assertEqual(q.dtype, torch.float32)

    def test_quantize_per_tensor_sym_keeps_input_shape_when_dtype_omitted(self):
        """场景：默认量化。预期：形状与输入相同。"""
        x = torch.tensor([-9.0, -8.0, 0.0, 7.0, 8.0])
        q, _ = Int4.quantize_per_tensor_sym(x, torch.tensor(1.0))
        self.assertEqual(tuple(q.shape), tuple(x.shape))

    def test_quantize_per_tensor_sym_returns_neg8_when_input_is_qmin(self):
        """场景：输入 -8。预期：码值 -8。"""
        q, _ = Int4.quantize_per_tensor_sym(torch.tensor([-8.0]), torch.tensor(1.0))
        _eq(q, torch.tensor([-8.0]), "int4下限异常")

    def test_quantize_per_tensor_sym_returns_7_when_input_is_qmax(self):
        """场景：输入 7。预期：码值 7。"""
        q, _ = Int4.quantize_per_tensor_sym(torch.tensor([7.0]), torch.tensor(1.0))
        _eq(q, torch.tensor([7.0]), "int4上限异常")

    def test_quantize_per_tensor_sym_clamps_to_neg8_when_input_is_neg9(self):
        """场景：输入 -9。预期：饱和到 -8。"""
        q, _ = Int4.quantize_per_tensor_sym(torch.tensor([-9.0]), torch.tensor(1.0))
        _eq(q, torch.tensor([-8.0]), "int4下溢未饱和到-8")

    def test_quantize_per_tensor_sym_clamps_to_7_when_input_is_8(self):
        """场景：输入 8。预期：饱和到 7。"""
        q, _ = Int4.quantize_per_tensor_sym(torch.tensor([8.0]), torch.tensor(1.0))
        _eq(q, torch.tensor([7.0]), "int4溢出8未饱和到7")

    def test_quantize_per_tensor_sym_returns_int8_same_values_when_dtype_is_int8(self):
        """场景：dtype=int8。预期：与 float32 网格同值。"""
        x = torch.tensor([-8.0, -0.4, 0.0, 1.2, 7.0, 9.0])
        q32, _ = Int4.quantize_per_tensor_sym(x, torch.tensor(1.0))
        q8, _ = Int4.quantize_per_tensor_sym(x, torch.tensor(1.0), dtype=torch.int8)
        self.assertEqual(q8.dtype, torch.int8)
        _eq(q8.to(torch.float32), q32, "int4的8bit与float32网格不一致")

    def test_quantize_per_tensor_sym_keeps_input_shape_when_dtype_is_int8(self):
        """场景：dtype=int8。预期：形状不变，不打包。"""
        x = torch.zeros(6)
        q, _ = Int4.quantize_per_tensor_sym(x, torch.tensor(1.0), dtype=torch.int8)
        self.assertEqual(tuple(q.shape), (6,))

    def test_quantize_per_channel_sym_returns_channel_codes_when_scale_is_one(self):
        """场景：per-channel scale=1。预期：码值舍入到整数网格。"""
        x = torch.tensor([[1.2, -0.4], [2.0, -3.0]])
        q, _ = Int4.quantize_per_channel_sym(x, torch.ones(2, 1), axis=0)
        _eq(q, torch.tensor([[1.0, 0.0], [2.0, -3.0]]), "int4 per-channel码值异常")

    def test_quantize_per_group_sym_restores_length_when_group_size_4(self):
        """场景：长度 8、组 4。预期：q 长度 8，码值在 [-8,7]。"""
        x = torch.linspace(-3, 3, 8)
        q, _ = Int4.quantize_per_group_sym(x, torch.ones(2, 1), axis=-1, group_size=4)
        self.assertEqual(tuple(q.shape), (8,))
        self.assertTrue(torch.all((q >= -8) & (q <= 7)))

    def test_dequantize_per_tensor_sym_equals_q_times_scale_when_scale_is_two(self):
        """场景：码值 [1,-2]，scale=2。预期：[2,-4]。"""
        y = Int4.dequantize_per_tensor_sym(torch.tensor([1.0, -2.0]), torch.tensor(2.0))
        _eq(y, torch.tensor([2.0, -4.0]), "int4反量化异常")

    def test_dequantize_per_tensor_sym_accepts_int8_grid_when_dtype_was_int8(self):
        """场景：int8 网格反量化。预期：与 float32 网格相同。"""
        x = torch.tensor([-8.0, 0.0, 7.0])
        q8, _ = Int4.quantize_per_tensor_sym(x, torch.tensor(1.0), dtype=torch.int8)
        y = Int4.dequantize_per_tensor_sym(q8, torch.tensor(1.0))
        _eq(y, x, "int4 int8网格反量化异常")

    def test_dequantize_per_channel_sym_equals_broadcast_product_when_row_scales(self):
        """场景：两行 scale 2 与 3。预期：按行相乘。"""
        q = torch.tensor([[1.0], [-1.0]])
        y = Int4.dequantize_per_channel_sym(q, torch.tensor([[2.0], [3.0]]), axis=0)
        _eq(y, torch.tensor([[2.0], [-3.0]]), "int4 per-channel反量化异常")

    def test_dequantize_per_group_sym_equals_grouped_product_when_two_groups(self):
        """场景：两组 scale 1 与 2。预期：后一组翻倍。"""
        q = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y = Int4.dequantize_per_group_sym(q, torch.tensor([[1.0], [2.0]]), axis=-1, group_size=2)
        _eq(y, torch.tensor([1.0, 2.0, 6.0, 8.0]), "int4 per-group反量化异常")

    def test_quantize_dynamic_per_tensor_sym_detaches_scale_when_x_requires_grad(self):
        """场景：动态对称。预期：scale 已 detach，q 为 float32。"""
        x = torch.tensor([0.0, 15.0], requires_grad=True)
        q, scale = Int4.quantize_dynamic_per_tensor_sym(x)
        self.assertFalse(scale.requires_grad)
        self.assertEqual(q.dtype, torch.float32)

    def test_quantize_dynamic_per_channel_sym_returns_channel_scale_when_two_rows(self):
        """场景：动态 per-channel。预期：scale 形状 (2, 1)。"""
        x = torch.tensor([[7.0, -7.0], [1.0, -1.0]])
        _, scale = Int4.quantize_dynamic_per_channel_sym(x, axis=0)
        self.assertEqual(tuple(scale.shape), (2, 1))

    def test_quantize_dynamic_per_group_sym_returns_group_scale_when_group_size_4(self):
        """场景：动态 per-group。预期：scale 形状 (2, 1)。"""
        _, scale = Int4.quantize_dynamic_per_group_sym(torch.arange(8, dtype=torch.float32), axis=-1, group_size=4)
        self.assertEqual(tuple(scale.shape), (2, 1))

    def test_quantize_per_tensor_sym_cuts_input_grad_when_dtype_is_int8(self):
        """场景：dtype=int8。预期：输入无梯度。"""
        x = torch.tensor([0.2, 1.6], requires_grad=True)
        scale = torch.tensor(1.0, requires_grad=True)
        q, scale = Int4.quantize_per_tensor_sym(x, scale, dtype=torch.int8)
        Int4.dequantize_per_tensor_sym(q, scale).sum().backward()
        self.assertIsNone(x.grad)

    def test_calc_qparam_per_tensor_asym_equals_one_when_span_is_15(self):
        """场景：非对称 [0, 15]。预期：scale=1。"""
        scale, _offset = Int4.calc_qparam_per_tensor_asym(torch.tensor([0.0, 15.0]))
        self.assertTrue(torch.allclose(scale, torch.tensor(1.0)))

    def test_quantize_per_tensor_asym_stays_inside_int4_when_span_is_15(self):
        """场景：非对称 [0, 15]。预期：码值落在 [-8, 7]。"""
        x = torch.tensor([0.0, 15.0])
        scale, offset = Int4.calc_qparam_per_tensor_asym(x)
        q, _, _ = Int4.quantize_per_tensor_asym(x, scale, offset)
        self.assertTrue(torch.all((q >= -8) & (q <= 7)))

    def test_calc_qparam_per_channel_asym_returns_keepdim_when_axis_0(self):
        """场景：非对称 per-channel。预期：参数形状 (2, 1)。"""
        scale, offset = Int4.calc_qparam_per_channel_asym(torch.tensor([[0.0, 7.0], [1.0, 2.0]]), axis=0)
        self.assertEqual(tuple(scale.shape), (2, 1))
        self.assertEqual(tuple(offset.shape), (2, 1))

    def test_calc_qparam_per_group_asym_returns_two_groups_when_length_8_group_4(self):
        """场景：非对称 per-group。预期：参数形状 (2, 1)。"""
        scale, offset = Int4.calc_qparam_per_group_asym(torch.arange(8, dtype=torch.float32), axis=-1, group_size=4)
        self.assertEqual(tuple(scale.shape), (2, 1))
        self.assertEqual(tuple(offset.shape), (2, 1))

    def test_quantize_per_channel_asym_returns_float32_when_dtype_omitted(self):
        """场景：非对称 per-channel。预期：q 为 float32。"""
        x = torch.tensor([[0.0, 7.0], [1.0, 2.0]])
        scale, offset = Int4.calc_qparam_per_channel_asym(x, axis=0)
        q, _, _ = Int4.quantize_per_channel_asym(x, scale, offset, axis=0)
        self.assertEqual(q.dtype, torch.float32)

    def test_quantize_per_group_asym_restores_length_when_group_size_4(self):
        """场景：非对称 per-group。预期：q 长度 8。"""
        x = torch.arange(8, dtype=torch.float32)
        scale, offset = Int4.calc_qparam_per_group_asym(x, axis=-1, group_size=4)
        q, _, _ = Int4.quantize_per_group_asym(x, scale, offset, axis=-1, group_size=4)
        self.assertEqual(tuple(q.shape), (8,))

    def test_dequantize_per_tensor_asym_equals_q_minus_offset_times_scale_when_offset_zero(self):
        """场景：offset=0、scale=2。预期：[2, -4]。"""
        y = Int4.dequantize_per_tensor_asym(torch.tensor([1.0, -2.0]), torch.tensor(2.0), torch.tensor(0.0))
        _eq(y, torch.tensor([2.0, -4.0]), "int4非对称反量化异常")

    def test_dequantize_per_channel_asym_uses_offset_when_row_params_differ(self):
        """场景：第二行 offset=1、scale=2。预期：(2-1)*2=2。"""
        q = torch.tensor([[0.0], [2.0]])
        y = Int4.dequantize_per_channel_asym(q, torch.tensor([[1.0], [2.0]]), torch.tensor([[0.0], [1.0]]), axis=0)
        _eq(y, torch.tensor([[0.0], [2.0]]), "int4 per-channel非对称反量化异常")

    def test_dequantize_per_group_asym_uses_group_offset_when_two_groups(self):
        """场景：第二组 offset=1。预期：第二组减 1。"""
        q = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y = Int4.dequantize_per_group_asym(q, torch.ones(2, 1), torch.tensor([[0.0], [1.0]]), axis=-1, group_size=2)
        _eq(y, torch.tensor([1.0, 2.0, 2.0, 3.0]), "int4 per-group非对称反量化异常")

    def test_quantize_dynamic_per_tensor_asym_detaches_params_when_x_requires_grad(self):
        """场景：非对称动态。预期：参数已 detach。"""
        x = torch.tensor([0.0, 15.0], requires_grad=True)
        _, scale, offset = Int4.quantize_dynamic_per_tensor_asym(x)
        self.assertFalse(scale.requires_grad)
        self.assertFalse(offset.requires_grad)

    def test_quantize_dynamic_per_channel_asym_returns_channel_params_when_two_rows(self):
        """场景：非对称动态 per-channel。预期：参数形状 (2, 1)。"""
        _, scale, offset = Int4.quantize_dynamic_per_channel_asym(torch.tensor([[0.0, 7.0], [1.0, 2.0]]), axis=0)
        self.assertEqual(tuple(scale.shape), (2, 1))
        self.assertEqual(tuple(offset.shape), (2, 1))

    def test_quantize_dynamic_per_group_asym_returns_group_params_when_group_size_4(self):
        """场景：非对称动态 per-group。预期：参数形状 (2, 1)。"""
        _, scale, offset = Int4.quantize_dynamic_per_group_asym(
            torch.arange(8, dtype=torch.float32), axis=-1, group_size=4
        )
        self.assertEqual(tuple(scale.shape), (2, 1))
        self.assertEqual(tuple(offset.shape), (2, 1))

    def test_quantize_per_tensor_sym_roundtrip_error_at_most_half_scale_when_random(self):
        """场景：随机 seed 0/1/2。预期：|y-x| <= 0.5*scale。"""
        for seed in (0, 1, 2):
            gen = torch.Generator().manual_seed(seed)
            x = torch.randn(4, 8, generator=gen)
            scale = Int4.calc_qparam_per_tensor_sym(x)
            q, scale = Int4.quantize_per_tensor_sym(x, scale)
            y = Int4.dequantize_per_tensor_sym(q, scale)
            self.assertTrue(torch.all((y - x).abs() <= 0.5 * scale + 1e-4), f"int4随机往返超界 seed={seed}")


if __name__ == "__main__":
    unittest.main()
