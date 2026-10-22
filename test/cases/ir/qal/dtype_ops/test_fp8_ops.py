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

from msmodelslim.ir.qal.dtype_ops.fp8_ops import Fp8


class TestFp8(unittest.TestCase):
    """被测目标：Fp8。默认 float32，dtype 选原生 float8。"""

    def test_calc_qparam_per_tensor_sym_equals_amax_over_448_when_fmt_e4m3(self):
        """场景：e4m3，amax=448。预期：scale=1。"""
        scale = Fp8.calc_qparam_per_tensor_sym(torch.tensor([448.0, -10.0]), fmt="e4m3")
        self.assertTrue(torch.allclose(scale, torch.tensor(1.0)))

    def test_calc_qparam_per_channel_sym_keeps_finite_channel_when_other_is_inf(self):
        """场景：第二列 inf。预期：第一列 scale 有限，第二列 inf。"""
        x = torch.tensor([[1.0, 2.0], [float("inf"), -1.0]])
        scale = Fp8.calc_qparam_per_channel_sym(x, axis=0)
        self.assertTrue(torch.isfinite(scale[0]).all())
        self.assertTrue(torch.isinf(scale[1]).all())

    def test_calc_qparam_per_tensor_sym_raises_value_error_when_fmt_unknown(self):
        """场景：fmt=e3m4。预期：ValueError。"""
        with self.assertRaises(ValueError) as ctx:
            Fp8.calc_qparam_per_tensor_sym(torch.ones(2), fmt="e3m4")
        self.assertIn("unsupported FP8 fmt", str(ctx.exception))

    def test_calc_qparam_per_tensor_sym_returns_nan_when_input_has_nan(self):
        """场景：输入含 nan。预期：scale 为 nan。"""
        scale = Fp8.calc_qparam_per_tensor_sym(torch.tensor([1.0, float("nan")]))
        self.assertTrue(torch.isnan(scale))

    def test_quantize_per_tensor_sym_returns_float32_when_dtype_omitted(self):
        """场景：省略 dtype。预期：q 为 float32。"""
        x = torch.tensor([0.0, 1.0, -2.0, 8.0])
        scale = Fp8.calc_qparam_per_tensor_sym(x, fmt="e4m3")
        q, _ = Fp8.quantize_per_tensor_sym(x, scale, fmt="e4m3")
        self.assertEqual(q.dtype, torch.float32)

    def test_quantize_per_tensor_sym_returns_same_scale_object_when_static(self):
        """场景：静态量化。预期：返回的 scale 是传入对象。"""
        scale = torch.tensor(1.0)
        _, returned = Fp8.quantize_per_tensor_sym(torch.tensor([1.0]), scale)
        self.assertIs(returned, scale)

    def test_quantize_per_tensor_sym_returns_float8_when_dtype_is_e4m3fn(self):
        """场景：dtype=float8_e4m3fn。预期：q 的 dtype 为 float8_e4m3fn。"""
        x = torch.tensor([0.0, 1.0, -2.0, 8.0])
        scale = Fp8.calc_qparam_per_tensor_sym(x, fmt="e4m3")
        q, _ = Fp8.quantize_per_tensor_sym(x, scale, fmt="e4m3", dtype=torch.float8_e4m3fn)
        self.assertEqual(q.dtype, torch.float8_e4m3fn)

    def test_dequantize_per_tensor_sym_matches_fp32_grid_when_dtype_is_float8(self):
        """场景：float8 与 float32 网格反量化。预期：结果一致。"""
        x = torch.tensor([0.0, 1.0, -2.0, 8.0])
        scale = Fp8.calc_qparam_per_tensor_sym(x, fmt="e4m3")
        q32, _ = Fp8.quantize_per_tensor_sym(x, scale, fmt="e4m3")
        q8, _ = Fp8.quantize_per_tensor_sym(x, scale, fmt="e4m3", dtype=torch.float8_e4m3fn)
        y8 = Fp8.dequantize_per_tensor_sym(q8, scale)
        y32 = Fp8.dequantize_per_tensor_sym(q32, scale)
        self.assertTrue(torch.allclose(y8, y32))

    def test_dequantize_per_tensor_sym_roundtrips_zero_when_input_all_zero(self):
        """场景：全 0。预期：反量化仍为 0。"""
        z = torch.zeros(4)
        scale = Fp8.calc_qparam_per_tensor_sym(z)
        q, scale = Fp8.quantize_per_tensor_sym(z, scale)
        y = Fp8.dequantize_per_tensor_sym(q, scale)
        self.assertTrue(torch.equal(y, z))

    def test_quantize_per_channel_sym_keeps_input_shape_when_axis_0(self):
        """场景：per-channel 量化。预期：q 形状与 x 相同。"""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        scale = Fp8.calc_qparam_per_channel_sym(x, axis=0)
        q, _ = Fp8.quantize_per_channel_sym(x, scale, axis=0)
        self.assertEqual(tuple(q.shape), (2, 2))

    def test_dequantize_per_channel_sym_keeps_finite_channel_when_other_is_inf(self):
        """场景：第二列 inf。预期：第一列有限，第二列含 nan。"""
        x = torch.tensor([[1.0, 2.0], [float("inf"), -1.0]])
        scale = Fp8.calc_qparam_per_channel_sym(x, axis=0)
        q, scale = Fp8.quantize_per_channel_sym(x, scale, axis=0)
        y = Fp8.dequantize_per_channel_sym(q, scale, axis=0)
        self.assertTrue(torch.isfinite(y[0]).all())
        self.assertTrue(torch.isnan(y[1]).any())

    def test_quantize_dynamic_per_tensor_sym_detaches_scale_when_x_requires_grad(self):
        """场景：动态量化。预期：scale 已 detach。"""
        x = torch.linspace(-10, 10, 32)
        x.requires_grad_(True)
        q, scale = Fp8.quantize_dynamic_per_tensor_sym(x, fmt="e4m3")
        self.assertFalse(scale.requires_grad)
        self.assertEqual(q.dtype, torch.float32)

    def test_quantize_dynamic_per_channel_sym_returns_channel_scale_when_two_rows(self):
        """场景：动态 per-channel。预期：scale 形状 (2, 1)。"""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        _, scale = Fp8.quantize_dynamic_per_channel_sym(x, axis=0)
        self.assertEqual(tuple(scale.shape), (2, 1))

    def test_quantize_per_tensor_sym_scale_grad_near_zero_when_fmt_e4m3(self):
        """场景：scale=1，输入在 ±10。预期：scale 留在图里且 grad≈0。"""
        x = torch.linspace(-10, 10, 32)
        x.requires_grad_(True)
        scale = torch.tensor(1.0, requires_grad=True)
        q, scale = Fp8.quantize_per_tensor_sym(x, scale, fmt="e4m3")
        Fp8.dequantize_per_tensor_sym(q, scale).sum().backward()
        self.assertIsNotNone(scale.grad)
        self.assertAlmostEqual(scale.grad.item(), 0.0, places=3)

    def test_quantize_per_tensor_sym_cuts_input_grad_when_dtype_is_float8(self):
        """场景：dtype=float8。预期：反量化不在计算图里。"""
        x = torch.linspace(-4, 4, 8, requires_grad=True)
        scale = Fp8.calc_qparam_per_tensor_sym(x.detach()).detach()
        q, scale = Fp8.quantize_per_tensor_sym(x, scale, dtype=torch.float8_e4m3fn)
        y = Fp8.dequantize_per_tensor_sym(q, scale)
        self.assertFalse(y.requires_grad)
        self.assertIsNone(x.grad)


if __name__ == "__main__":
    unittest.main()
