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

import math
import unittest

import torch

from msmodelslim.ir.qal.dtype_ops.mxfp8_ops import Mxfp8


class TestMxfp8(unittest.TestCase):
    """被测目标：Mxfp8。块长 32，calc_qparam 返回 uint8 E8M0。"""

    def test_calc_qparam_sym_raises_value_error_when_fmt_unknown(self):
        """场景：fmt=e3m4。预期：ValueError。"""
        with self.assertRaises(ValueError) as ctx:
            Mxfp8.calc_qparam_sym(torch.randn(32), fmt="e3m4")
        self.assertIn("unsupported MXFP8 fmt", str(ctx.exception))

    def test_calc_qparam_sym_returns_uint8_when_length_32(self):
        """场景：长度 32。预期：共享指数 dtype 为 uint8。"""
        se = Mxfp8.calc_qparam_sym(torch.randn(32))
        self.assertEqual(se.dtype, torch.uint8)

    def test_calc_qparam_sym_requires_grad_false_when_x_requires_grad(self):
        """场景：x 需要梯度。预期：uint8 指数不进梯度。"""
        x = torch.randn(32)
        x.requires_grad_(True)
        se = Mxfp8.calc_qparam_sym(x)
        self.assertFalse(se.requires_grad)

    def test_calc_qparam_sym_stays_inside_0_254_when_random_block(self):
        """场景：随机块。预期：E8M0 落在 [0, 254]。"""
        se = Mxfp8.calc_qparam_sym(torch.randn(32))
        self.assertTrue(torch.all((se >= 0) & (se <= 254)))

    def test_quantize_sym_returns_float32_when_dtype_omitted(self):
        """场景：省略 dtype。预期：q 为 float32。"""
        x = torch.arange(32, dtype=torch.float32) * 0.1
        q, _ = Mxfp8.quantize_dynamic_sym(x)
        self.assertEqual(q.dtype, torch.float32)

    def test_quantize_sym_returns_same_shared_exp_object_when_static(self):
        """场景：静态量化。预期：返回的共享指数是传入对象。"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0] * 8)
        exp = torch.zeros(())
        _, se = Mxfp8.quantize_sym(x, exp)
        self.assertIs(se, exp)

    def test_quantize_sym_returns_float8_when_dtype_is_e4m3fn(self):
        """场景：dtype=float8_e4m3fn。预期：q 的 dtype 为 float8。"""
        x = torch.randn(32)
        se = Mxfp8.calc_qparam_sym(x)
        q, _ = Mxfp8.quantize_sym(x, se, dtype=torch.float8_e4m3fn)
        self.assertEqual(q.dtype, torch.float8_e4m3fn)

    def test_dequantize_sym_restores_shape_when_length_not_multiple_of_32(self):
        """场景：长度 1/17/31/33/40/63。预期：反量化形状还原。"""
        for n in (1, 17, 31, 33, 40, 63):
            x = torch.arange(n, dtype=torch.float32) * 0.1
            q, se = Mxfp8.quantize_dynamic_sym(x)
            y = Mxfp8.dequantize_sym(q, se)
            self.assertEqual(tuple(y.shape), (n,), f"mxfp8长度{n}形状异常")

    def test_dequantize_sym_roundtrips_zero_when_block_all_zero(self):
        """场景：全 0 块。预期：反量化仍为 0。"""
        z = torch.zeros(32)
        q, se = Mxfp8.quantize_dynamic_sym(z)
        y = Mxfp8.dequantize_sym(q, se)
        self.assertTrue(torch.equal(y, z))

    def test_dequantize_sym_contains_nan_when_block_has_inf(self):
        """场景：块内含 inf。预期：形状 32，输出含 nan。"""
        x = torch.zeros(32)
        x[0] = float("inf")
        q, se = Mxfp8.quantize_dynamic_sym(x)
        y = Mxfp8.dequantize_sym(q, se)
        self.assertEqual(tuple(y.shape), (32,))
        self.assertTrue(torch.isnan(y).any())

    def test_quantize_dynamic_sym_detaches_shared_exp_when_called(self):
        """场景：动态量化。预期：uint8 指数已 detach。"""
        q, se = Mxfp8.quantize_dynamic_sym(torch.randn(32))
        self.assertEqual(se.dtype, torch.uint8)
        self.assertFalse(se.requires_grad)
        self.assertEqual(q.dtype, torch.float32)

    def test_quantize_sym_shared_exp_grad_equals_sum_q_ln2_when_exp_is_zero(self):
        """场景：float 共享指数 0。预期：grad = Σ q · ln2。"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0] * 8)
        x.requires_grad_(True)
        exp = torch.zeros(())
        exp.requires_grad_(True)
        q, se = Mxfp8.quantize_sym(x, exp)
        Mxfp8.dequantize_sym(q, se).sum().backward()
        expect = q.detach().sum() * math.log(2.0)
        self.assertTrue(torch.allclose(exp.grad, expect), f"mxfp8共享指数梯度异常: {exp.grad} vs {expect}")

    def test_quantize_sym_keeps_x_grad_when_dtype_omitted(self):
        """场景：默认 float32。预期：输入有梯度。"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0] * 8)
        x.requires_grad_(True)
        exp = torch.zeros(())
        exp.requires_grad_(True)
        q, se = Mxfp8.quantize_sym(x, exp)
        Mxfp8.dequantize_sym(q, se).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_quantize_dynamic_sym_restores_shape_when_axis_is_two_dims(self):
        """场景：axis=[0,1]，形状 4×8。预期：反量化形状 (4, 8)。"""
        x = torch.randn(4, 8)
        q, se = Mxfp8.quantize_dynamic_sym(x, axis=[0, 1])
        y = Mxfp8.dequantize_sym(q, se, axis=[0, 1])
        self.assertEqual(tuple(y.shape), (4, 8))
        self.assertEqual(q.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
