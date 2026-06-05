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

import torch

from msmodelslim.ir.api.impl.mx_quantization import (
    FP32_EXPONENT_BIAS,
    FP32_MIN_NORMAL,
    calculate_mx_qparam,
    mxfp_per_block_quantize,
    mxfp_per_block_dequantize,
    calculate_mxfp4_qparam,
    mxfp4_quantize,
    _quant,
    _clamp_out,
)
from msmodelslim.ir.qal import QDType, QParam, QScope, QScheme, QStorage


class TestMXConstants(unittest.TestCase):
    """测试 MX 常量"""

    def test_fp32_exponent_bias_should_be_127(self):
        """测试 FP32_EXPONENT_BIAS 值"""
        self.assertEqual(FP32_EXPONENT_BIAS, 127)

    def test_fp32_min_normal_should_be_correct(self):
        """测试 FP32_MIN_NORMAL 值"""
        self.assertEqual(FP32_MIN_NORMAL, 2 ** (-126))


class TestQuantFunction(unittest.TestCase):
    """测试 _quant 函数"""

    def test_should_quantize_with_exp(self):
        """测试带指数的量化"""
        a = torch.tensor([1.0, 2.0, 3.0])
        exp = torch.tensor([0.0, 0.0, 0.0])
        bits = 3
        exp_bits = 2

        result = _quant(a, bits, exp, exp_bits)

        self.assertEqual(result.shape, a.shape)
        self.assertTrue(torch.isfinite(result).all())

    def test_should_clip_exp_to_min(self):
        """测试指数裁剪到最小值"""
        a = torch.tensor([1.0, 2.0])
        exp = torch.tensor([-10.0, -10.0])  # 低于最小值
        bits = 3
        exp_bits = 2

        result = _quant(a, bits, exp, exp_bits)

        self.assertEqual(result.shape, a.shape)

    def test_should_handle_zero_values(self):
        """测试处理零值"""
        a = torch.tensor([0.0, 1.0, 0.0])
        exp = torch.tensor([0.0, 0.0, 0.0])
        bits = 3
        exp_bits = 2

        result = _quant(a, bits, exp, exp_bits)

        self.assertTrue(torch.isfinite(result).all())


class TestClampOutFunction(unittest.TestCase):
    """测试 _clamp_out 函数"""

    def test_should_clamp_to_max_norm(self):
        """测试裁剪到 max_norm"""
        out = torch.tensor([10.0, -10.0, 5.0])
        a = torch.tensor([10.0, -10.0, 5.0])
        max_norm = 6.0

        result = _clamp_out(out, a, max_norm)

        self.assertTrue(torch.all(result <= max_norm))
        self.assertTrue(torch.all(result >= -max_norm))

    def test_should_preserve_inf(self):
        """测试保留 Inf"""
        out = torch.tensor([10.0, 0.0])
        a = torch.tensor([float("Inf"), 0.0])
        max_norm = 6.0

        result = _clamp_out(out, a, max_norm)

        self.assertEqual(result[0], float("Inf"))

    def test_should_preserve_negative_inf(self):
        """测试保留 -Inf"""
        out = torch.tensor([-10.0, 0.0])
        a = torch.tensor([-float("Inf"), 0.0])
        max_norm = 6.0

        result = _clamp_out(out, a, max_norm)

        self.assertEqual(result[0], -float("Inf"))


class TestCalculateMxQparam(unittest.TestCase):
    """测试 calculate_mx_qparam 函数"""

    def test_should_return_valid_qparam(self):
        """测试返回有效的 QParam"""
        min_val = torch.tensor([-1.0])
        max_val = torch.tensor([1.0])

        q_param = calculate_mx_qparam(min_val, max_val, QDType.MXFP8, QScope.PER_BLOCK, True)

        self.assertIsInstance(q_param, QParam)
        self.assertIn('scale', q_param.ext)
        self.assertIn('offset', q_param.ext)

    def test_should_compute_shared_exp(self):
        """测试计算 shared_exp"""
        min_val = torch.tensor([-1.0])
        max_val = torch.tensor([1.0])

        q_param = calculate_mx_qparam(min_val, max_val, QDType.MXFP8, QScope.PER_BLOCK, True)

        # shared_exp = floor(log2(max_val)) - emax
        # floor(log2(1.0)) = 0, emax = 8
        # shared_exp = 0 - 8 = -8
        self.assertEqual(q_param.ext['scale'].item(), -8)

    def test_should_handle_zero_values(self):
        """测试处理零值"""
        min_val = torch.tensor([0.0])
        max_val = torch.tensor([0.0])

        q_param = calculate_mx_qparam(min_val, max_val, QDType.MXFP8, QScope.PER_BLOCK, True)

        self.assertIsInstance(q_param, QParam)

    def test_should_compute_keep_mask_when_flush(self):
        """测试 flush_fp32_subnorms 时计算 keep_mask"""
        min_val = torch.tensor([-1.0])
        max_val = torch.tensor([1.0])

        q_param = calculate_mx_qparam(min_val, max_val, QDType.MXFP8, QScope.PER_BLOCK, True)

        # MXFP8 的 flush_fp32_subnorms 为 False，所以 keep_mask 应该是 None
        self.assertIsNone(q_param.ext['keep_mask'])


class TestMxfpPerBlockQuantize(unittest.TestCase):
    """测试 mxfp_per_block_quantize 函数"""

    def test_should_return_qstorage(self):
        """测试返回 QStorage"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        shared_exp = torch.tensor([0.0])

        q_param = QParam(
            scheme=QScheme(dtype=QDType.MXFP8, scope=QScope.PER_BLOCK, symmetric=True),
            ext={"scale": shared_exp, "offset": torch.zeros_like(shared_exp)},
        )

        result = mxfp_per_block_quantize(QStorage(QDType.FLOAT, x), q_param)

        self.assertIsInstance(result, QStorage)
        self.assertEqual(result.dtype, QDType.MXFP8)

    def test_should_quantize_values(self):
        """测试量化值"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        shared_exp = torch.tensor([0.0])

        q_param = QParam(
            scheme=QScheme(dtype=QDType.MXFP8, scope=QScope.PER_BLOCK, symmetric=True),
            ext={"scale": shared_exp, "offset": torch.zeros_like(shared_exp)},
        )

        result = mxfp_per_block_quantize(QStorage(QDType.FLOAT, x), q_param)

        # 量化后的值应该在合理范围内
        self.assertTrue(torch.isfinite(result.value).all())

    def test_should_handle_zero_values(self):
        """测试处理零值"""
        x = torch.tensor([0.0, 0.0, 0.0, 0.0])
        shared_exp = torch.tensor([0.0])

        q_param = QParam(
            scheme=QScheme(dtype=QDType.MXFP8, scope=QScope.PER_BLOCK, symmetric=True),
            ext={"scale": shared_exp, "offset": torch.zeros_like(shared_exp)},
        )

        result = mxfp_per_block_quantize(QStorage(QDType.FLOAT, x), q_param)

        self.assertTrue(torch.isfinite(result.value).all())


class TestMxfpPerBlockDequantize(unittest.TestCase):
    """测试 mxfp_per_block_dequantize 函数"""

    def test_should_return_float_storage(self):
        """测试返回 FLOAT 存储"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        shared_exp = torch.tensor([0.0])

        q_param = QParam(
            scheme=QScheme(dtype=QDType.MXFP8, scope=QScope.PER_BLOCK, symmetric=True), ext={"scale": shared_exp}
        )

        result = mxfp_per_block_dequantize(QStorage(QDType.MXFP8, x), q_param)

        self.assertIsInstance(result, QStorage)

    def test_should_dequantize_values(self):
        """测试反量化值"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        shared_exp = torch.tensor([1.0])

        q_param = QParam(
            scheme=QScheme(dtype=QDType.MXFP8, scope=QScope.PER_BLOCK, symmetric=True), ext={"scale": shared_exp}
        )

        result = mxfp_per_block_dequantize(QStorage(QDType.MXFP8, x), q_param)

        # 反量化应该是 x * 2^shared_exp
        expected = x * (2**shared_exp)
        self.assertTrue(torch.allclose(result.value, expected))


class TestCalculateMxfp4Qparam(unittest.TestCase):
    """测试 calculate_mxfp4_qparam 函数"""

    def test_should_return_valid_qparam(self):
        """测试返回有效的 QParam"""
        min_val = torch.tensor([-1.0])
        max_val = torch.tensor([1.0])

        q_param = calculate_mxfp4_qparam(min_val, max_val, QDType.MXFP4, QScope.PER_BLOCK, True)

        self.assertIsInstance(q_param, QParam)
        self.assertIn('scale', q_param.ext)

    def test_should_compute_shared_exp(self):
        """测试计算 shared_exp"""
        min_val = torch.tensor([-1.0])
        max_val = torch.tensor([1.0])

        q_param = calculate_mxfp4_qparam(min_val, max_val, QDType.MXFP4, QScope.PER_BLOCK, True)

        # shared_exp 应该是一个有限值
        self.assertTrue(torch.isfinite(q_param.ext['scale']).all())


class TestMxfp4Quantize(unittest.TestCase):
    """测试 mxfp4_quantize 函数"""

    def test_should_return_qstorage(self):
        """测试返回 QStorage"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        shared_exp = torch.tensor([0.0])

        q_param = QParam(
            scheme=QScheme(dtype=QDType.MXFP4, scope=QScope.PER_BLOCK, symmetric=True), ext={"scale": shared_exp}
        )

        result = mxfp4_quantize(QStorage(QDType.FLOAT, x), q_param)

        self.assertIsInstance(result, QStorage)
        self.assertEqual(result.dtype, QDType.MXFP4)

    def test_should_quantize_values(self):
        """测试量化值"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        shared_exp = torch.tensor([0.0])

        q_param = QParam(
            scheme=QScheme(dtype=QDType.MXFP4, scope=QScope.PER_BLOCK, symmetric=True), ext={"scale": shared_exp}
        )

        result = mxfp4_quantize(QStorage(QDType.FLOAT, x), q_param)

        # 量化后的值应该在合理范围内
        self.assertTrue(torch.isfinite(result.value).all())

    def test_should_handle_negative_values(self):
        """测试处理负值"""
        x = torch.tensor([-1.0, -2.0, -3.0, -4.0])
        shared_exp = torch.tensor([0.0])

        q_param = QParam(
            scheme=QScheme(dtype=QDType.MXFP4, scope=QScope.PER_BLOCK, symmetric=True), ext={"scale": shared_exp}
        )

        result = mxfp4_quantize(QStorage(QDType.FLOAT, x), q_param)

        self.assertTrue(torch.isfinite(result.value).all())

    def test_should_handle_zero_values(self):
        """测试处理零值"""
        x = torch.tensor([0.0, 0.0, 0.0, 0.0])
        shared_exp = torch.tensor([0.0])

        q_param = QParam(
            scheme=QScheme(dtype=QDType.MXFP4, scope=QScope.PER_BLOCK, symmetric=True), ext={"scale": shared_exp}
        )

        result = mxfp4_quantize(QStorage(QDType.FLOAT, x), q_param)

        self.assertTrue(torch.isfinite(result.value).all())


if __name__ == '__main__':
    unittest.main()
