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

# 导入实现以确保注册
from msmodelslim.ir.api import calculate_qparam, quantize, dequantize, fake_quantize
from msmodelslim.ir.qal import QDType, QParam, QScope, QStorage
from msmodelslim.ir.api.impl.fp_quantization import FP8_E4M3_MAX, FP8_E4M3_MIN


class TestFP8Constants(unittest.TestCase):
    """测试 FP8 常量"""

    def test_fp8_e4m3_max_should_be_448(self):
        """测试 FP8_E4M3_MAX 值"""
        self.assertEqual(FP8_E4M3_MAX, 448)

    def test_fp8_e4m3_min_should_be_minus_448(self):
        """测试 FP8_E4M3_MIN 值"""
        self.assertEqual(FP8_E4M3_MIN, -448)


class TestCalculateFP8Qparam(unittest.TestCase):
    """测试 FP8 的 calculate_qparam"""

    def test_per_tensor_symmetric_should_return_valid_qparam(self):
        """测试 PER_TENSOR 对称量化参数"""
        min_val = torch.tensor([-10.0])
        max_val = torch.tensor([10.0])
        q_param = calculate_qparam(min_val, max_val, QDType.FP8_E4M3, QScope.PER_TENSOR, True)
        self.assertIsInstance(q_param, QParam)
        self.assertIn('scale', q_param.ext)
        self.assertIn('offset', q_param.ext)

    def test_per_channel_symmetric_should_return_valid_qparam(self):
        """测试 PER_CHANNEL 对称量化参数"""
        min_val = torch.tensor([-10.0, -20.0, -30.0])
        max_val = torch.tensor([10.0, 20.0, 30.0])
        q_param = calculate_qparam(min_val, max_val, QDType.FP8_E4M3, QScope.PER_CHANNEL, True)
        self.assertIsInstance(q_param, QParam)
        self.assertEqual(q_param.ext['scale'].shape, min_val.shape)

    def test_per_token_symmetric_should_return_valid_qparam(self):
        """测试 PER_TOKEN 对称量化参数"""
        min_val = torch.tensor([[-10.0, -20.0]])
        max_val = torch.tensor([[10.0, 20.0]])
        q_param = calculate_qparam(min_val, max_val, QDType.FP8_E4M3, QScope.PER_TOKEN, True)
        self.assertIsInstance(q_param, QParam)

    def test_scale_should_be_correct(self):
        """测试 scale 计算正确"""
        min_val = torch.tensor([-100.0])
        max_val = torch.tensor([100.0])
        q_param = calculate_qparam(min_val, max_val, QDType.FP8_E4M3, QScope.PER_TENSOR, True)
        # scale = amax / FP8_E4M3_MAX = 100 / 448
        expected_scale = torch.tensor([100.0 / 448.0])
        self.assertTrue(torch.allclose(q_param.ext['scale'], expected_scale))

    def test_offset_should_be_zeros(self):
        """测试 offset 为零"""
        min_val = torch.tensor([-10.0])
        max_val = torch.tensor([10.0])
        q_param = calculate_qparam(min_val, max_val, QDType.FP8_E4M3, QScope.PER_TENSOR, True)
        self.assertTrue(torch.all(q_param.ext['offset'] == 0))


class TestFP8Quantize(unittest.TestCase):
    """测试 FP8 的 quantize"""

    def test_should_return_qstorage(self):
        """测试返回 QStorage"""
        x = torch.tensor([100.0, 200.0, 300.0])
        min_val = torch.tensor([-400.0])
        max_val = torch.tensor([400.0])
        q_param = calculate_qparam(min_val, max_val, QDType.FP8_E4M3, QScope.PER_TENSOR, True)
        result = quantize(QStorage(QDType.FLOAT, x), q_param)
        self.assertIsInstance(result, QStorage)
        self.assertEqual(result.dtype, QDType.FP8_E4M3)

    def test_should_clamp_values(self):
        """测试值被限制在范围内"""
        x = torch.tensor([1000.0, -1000.0])  # 超出 FP8 范围
        min_val = torch.tensor([-1000.0])
        max_val = torch.tensor([1000.0])
        q_param = calculate_qparam(min_val, max_val, QDType.FP8_E4M3, QScope.PER_TENSOR, True)
        result = quantize(QStorage(QDType.FLOAT, x), q_param)
        self.assertTrue(torch.all(result.value <= FP8_E4M3_MAX))
        self.assertTrue(torch.all(result.value >= FP8_E4M3_MIN))


class TestFP8Dequantize(unittest.TestCase):
    """测试 FP8 的 dequantize"""

    def test_should_return_float_storage(self):
        """测试返回 FLOAT 类型的 QStorage"""
        x = torch.tensor([100.0, 200.0, 300.0])
        min_val = torch.tensor([-400.0])
        max_val = torch.tensor([400.0])
        q_param = calculate_qparam(min_val, max_val, QDType.FP8_E4M3, QScope.PER_TENSOR, True)
        # 先量化
        x_q = quantize(QStorage(QDType.FLOAT, x), q_param)
        # 再反量化
        result = dequantize(x_q, q_param)
        self.assertIsInstance(result, QStorage)
        self.assertEqual(result.dtype, QDType.FLOAT)

    def test_should_approximate_identity_with_fake_quantize(self):
        """测试 fake_quantize 近似恒等变换"""
        x = torch.tensor([100.0, 200.0, 300.0, 400.0])
        min_val = torch.tensor([-400.0])
        max_val = torch.tensor([400.0])
        q_param = calculate_qparam(min_val, max_val, QDType.FP8_E4M3, QScope.PER_TENSOR, True)
        result = fake_quantize(QStorage(QDType.FLOAT, x), q_param)
        # 由于量化误差，结果应该近似但不完全等于输入
        self.assertTrue(torch.allclose(result.value, x, atol=10.0))


if __name__ == '__main__':
    unittest.main()
