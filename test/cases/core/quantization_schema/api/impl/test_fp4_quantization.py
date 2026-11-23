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

from msmodelslim.core.quantization_schema.api import calculate_qparam, quantize, dequantize, fake_quantize
from msmodelslim.core.quantization_schema.qal import QDType, QParam, QScope, QStorage
from msmodelslim.core.quantization_schema.api.impl.fp_quantization import FP4_E2M1_MAX, FP4_E2M1_MIN


class TestFP4Constants(unittest.TestCase):
    """测试 FP4_E2M1 常量"""

    def test_fp4_e2m1_max_should_be_6(self):
        self.assertEqual(FP4_E2M1_MAX, 6.0)

    def test_fp4_e2m1_min_should_be_minus_6(self):
        self.assertEqual(FP4_E2M1_MIN, -6.0)


class TestCalculateFP4Qparam(unittest.TestCase):
    """测试 FP4_E2M1 的 calculate_qparam"""

    def test_per_tensor_symmetric_should_return_valid_qparam(self):
        min_val = torch.tensor([-10.0])
        max_val = torch.tensor([10.0])
        q_param = calculate_qparam(min_val, max_val, QDType.FP4_E2M1, QScope.PER_TENSOR, True)
        self.assertIsInstance(q_param, QParam)
        self.assertIn('scale', q_param.ext)
        self.assertIn('offset', q_param.ext)

    def test_per_channel_symmetric_should_return_valid_qparam(self):
        min_val = torch.tensor([-10.0, -20.0, -30.0])
        max_val = torch.tensor([10.0, 20.0, 30.0])
        q_param = calculate_qparam(min_val, max_val, QDType.FP4_E2M1, QScope.PER_CHANNEL, True)
        self.assertIsInstance(q_param, QParam)
        self.assertEqual(q_param.ext['scale'].shape, min_val.shape)

    def test_per_token_symmetric_should_return_valid_qparam(self):
        min_val = torch.tensor([[-10.0, -20.0]])
        max_val = torch.tensor([[10.0, 20.0]])
        q_param = calculate_qparam(min_val, max_val, QDType.FP4_E2M1, QScope.PER_TOKEN, True)
        self.assertIsInstance(q_param, QParam)

    def test_per_head_symmetric_should_return_valid_qparam(self):
        min_val = torch.tensor([1.0, 2.0, 3.0, 4.0])
        max_val = torch.tensor([5.0, 6.0, 7.0, 8.0])
        q_param = calculate_qparam(min_val, max_val, QDType.FP4_E2M1, QScope.PER_HEAD, True)
        self.assertIsInstance(q_param, QParam)
        self.assertIn('scale', q_param.ext)

    def test_scale_should_be_correct(self):
        min_val = torch.tensor([-12.0])
        max_val = torch.tensor([12.0])
        q_param = calculate_qparam(min_val, max_val, QDType.FP4_E2M1, QScope.PER_TENSOR, True)
        expected_scale = torch.tensor([12.0 / 6.0])
        self.assertTrue(torch.allclose(q_param.ext['scale'], expected_scale))

    def test_offset_should_be_zeros(self):
        min_val = torch.tensor([-10.0])
        max_val = torch.tensor([10.0])
        q_param = calculate_qparam(min_val, max_val, QDType.FP4_E2M1, QScope.PER_TENSOR, True)
        self.assertTrue(torch.all(q_param.ext['offset'] == 0))


class TestFP4Quantize(unittest.TestCase):
    """测试 FP4_E2M1 的 quantize"""

    def test_should_return_qstorage(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        min_val = torch.tensor([-4.0])
        max_val = torch.tensor([4.0])
        q_param = calculate_qparam(min_val, max_val, QDType.FP4_E2M1, QScope.PER_TENSOR, True)
        result = quantize(QStorage(QDType.FLOAT, x), q_param)
        self.assertIsInstance(result, QStorage)
        self.assertEqual(result.dtype, QDType.FP4_E2M1)

    def test_should_clamp_values(self):
        x = torch.tensor([100.0, -100.0])
        min_val = torch.tensor([-100.0])
        max_val = torch.tensor([100.0])
        q_param = calculate_qparam(min_val, max_val, QDType.FP4_E2M1, QScope.PER_TENSOR, True)
        result = quantize(QStorage(QDType.FLOAT, x), q_param)
        self.assertTrue(torch.all(result.value <= FP4_E2M1_MAX))
        self.assertTrue(torch.all(result.value >= FP4_E2M1_MIN))

    def test_per_channel_quantize_should_work(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        min_val = torch.tensor([-3.0, -3.0, -3.0])
        max_val = torch.tensor([3.0, 3.0, 3.0])
        q_param = calculate_qparam(min_val, max_val, QDType.FP4_E2M1, QScope.PER_CHANNEL, True)
        result = quantize(QStorage(QDType.FLOAT, x), q_param)
        self.assertIsInstance(result, QStorage)
        self.assertEqual(result.dtype, QDType.FP4_E2M1)
        self.assertEqual(result.value.shape, x.shape)


class TestFP4Dequantize(unittest.TestCase):
    """测试 FP4_E2M1 的 dequantize"""

    def test_should_return_float_storage(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        min_val = torch.tensor([-4.0])
        max_val = torch.tensor([4.0])
        q_param = calculate_qparam(min_val, max_val, QDType.FP4_E2M1, QScope.PER_TENSOR, True)
        x_q = quantize(QStorage(QDType.FLOAT, x), q_param)
        result = dequantize(x_q, q_param)
        self.assertIsInstance(result, QStorage)
        self.assertEqual(result.dtype, QDType.FLOAT)

    def test_should_approximate_identity_with_fake_quantize(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        min_val = torch.tensor([-4.0])
        max_val = torch.tensor([4.0])
        q_param = calculate_qparam(min_val, max_val, QDType.FP4_E2M1, QScope.PER_TENSOR, True)
        result = fake_quantize(QStorage(QDType.FLOAT, x), q_param)
        self.assertTrue(torch.allclose(result.value, x, atol=1.0))

    def test_round_trip_per_channel(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
        min_val = torch.tensor([-6.0, -6.0])
        max_val = torch.tensor([6.0, 6.0])
        q_param = calculate_qparam(min_val, max_val, QDType.FP4_E2M1, QScope.PER_CHANNEL, True)
        result = fake_quantize(QStorage(QDType.FLOAT, x), q_param)
        self.assertEqual(result.value.shape, x.shape)
        self.assertTrue(torch.isfinite(result.value).all())


if __name__ == '__main__':
    unittest.main()
