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


class TestCalculateInt8Qparam(unittest.TestCase):
    """测试 INT8 的 calculate_qparam"""

    def test_per_tensor_symmetric_should_return_valid_qparam(self):
        """测试 PER_TENSOR 对称量化参数"""
        min_val = torch.tensor([-10.0])
        max_val = torch.tensor([10.0])
        q_param = calculate_qparam(min_val, max_val, QDType.INT8, QScope.PER_TENSOR, True)
        self.assertIsInstance(q_param, QParam)
        self.assertIn('scale', q_param.ext)
        self.assertIn('offset', q_param.ext)
        # 对称量化 offset 应该为 0
        self.assertTrue(torch.all(q_param.ext['offset'] == 0))

    def test_per_tensor_asymmetric_should_return_valid_qparam(self):
        """测试 PER_TENSOR 非对称量化参数"""
        min_val = torch.tensor([-10.0])
        max_val = torch.tensor([10.0])
        q_param = calculate_qparam(min_val, max_val, QDType.INT8, QScope.PER_TENSOR, False)
        self.assertIsInstance(q_param, QParam)
        self.assertIn('scale', q_param.ext)
        self.assertIn('offset', q_param.ext)

    def test_per_channel_symmetric_should_return_valid_qparam(self):
        """测试 PER_CHANNEL 对称量化参数"""
        min_val = torch.tensor([-10.0, -20.0, -30.0])
        max_val = torch.tensor([10.0, 20.0, 30.0])
        q_param = calculate_qparam(min_val, max_val, QDType.INT8, QScope.PER_CHANNEL, True)
        self.assertIsInstance(q_param, QParam)
        self.assertEqual(q_param.ext['scale'].shape, min_val.shape)

    def test_per_channel_asymmetric_should_return_valid_qparam(self):
        """测试 PER_CHANNEL 非对称量化参数"""
        min_val = torch.tensor([-10.0, -20.0, -30.0])
        max_val = torch.tensor([10.0, 20.0, 30.0])
        q_param = calculate_qparam(min_val, max_val, QDType.INT8, QScope.PER_CHANNEL, False)
        self.assertIsInstance(q_param, QParam)

    def test_per_token_symmetric_should_return_valid_qparam(self):
        """测试 PER_TOKEN 对称量化参数"""
        min_val = torch.tensor([[-10.0, -20.0]])
        max_val = torch.tensor([[10.0, 20.0]])
        q_param = calculate_qparam(min_val, max_val, QDType.INT8, QScope.PER_TOKEN, True)
        self.assertIsInstance(q_param, QParam)

    def test_per_head_symmetric_should_return_valid_qparam(self):
        """测试 PER_HEAD 对称量化参数"""
        min_val = torch.tensor([-10.0, -20.0])
        max_val = torch.tensor([10.0, 20.0])
        q_param = calculate_qparam(min_val, max_val, QDType.INT8, QScope.PER_HEAD, True)
        self.assertIsInstance(q_param, QParam)

    def test_symmetric_scale_should_be_correct(self):
        """测试对称量化 scale 计算"""
        min_val = torch.tensor([-100.0])
        max_val = torch.tensor([100.0])
        q_param = calculate_qparam(min_val, max_val, QDType.INT8, QScope.PER_TENSOR, True)
        # scale = max_val / 127 = 100 / 127
        expected_scale = torch.tensor([100.0 / 127.0])
        self.assertTrue(torch.allclose(q_param.ext['scale'], expected_scale))

    def test_asymmetric_scale_should_be_correct(self):
        """测试非对称量化 scale 计算"""
        min_val = torch.tensor([-100.0])
        max_val = torch.tensor([200.0])
        q_param = calculate_qparam(min_val, max_val, QDType.INT8, QScope.PER_TENSOR, False)
        # scale = (max - min) / 255 = 300 / 255
        expected_scale = torch.tensor([300.0 / 255.0])
        self.assertTrue(torch.allclose(q_param.ext['scale'], expected_scale))


class TestCalculateInt4Qparam(unittest.TestCase):
    """测试 INT4 的 calculate_qparam"""

    def test_per_channel_symmetric_should_return_valid_qparam(self):
        """测试 PER_CHANNEL 对称量化参数"""
        min_val = torch.tensor([-10.0, -20.0])
        max_val = torch.tensor([10.0, 20.0])
        q_param = calculate_qparam(min_val, max_val, QDType.INT4, QScope.PER_CHANNEL, True)
        self.assertIsInstance(q_param, QParam)
        # INT4 对称 max_bound = 7
        expected_scale = torch.tensor([10.0 / 7.0, 20.0 / 7.0])
        self.assertTrue(torch.allclose(q_param.ext['scale'], expected_scale))

    def test_per_channel_asymmetric_should_return_valid_qparam(self):
        """测试 PER_CHANNEL 非对称量化参数"""
        min_val = torch.tensor([-10.0, -20.0])
        max_val = torch.tensor([10.0, 20.0])
        q_param = calculate_qparam(min_val, max_val, QDType.INT4, QScope.PER_CHANNEL, False)
        self.assertIsInstance(q_param, QParam)

    def test_per_token_symmetric_should_return_valid_qparam(self):
        """测试 PER_TOKEN 对称量化参数"""
        min_val = torch.tensor([[-10.0, -20.0]])
        max_val = torch.tensor([[10.0, 20.0]])
        q_param = calculate_qparam(min_val, max_val, QDType.INT4, QScope.PER_TOKEN, True)
        self.assertIsInstance(q_param, QParam)

    def test_per_token_asymmetric_should_return_valid_qparam(self):
        """测试 PER_TOKEN 非对称量化参数"""
        min_val = torch.tensor([[-10.0, -20.0]])
        max_val = torch.tensor([[10.0, 20.0]])
        q_param = calculate_qparam(min_val, max_val, QDType.INT4, QScope.PER_TOKEN, False)
        self.assertIsInstance(q_param, QParam)


class TestCalculateInt8PerGroupQparam(unittest.TestCase):
    """测试 INT8 PER_GROUP 的 calculate_qparam"""

    def test_symmetric_should_return_valid_qparam_with_group_size(self):
        """测试对称量化参数包含 group_size"""
        min_val = torch.tensor([-10.0])
        max_val = torch.tensor([10.0])
        q_param = calculate_qparam(min_val, max_val, QDType.INT8, QScope.PER_GROUP, True)
        self.assertIsInstance(q_param, QParam)
        self.assertIn('group_size', q_param.ext)
        self.assertEqual(q_param.ext['group_size'], -1)  # 默认值

    def test_asymmetric_should_return_valid_qparam_with_group_size(self):
        """测试非对称量化参数包含 group_size"""
        min_val = torch.tensor([-10.0])
        max_val = torch.tensor([10.0])
        q_param = calculate_qparam(min_val, max_val, QDType.INT8, QScope.PER_GROUP, False)
        self.assertIsInstance(q_param, QParam)
        self.assertIn('group_size', q_param.ext)


class TestInt8Quantize(unittest.TestCase):
    """测试 INT8 的 quantize"""

    def test_per_tensor_symmetric_should_return_int8_storage(self):
        """测试 PER_TENSOR 对称量化返回 INT8 存储"""
        x = torch.tensor([100.0, 200.0, 300.0])
        min_val = torch.tensor([-400.0])
        max_val = torch.tensor([400.0])
        q_param = calculate_qparam(min_val, max_val, QDType.INT8, QScope.PER_TENSOR, True)
        result = quantize(QStorage(QDType.FLOAT, x), q_param)
        self.assertIsInstance(result, QStorage)
        self.assertEqual(result.dtype, QDType.INT8)

    def test_should_clamp_values(self):
        """测试值被限制在 INT8 范围内"""
        x = torch.tensor([1000.0, -1000.0])  # 超出 INT8 范围
        min_val = torch.tensor([-1000.0])
        max_val = torch.tensor([1000.0])
        q_param = calculate_qparam(min_val, max_val, QDType.INT8, QScope.PER_TENSOR, True)
        result = quantize(QStorage(QDType.FLOAT, x), q_param)
        # INT8 范围: [-128, 127]
        self.assertTrue(torch.all(result.value <= 127))
        self.assertTrue(torch.all(result.value >= -128))


class TestInt4Quantize(unittest.TestCase):
    """测试 INT4 的 quantize"""

    def test_per_channel_symmetric_should_return_int4_storage(self):
        """测试 PER_CHANNEL 对称量化返回 INT4 存储"""
        x = torch.tensor([5.0, -5.0])
        min_val = torch.tensor([-10.0, -10.0])
        max_val = torch.tensor([10.0, 10.0])
        q_param = calculate_qparam(min_val, max_val, QDType.INT4, QScope.PER_CHANNEL, True)
        result = quantize(QStorage(QDType.FLOAT, x), q_param)
        self.assertIsInstance(result, QStorage)
        self.assertEqual(result.dtype, QDType.INT4)

    def test_should_clamp_values(self):
        """测试值被限制在 INT4 范围内"""
        x = torch.tensor([100.0, -100.0])  # 超出 INT4 范围
        min_val = torch.tensor([-100.0, -100.0])
        max_val = torch.tensor([100.0, 100.0])
        q_param = calculate_qparam(min_val, max_val, QDType.INT4, QScope.PER_CHANNEL, True)
        result = quantize(QStorage(QDType.FLOAT, x), q_param)
        # INT4 范围: [-8, 7]
        self.assertTrue(torch.all(result.value <= 7))
        self.assertTrue(torch.all(result.value >= -8))


class TestInt8Dequantize(unittest.TestCase):
    """测试 INT8 的 dequantize"""

    def test_should_return_float_storage(self):
        """测试返回 FLOAT 类型的 QStorage"""
        x = torch.tensor([100.0, 200.0, 300.0])
        min_val = torch.tensor([-400.0])
        max_val = torch.tensor([400.0])
        q_param = calculate_qparam(min_val, max_val, QDType.INT8, QScope.PER_TENSOR, True)
        # 先量化
        x_q = quantize(QStorage(QDType.FLOAT, x), q_param)
        # 再反量化
        result = dequantize(x_q, q_param)
        self.assertIsInstance(result, QStorage)
        self.assertEqual(result.dtype, QDType.FLOAT)


class TestInt8FakeQuantize(unittest.TestCase):
    """测试 INT8 的 fake_quantize"""

    def test_should_approximate_identity(self):
        """测试 fake_quantize 近似恒等变换"""
        x = torch.tensor([100.0, 200.0, 300.0, 400.0])
        min_val = torch.tensor([-400.0])
        max_val = torch.tensor([400.0])
        q_param = calculate_qparam(min_val, max_val, QDType.INT8, QScope.PER_TENSOR, True)
        result = fake_quantize(QStorage(QDType.FLOAT, x), q_param)
        # 由于量化误差，结果应该近似但不完全等于输入
        self.assertTrue(torch.allclose(result.value, x, atol=5.0))

    def test_should_preserve_shape(self):
        """测试保持形状"""
        x = torch.randn(2, 3, 4)
        min_val = torch.tensor([-1.0])
        max_val = torch.tensor([1.0])
        q_param = calculate_qparam(min_val, max_val, QDType.INT8, QScope.PER_TENSOR, True)
        result = fake_quantize(QStorage(QDType.FLOAT, x), q_param)
        self.assertEqual(result.value.shape, x.shape)


class TestInt8PerGroupQuantize(unittest.TestCase):
    """测试 INT8 PER_GROUP 的 quantize"""

    def test_should_return_int8_storage(self):
        """测试返回 INT8 存储"""
        x = torch.randn(4, 8)
        group_size = 4
        min_val = torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        max_val = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        q_param = calculate_qparam(min_val, max_val, QDType.INT8, QScope.PER_GROUP, True)
        q_param.ext['group_size'] = group_size
        result = quantize(QStorage(QDType.FLOAT, x), q_param)
        self.assertIsInstance(result, QStorage)
        self.assertEqual(result.dtype, QDType.INT8)

    def test_should_raise_for_invalid_group_size(self):
        """测试无效 group_size 抛出异常"""
        x = torch.randn(4, 8)
        min_val = torch.tensor([-1.0])
        max_val = torch.tensor([1.0])
        q_param = calculate_qparam(min_val, max_val, QDType.INT8, QScope.PER_GROUP, True)
        q_param.ext['group_size'] = -1  # 无效
        with self.assertRaises(Exception):
            quantize(QStorage(QDType.FLOAT, x), q_param)


class TestInt8PerGroupDequantize(unittest.TestCase):
    """测试 INT8 PER_GROUP 的 dequantize"""

    def test_should_return_float_storage(self):
        """测试返回 FLOAT 存储"""
        x = torch.randn(4, 8)
        group_size = 4
        min_val = torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        max_val = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        q_param = calculate_qparam(min_val, max_val, QDType.INT8, QScope.PER_GROUP, True)
        q_param.ext['group_size'] = group_size
        # 先量化
        x_q = quantize(QStorage(QDType.FLOAT, x), q_param)
        # 再反量化
        result = dequantize(x_q, q_param)
        self.assertIsInstance(result, QStorage)
        self.assertEqual(result.dtype, QDType.FLOAT)


class TestReshapePadTensorByGroupSize(unittest.TestCase):
    """测试 reshape_pad_tensor_by_group_size 函数"""

    def test_group_size_0_should_reshape_to_2d(self):
        """测试 group_size=0 时重塑为 2D"""
        from msmodelslim.ir.api.impl.int_quantization import reshape_pad_tensor_by_group_size

        data = torch.randn(2, 3, 4)
        result, orig_shape, pad_len = reshape_pad_tensor_by_group_size(data, 0)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], 24)

    def test_group_size_minus1_should_not_reshape(self):
        """测试 group_size=-1 时不重塑"""
        from msmodelslim.ir.api.impl.int_quantization import reshape_pad_tensor_by_group_size

        data = torch.randn(2, 4)
        result, orig_shape, pad_len = reshape_pad_tensor_by_group_size(data, -1)
        self.assertEqual(result.shape, (2, 4))

    def test_group_size_divisible_should_reshape(self):
        """测试 group_size 可整除时重塑"""
        from msmodelslim.ir.api.impl.int_quantization import reshape_pad_tensor_by_group_size

        data = torch.randn(2, 8)
        result, orig_shape, pad_len = reshape_pad_tensor_by_group_size(data, 4)
        self.assertEqual(result.shape, (4, 4))

    def test_group_size_not_divisible_should_pad_and_reshape(self):
        """测试 group_size 不可整除时填充并重塑"""
        from msmodelslim.ir.api.impl.int_quantization import reshape_pad_tensor_by_group_size

        data = torch.randn(2, 10)
        result, orig_shape, pad_len = reshape_pad_tensor_by_group_size(data, 4)
        self.assertEqual(result.shape[1], 4)
        self.assertEqual(pad_len, 2)


class TestRevertTensorByPad(unittest.TestCase):
    """测试 revert_tensor_by_pad 函数"""

    def test_no_padding_should_reshape(self):
        """测试无填充时重塑"""
        from msmodelslim.ir.api.impl.int_quantization import revert_tensor_by_pad

        data = torch.tensor([1, 2, 3, 4, 5, 6])
        result = revert_tensor_by_pad(data, (2, 3), 0)
        self.assertEqual(result.shape, (2, 3))

    def test_with_padding_should_remove_padding(self):
        """测试有填充时移除填充"""
        from msmodelslim.ir.api.impl.int_quantization import revert_tensor_by_pad

        data = torch.arange(12).reshape(3, 4)
        result = revert_tensor_by_pad(data, (3, 3), 1)
        self.assertEqual(result.shape, (3, 3))


if __name__ == '__main__':
    unittest.main()
