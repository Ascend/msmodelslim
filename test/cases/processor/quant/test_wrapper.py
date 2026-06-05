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
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from msmodelslim.processor.quant.autoround_utils.wrapper import (
    reshape_and_pad_tensor,
    get_scale_shape,
    WrapperLinear,
)


class TestReshapeAndPadTensor(unittest.TestCase):
    """测试 reshape_and_pad_tensor 函数"""

    def test_group_size_0_should_reshape_to_flat(self):
        """测试 group_size=0 时重塑为 1D"""
        v = torch.randn(2, 3, 4)
        result = reshape_and_pad_tensor(v, 0)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], 24)

    def test_group_size_minus1_should_not_reshape(self):
        """测试 group_size=-1 时不重塑"""
        v = torch.randn(2, 4)
        result = reshape_and_pad_tensor(v, -1)
        self.assertEqual(result.shape, (2, 4))

    def test_small_tensor_should_not_reshape(self):
        """测试小张量不重塑"""
        v = torch.randn(2, 4)
        result = reshape_and_pad_tensor(v, 8)
        self.assertEqual(result.shape, (2, 4))

    def test_divisible_should_reshape(self):
        """测试可整除时重塑"""
        v = torch.randn(2, 8)
        result = reshape_and_pad_tensor(v, 4)
        self.assertEqual(result.shape, (4, 4))

    def test_not_divisible_should_pad_and_reshape(self):
        """测试不可整除时填充并重塑"""
        v = torch.randn(2, 10)
        result = reshape_and_pad_tensor(v, 4)
        self.assertEqual(result.shape[1], 4)


class TestGetScaleShape(unittest.TestCase):
    """测试 get_scale_shape 函数"""

    def test_group_size_0_should_return_1(self):
        """测试 group_size=0 返回 1"""
        weight = torch.randn(10, 20)
        result = get_scale_shape(weight, 0)
        self.assertEqual(result, 1)

    def test_group_size_minus1_should_return_weight_dim0(self):
        """测试 group_size=-1 返回 weight.shape[0]"""
        weight = torch.randn(10, 20)
        result = get_scale_shape(weight, -1)
        self.assertEqual(result, 10)

    def test_group_size_greater_than_last_dim_should_return_weight_dim0(self):
        """测试 group_size 大于最后一维返回 weight.shape[0]"""
        weight = torch.randn(10, 20)
        result = get_scale_shape(weight, 30)
        self.assertEqual(result, 10)

    def test_group_size_divisible_should_return_correct_shape(self):
        """测试 group_size 可整除返回正确形状"""
        weight = torch.randn(10, 20)
        result = get_scale_shape(weight, 5)
        self.assertEqual(result, 10 * (20 // 5))


def create_mock_linear(in_features=8, out_features=4, bias=True):
    """创建模拟的 linear 层"""
    linear = nn.Linear(in_features, out_features, bias=bias)
    # 添加必要的属性
    linear.data_type = "int"
    linear.bits = 4
    linear.sym = True
    linear.group_size = -1
    linear.act_bits = 8
    linear.act_data_type = "int"
    linear.act_sym = True
    linear.act_group_size = -1
    linear.act_dynamic = False
    linear.scale_dtype = torch.float16
    linear.name = "test_layer"
    return linear


class TestWrapperLinear(unittest.TestCase):
    """测试 WrapperLinear 类"""

    @patch('msmodelslim.processor.quant.autoround_utils.wrapper.get_quant_func')
    def test_should_initialize_with_linear(self, mock_get_quant_func):
        """测试使用 linear 层初始化"""
        mock_get_quant_func.return_value = (MagicMock(), "int_sym")

        linear = create_mock_linear()
        wrapper = WrapperLinear(linear)

        self.assertEqual(wrapper.orig_layer, linear)
        self.assertTrue(wrapper.enable_minmax_tuning)
        self.assertTrue(wrapper.enable_round_tuning)

    @patch('msmodelslim.processor.quant.autoround_utils.wrapper.get_quant_func')
    def test_should_disable_minmax_tuning(self, mock_get_quant_func):
        """测试禁用 minmax 调优"""
        mock_get_quant_func.return_value = (MagicMock(), "int_sym")

        linear = create_mock_linear()
        wrapper = WrapperLinear(linear, enable_minmax_tuning=False)

        self.assertFalse(wrapper.enable_minmax_tuning)

    @patch('msmodelslim.processor.quant.autoround_utils.wrapper.get_quant_func')
    def test_should_disable_round_tuning(self, mock_get_quant_func):
        """测试禁用 round 调优"""
        mock_get_quant_func.return_value = (MagicMock(), "int_sym")

        linear = create_mock_linear()
        wrapper = WrapperLinear(linear, enable_round_tuning=False)

        self.assertFalse(wrapper.enable_round_tuning)

    @patch('msmodelslim.processor.quant.autoround_utils.wrapper.get_quant_func')
    def test_should_set_linear_forward(self, mock_get_quant_func):
        """测试设置 linear forward"""
        mock_get_quant_func.return_value = (MagicMock(), "int_sym")

        linear = create_mock_linear()
        wrapper = WrapperLinear(linear)

        self.assertEqual(wrapper.orig_forward, wrapper.linear_forward)

    @patch('msmodelslim.processor.quant.autoround_utils.wrapper.get_quant_func')
    def test_init_params_tunable_should_create_parameter(self, mock_get_quant_func):
        """测试可调参数创建"""
        mock_get_quant_func.return_value = (MagicMock(), "int_sym")

        linear = create_mock_linear()
        wrapper = WrapperLinear(linear)

        # value 应该是可调参数
        self.assertIn("value", wrapper.params)
        self.assertIsInstance(wrapper.params["value"], nn.Parameter)

    @patch('msmodelslim.processor.quant.autoround_utils.wrapper.get_quant_func')
    def test_init_params_not_tunable_should_create_tensor(self, mock_get_quant_func):
        """测试不可调参数创建"""
        mock_get_quant_func.return_value = (MagicMock(), "int_sym")

        linear = create_mock_linear()
        wrapper = WrapperLinear(linear, enable_minmax_tuning=False)

        # min_scale 和 max_scale 应该是普通 tensor
        self.assertIsInstance(wrapper.min_scale, torch.Tensor)
        self.assertNotIsInstance(wrapper.min_scale, nn.Parameter)

    @patch('msmodelslim.processor.quant.autoround_utils.wrapper.get_quant_func')
    def test_unwrapper_should_return_orig_layer(self, mock_get_quant_func):
        """测试 unwrapper 返回原始层"""
        mock_quant_func = MagicMock()
        mock_quant_func.return_value = (torch.randn(4, 8), torch.randn(4, 1), None)
        mock_get_quant_func.return_value = (mock_quant_func, "int_sym")

        linear = create_mock_linear()
        wrapper = WrapperLinear(linear)

        result = wrapper.unwrapper({})
        self.assertEqual(result, linear)

    @patch('msmodelslim.processor.quant.autoround_utils.wrapper.get_quant_func')
    def test_unwrapper_should_set_scale_and_zp(self, mock_get_quant_func):
        """测试 unwrapper 设置 scale 和 zp"""
        mock_quant_func = MagicMock()
        mock_quant_func.return_value = (torch.randn(4, 8), torch.randn(4, 1), torch.zeros(4, 1))
        mock_get_quant_func.return_value = (mock_quant_func, "int_sym")

        linear = create_mock_linear()
        wrapper = WrapperLinear(linear)

        wrapper.unwrapper({})
        self.assertTrue(hasattr(linear, "scale"))
        self.assertTrue(hasattr(linear, "zp"))

    @patch('msmodelslim.processor.quant.autoround_utils.wrapper.get_quant_func')
    def test_unwrapper_with_none_zp_should_set_none(self, mock_get_quant_func):
        """测试 unwrapper zp 为 None"""
        mock_quant_func = MagicMock()
        mock_quant_func.return_value = (torch.randn(4, 8), torch.randn(4, 1), None)
        mock_get_quant_func.return_value = (mock_quant_func, "int_sym")

        linear = create_mock_linear()
        wrapper = WrapperLinear(linear)

        wrapper.unwrapper({})
        self.assertIsNone(linear.zp)

    @patch('msmodelslim.processor.quant.autoround_utils.wrapper.get_quant_func')
    def test_linear_forward_should_return_correct_shape(self, mock_get_quant_func):
        """测试 linear forward 返回正确形状"""
        mock_get_quant_func.return_value = (MagicMock(), "int_sym")

        linear = create_mock_linear()
        wrapper = WrapperLinear(linear)

        x = torch.randn(2, 8)
        weight = torch.randn(4, 8)
        bias = torch.randn(4)
        result = wrapper.linear_forward(x, weight, bias)

        self.assertEqual(result.shape, (2, 4))

    @patch('msmodelslim.processor.quant.autoround_utils.wrapper.get_quant_func')
    def test_forward_should_return_output(self, mock_get_quant_func):
        """测试 forward 返回输出"""
        mock_quant_func = MagicMock()
        mock_quant_func.return_value = (torch.randn(4, 8), torch.randn(4, 1), None)
        mock_get_quant_func.return_value = (mock_quant_func, "int_sym")

        linear = create_mock_linear()
        wrapper = WrapperLinear(linear)

        x = torch.randn(2, 8)
        result = wrapper.forward(x)

        self.assertEqual(result.shape, (2, 4))

    @patch('msmodelslim.processor.quant.autoround_utils.wrapper.get_quant_func')
    def test_should_raise_for_act_bits_greater_than_8(self, mock_get_quant_func):
        """测试 act_bits > 8 时禁用 act quant"""
        mock_get_quant_func.return_value = (MagicMock(), "int_sym")

        linear = create_mock_linear()
        linear.act_bits = 16
        wrapper = WrapperLinear(linear)

        self.assertFalse(wrapper.enable_act_quant)


class TestWrapperLinearAdvanced(unittest.TestCase):
    """测试 WrapperLinear 高级功能"""

    @patch('msmodelslim.processor.quant.autoround_utils.wrapper.get_quant_func')
    def test_unwrapper_with_dict_scale_should_set_attrs(self, mock_get_quant_func):
        """测试 unwrapper 使用 dict scale 设置属性"""
        mock_quant_func = MagicMock()
        mock_quant_func.return_value = (
            torch.randn(4, 8),
            {"scale": torch.randn(4, 1), "other": torch.randn(4, 1)},
            None,
        )
        mock_get_quant_func.return_value = (mock_quant_func, "int_sym")

        linear = create_mock_linear()
        wrapper = WrapperLinear(linear)

        wrapper.unwrapper({})
        self.assertTrue(hasattr(linear, "scale"))
        self.assertTrue(hasattr(linear, "w_other"))

    @patch('msmodelslim.processor.quant.autoround_utils.wrapper.get_quant_func')
    def test_unwrapper_with_dict_zp_should_set_attrs(self, mock_get_quant_func):
        """测试 unwrapper 使用 dict zp 设置属性"""
        mock_quant_func = MagicMock()
        mock_quant_func.return_value = (
            torch.randn(4, 8),
            torch.randn(4, 1),
            {"zp": torch.zeros(4, 1), "other": torch.ones(4, 1)},
        )
        mock_get_quant_func.return_value = (mock_quant_func, "int_sym")

        linear = create_mock_linear()
        wrapper = WrapperLinear(linear)

        wrapper.unwrapper({})
        self.assertTrue(hasattr(linear, "zp"))
        self.assertTrue(hasattr(linear, "w_other"))

    @patch('msmodelslim.processor.quant.autoround_utils.wrapper.get_quant_func')
    def test_unwrapper_with_best_params(self, mock_get_quant_func):
        """测试 unwrapper 使用 best_params"""
        mock_quant_func = MagicMock()
        mock_quant_func.return_value = (torch.randn(4, 8), torch.randn(4, 1), None)
        mock_get_quant_func.return_value = (mock_quant_func, "int_sym")

        linear = create_mock_linear()
        wrapper = WrapperLinear(linear)

        best_params = {
            "value": torch.randn(4, 8),
            "min_scale": torch.tensor(0.5),
            "max_scale": torch.tensor(1.0),
        }
        result = wrapper.unwrapper(best_params)
        self.assertEqual(result, linear)

    @patch('msmodelslim.processor.quant.autoround_utils.wrapper.get_quant_func')
    def test_forward_with_hooks(self, mock_get_quant_func):
        """测试 forward 有 hooks"""
        mock_quant_func = MagicMock()
        mock_quant_func.return_value = (torch.randn(4, 8), torch.randn(4, 1), None)
        mock_get_quant_func.return_value = (mock_quant_func, "int_sym")

        linear = create_mock_linear()
        wrapper = WrapperLinear(linear)

        # 添加 hook
        hook = MagicMock()
        hook.return_value = (torch.randn(2, 8),)
        linear._forward_pre_hooks = {0: hook}

        x = torch.randn(2, 8)
        result = wrapper.forward(x)

        self.assertEqual(result.shape, (2, 4))
        hook.assert_called_once()

    @patch('msmodelslim.processor.quant.autoround_utils.wrapper.get_quant_func')
    def test_forward_with_act_quant(self, mock_get_quant_func):
        """测试 forward 有 act quant"""
        mock_quant_func = MagicMock()
        mock_quant_func.return_value = (torch.randn(4, 8), torch.randn(4, 1), None)
        mock_act_quant_func = MagicMock()
        mock_act_quant_func.return_value = (torch.randn(2, 8), torch.tensor(1.0), None)
        mock_get_quant_func.return_value = (mock_quant_func, "int_sym")

        linear = create_mock_linear()
        linear.act_bits = 8
        wrapper = WrapperLinear(linear)
        wrapper.train_with_act_quant = True
        wrapper.act_quant_func = mock_act_quant_func

        x = torch.randn(2, 8)
        result = wrapper.forward(x)

        self.assertEqual(result.shape, (2, 4))

    @patch('msmodelslim.processor.quant.autoround_utils.wrapper.get_quant_func')
    def test_qdq_weight_with_smooth_scale(self, mock_get_quant_func):
        """测试 _qdq_weight 使用 smooth_scale"""
        mock_quant_func = MagicMock()
        mock_quant_func.return_value = (torch.randn(4, 8), torch.randn(4, 1), None)
        mock_get_quant_func.return_value = (mock_quant_func, "int_sym")

        linear = create_mock_linear()
        wrapper = WrapperLinear(linear)

        smooth_scale = torch.ones(8)
        result = wrapper._qdq_weight(torch.zeros(4, 8), torch.tensor(1.0), torch.tensor(1.0), smooth_scale=smooth_scale)

        self.assertIsNotNone(result)

    @patch('msmodelslim.processor.quant.autoround_utils.wrapper.get_quant_func')
    def test_qdq_weight_without_smooth_scale(self, mock_get_quant_func):
        """测试 _qdq_weight 不使用 smooth_scale"""
        mock_quant_func = MagicMock()
        mock_quant_func.return_value = (torch.randn(4, 8), torch.randn(4, 1), None)
        mock_get_quant_func.return_value = (mock_quant_func, "int_sym")

        linear = create_mock_linear()
        wrapper = WrapperLinear(linear)

        result = wrapper._qdq_weight(torch.zeros(4, 8), torch.tensor(1.0), torch.tensor(1.0))

        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
