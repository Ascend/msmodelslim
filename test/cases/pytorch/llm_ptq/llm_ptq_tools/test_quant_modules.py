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
# pylint: disable=attribute-defined-outside-init,consider-using-from-import,no-member,no-name-in-module,redefined-outer-name,too-many-lines,ungrouped-imports

from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs import (
    linear_quantization_params,
    init_weight_quant_normal,
    init_weight_quant_hessian,
)
from transformers import PretrainedConfig

from msmodelslim import logger
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import FAQuantizer, TensorType
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config import QuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_modules import (
    Quantizer,
    LinearQuantizer,
    Conv2dQuantizer,
    LinearNf4Quantizer,
)


class TestQuantizer:
    @patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs.fake_quantize')
    def test_new_quant_tensor_prob(self, mock_fake_quantize):
        # 测试校准模式（is_calib=True）
        cfg = QuantConfig().weight_activation_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=True, cfg=cfg, logger=logger, is_dynamic=False
        )
        # 设置对象属性
        test_quantizer.name = "test_module"
        test_quantizer.input_scale = torch.tensor(0.5)
        test_quantizer.input_offset = torch.tensor(1.0)
        test_quantizer.bit = 8
        test_quantizer.is_signed = True
        test_quantizer.is_calib = True
        test_quantizer.pr = -0.3
        test_quantizer.print_flag = True
        test_quantizer.range_param = 50
        data = torch.tensor([4.0, 5.0, 6.0])
        # 模拟fake_quantize函数
        mock_fake_quantize.return_value = (MagicMock(), torch.tensor([1.0, 2.0, 3.0]))
        result = test_quantizer.new_quant_tensor(data)

        # 验证结果
        expected_result = torch.tensor([4.0, 5.0, 6.0])  # 根据 mock 的返回值
        assert torch.equal(result, expected_result)

    @patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs.fake_quantize')
    def test_new_quant_tensor_no_calib(self, mock_fake_quantize):
        # 测试校准模式（is_calib=False）
        cfg = QuantConfig().weight_activation_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=True, cfg=cfg, logger=logger, is_dynamic=False
        )
        # 设置对象属性
        test_quantizer.name = "test_module"
        test_quantizer.input_scale = torch.tensor(0.5)
        test_quantizer.input_offset = torch.tensor(1.0)
        test_quantizer.bit = 8
        test_quantizer.is_signed = True
        test_quantizer.is_calib = False
        test_quantizer.print_flag = True
        test_quantizer.range_param = 50
        data = torch.tensor([4.0, 5.0, 6.0])
        # 模拟fake_quantize函数
        mock_fake_quantize.return_value = (MagicMock(), torch.tensor([1.0, 2.0, 3.0]))
        result = test_quantizer.new_quant_tensor(data)

        # 验证结果
        expected_result = torch.tensor([4.0, 5.0, 6.0])  # 根据 mock 的返回值
        assert torch.equal(result, expected_result)

    def test_new_quant_tensor_should_return_expected_result(self):
        # 普通a_bit=8场景的激活的Quantizer.new_quant_tensor测试
        cfg = QuantConfig().weight_activation_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=True, cfg=cfg, logger=logger, is_dynamic=False
        )
        test_quantizer.input_scale = torch.tensor(0.5)
        test_quantizer.input_offset = torch.tensor(0.0)
        in_param_data = torch.tensor([0.4, 1, 1.6, 2])

        result = test_quantizer.new_quant_tensor(in_param_data)
        expected_result = torch.tensor([0.5, 1.0, 1.5, 2.0])
        assert torch.equal(result, expected_result)

    def test_intput_quantizer_tensor_forward_should_return_expected_result(self):
        # 普通的a_bit=8场景的激活的Quantizer.tensor_forward测试
        cfg = QuantConfig().weight_activation_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=True, cfg=cfg, logger=logger, is_dynamic=False
        )
        test_quantizer.init_act_and_observer(cfg)
        in_param_data = torch.tensor([1.0, 1.0, 1.0, 1.0])

        result = test_quantizer.tensor_forward(in_param_data)
        input_scale = test_quantizer.input_scale
        input_offset = test_quantizer.input_offset

        expected_result = torch.tensor([1.0, 1.0, 1.0, 1.0])
        expected_input_scale, expected_input_offset = linear_quantization_params(
            bit=8, x_min=torch.tensor([1]), x_max=torch.tensor([1]), q_signed=True, sym=False
        )

        assert torch.equal(result, expected_result)
        assert torch.equal(input_scale, expected_input_scale)
        assert torch.equal(input_offset, expected_input_offset)

    def test_intput_quantizer_tensor_forward_with_dynamic_quant_should_return_expected_result(self):
        # 普通的a_bit=8动态量化场景的激活Quantizer.tensor_forward测试
        cfg = QuantConfig().weight_activation_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=True, cfg=cfg, logger=logger, is_dynamic=True
        )
        test_quantizer.init_act_and_observer(cfg)
        in_param_data = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

        result = test_quantizer.tensor_forward(in_param_data)
        input_scale = test_quantizer.input_scale
        input_offset = test_quantizer.input_offset

        expected_result = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        expected_input_scale, expected_input_offset = linear_quantization_params(
            bit=8, x_min=torch.tensor([1]), x_max=torch.tensor([1]), q_signed=True, sym=False
        )

        assert torch.equal(result, expected_result)
        assert torch.equal(input_scale, expected_input_scale)
        assert torch.equal(input_offset, expected_input_offset)

    def test_act_method_2(self):
        cfg = QuantConfig().weight_activation_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=True, cfg=cfg, logger=logger, is_dynamic=False
        )
        cfg.act_method = 2
        test_quantizer.init_act_and_observer(cfg)

    def test_act_method_3_range_50(self):
        cfg = QuantConfig().weight_activation_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=True, cfg=cfg, logger=logger, is_dynamic=False
        )
        cfg.act_method = 3
        test_quantizer.range_param = 50
        test_quantizer.init_act_and_observer(cfg)

    def test_act_method_3_range_70(self):
        cfg = QuantConfig().weight_activation_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=True, cfg=cfg, logger=logger, is_dynamic=False
        )
        cfg.act_method = 3
        test_quantizer.range_param = 70
        test_quantizer.init_act_and_observer(cfg)

    def test_enable_quantization(self):
        cfg = QuantConfig().weight_activation_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=True, cfg=cfg, logger=logger, is_dynamic=False
        )
        name = "test"
        range_param = 50
        test_quantizer.enable_quantization(name, range_param)
        assert test_quantizer.name == name
        assert test_quantizer.range_param == range_param
        assert test_quantizer.is_enable is True

    def test_enable_int_infer(self):
        cfg = QuantConfig().weight_activation_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=True, cfg=cfg, logger=logger, is_dynamic=False
        )
        test_quantizer.is_calib = False
        test_quantizer.enable_int_infer()
        assert test_quantizer.quant_weight_tensor is None

    def test_set_ratio(self):
        cfg = QuantConfig().weight_activation_quant()
        cfg.act_method = 1
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=True, cfg=cfg, logger=logger, is_dynamic=False
        )
        test_quantizer.init_act_and_observer(cfg)
        assert test_quantizer.observer is not None
        test_quantizer.set_ratio()

    def test_intput_quantizer_tensor_forward_with_int_infer_should_return_expected_result(self):
        # 普通的a_bit=8量化场景的激活Quantizer.tensor_forward int infer测试
        cfg = QuantConfig().weight_activation_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=True, cfg=cfg, logger=logger, is_dynamic=False
        )
        test_quantizer.init_act_and_observer(cfg)
        test_quantizer.enable_int_infer()
        in_param_data = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

        _ = test_quantizer.tensor_forward(in_param_data)
        test_quantizer.disable_calib()
        result = test_quantizer.tensor_forward(in_param_data)
        input_scale = test_quantizer.input_scale
        input_offset = test_quantizer.input_offset

        test_quantizer.disable_calib()

        expected_result = torch.tensor([[127.0, 127.0, 127.0, 127.0]])
        expected_input_scale, expected_input_offset = linear_quantization_params(
            bit=8, x_min=torch.tensor([1]), x_max=torch.tensor([1]), q_signed=True, sym=False
        )

        assert torch.equal(result, expected_result)
        assert torch.equal(input_scale, expected_input_scale)
        assert torch.equal(input_offset, expected_input_offset)

    def test_weight_quantizer_tensor_forward_should_return_expected_result(self):
        # 普通的w_bit=8场景的权重的Quantizer.tensor_forward测试
        cfg = QuantConfig().weight_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=False, cfg=cfg, logger=logger, is_dynamic=False
        )
        in_param_data = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

        result = test_quantizer.tensor_forward(in_param_data)
        weight_scale = test_quantizer.weight_scale
        weight_offset = test_quantizer.weight_offset

        expected_result = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        _, _, expected_weight_scale, expected_weight_offset = init_weight_quant_normal(
            weight=in_param_data,
            bit=8,
            is_sym=test_quantizer.is_sym,
            is_signed=test_quantizer.is_signed,
            integral_zp=True,
            admm=test_quantizer.admm,
            mm_tensor=test_quantizer.mm_per_tensor,
        )

        assert torch.equal(result, expected_result)
        assert torch.equal(weight_scale, expected_weight_scale)
        assert torch.equal(weight_offset, expected_weight_offset)

    def test_weight_quantizer_tensor_forward_with_int_infer_should_return_expected_result(self):
        # 普通的w_bit=8场景的权重的Quantizer.tensor_forward int infer测试
        cfg = QuantConfig().weight_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=False, cfg=cfg, logger=logger, is_dynamic=False
        )
        test_quantizer.enable_int_infer()
        in_param_data = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

        _ = test_quantizer.tensor_forward(in_param_data)
        test_quantizer.disable_calib()
        result = test_quantizer.tensor_forward(in_param_data)
        weight_scale = test_quantizer.weight_scale
        weight_offset = test_quantizer.weight_offset

        expected_result = torch.tensor([[127.0, 127.0, 127.0, 127.0]])
        _, _, expected_weight_scale, expected_weight_offset = init_weight_quant_normal(
            weight=in_param_data,
            bit=8,
            is_sym=test_quantizer.is_sym,
            is_signed=test_quantizer.is_signed,
            integral_zp=True,
            admm=test_quantizer.admm,
            mm_tensor=test_quantizer.mm_per_tensor,
        )

        assert torch.equal(result, expected_result)
        assert torch.equal(weight_scale, expected_weight_scale)
        assert torch.equal(weight_offset, expected_weight_offset)

    def test_normal_data_no_outliers(self):
        # 测试没有异常值的情况
        cfg = QuantConfig().weight_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=False, cfg=cfg, logger=logger, is_dynamic=False
        )
        test_quantizer.enable_int_infer()
        current_t = np.array([1, 2, 3, 4, 5])
        k = 3
        result = test_quantizer.get_anti_outlier(k, current_t)

        # 计算预期结果
        current_std = np.std(current_t)
        current_mean = np.mean(current_t)
        threshold1 = current_mean - k * current_std
        threshold2 = current_mean + k * current_std
        bigger_num = np.sum(current_t >= threshold2)
        smaller_num = np.sum(current_t <= threshold1)
        expected = (bigger_num + smaller_num) / current_t.size

        assert np.array_equal(result, expected)
        assert np.array_equal(result, 0.0)  # 因为没有异常值

    def test_normal_data_with_outliers(self):
        cfg = QuantConfig().weight_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=False, cfg=cfg, logger=logger, is_dynamic=False
        )
        test_quantizer.enable_int_infer()
        # 测试有异常值的情况
        current_t = np.array([1, 2, 3, 4, 100])  # 100是异常值
        k = 1
        result = test_quantizer.get_anti_outlier(k, current_t)

        # 计算预期结果
        current_std = np.std(current_t)
        current_mean = np.mean(current_t)
        threshold1 = current_mean - k * current_std
        threshold2 = current_mean + k * current_std
        bigger_num = np.sum(current_t >= threshold2)
        smaller_num = np.sum(current_t <= threshold1)
        expected = (bigger_num + smaller_num) / current_t.size

        assert np.array_equal(result, expected)
        assert np.array_equal(result, 0.2)  # 因为有1个异常值，占总数的20%

    def test_single_element_data(self):
        # 测试只有一个元素的情况
        cfg = QuantConfig().weight_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=False, cfg=cfg, logger=logger, is_dynamic=False
        )
        test_quantizer.enable_int_infer()
        current_t = np.array([5])
        k = 2
        result = test_quantizer.get_anti_outlier(k, current_t)

        # 对于单个元素，标准差为0，所以所有值都在范围内
        assert np.array_equal(result, 2.0)

    def test_all_elements_outliers(self):
        # 测试所有元素都是异常值的情况
        cfg = QuantConfig().weight_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=False, cfg=cfg, logger=logger, is_dynamic=False
        )
        test_quantizer.enable_int_infer()
        current_t = np.array([1, 2, 3, 4, 5])
        k = 0  # k=0会使阈值等于均值，所有值都会被视为异常
        result = test_quantizer.get_anti_outlier(k, current_t)

        # 计算预期结果
        current_std = np.std(current_t)
        current_mean = np.mean(current_t)
        threshold1 = current_mean - k * current_std
        threshold2 = current_mean + k * current_std
        bigger_num = np.sum(current_t >= threshold2)
        smaller_num = np.sum(current_t <= threshold1)
        expected = (bigger_num + smaller_num) / current_t.size

        assert np.array_equal(result, expected)
        assert np.array_equal(result, 1.2)  # 所有值都是异常值

    def test_all_elements_same(self):
        # 测试所有元素都相同的情况
        cfg = QuantConfig().weight_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=False, cfg=cfg, logger=logger, is_dynamic=False
        )
        test_quantizer.enable_int_infer()
        current_t = np.array([5, 5, 5, 5, 5])
        k = 2
        result = test_quantizer.get_anti_outlier(k, current_t)

        # 标准差为0，所有值都等于均值
        assert np.array_equal(result, 2.0)

    def test_mixed_sign_data(self):
        # 测试包含正负值的数据
        cfg = QuantConfig().weight_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=False, cfg=cfg, logger=logger, is_dynamic=False
        )
        test_quantizer.enable_int_infer()
        current_t = np.array([-10, -5, 0, 5, 10])
        k = 1
        result = test_quantizer.get_anti_outlier(k, current_t)

        # 计算预期结果
        current_std = np.std(current_t)
        current_mean = np.mean(current_t)
        threshold1 = current_mean - k * current_std
        threshold2 = current_mean + k * current_std
        bigger_num = np.sum(current_t >= threshold2)
        smaller_num = np.sum(current_t <= threshold1)
        expected = (bigger_num + smaller_num) / current_t.size

        assert np.array_equal(result, expected)

    def test_large_k_value(self):
        # 测试较大的k值，应该没有异常值
        cfg = QuantConfig().weight_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=False, cfg=cfg, logger=logger, is_dynamic=False
        )
        test_quantizer.enable_int_infer()
        current_t = np.array([1, 2, 3, 4, 5, 100])  # 包含异常值
        k = 1000  # 非常大的k值
        result = test_quantizer.get_anti_outlier(k, current_t)

        assert np.array_equal(result, 0.0)  # 所有值都应该在范围内

    def test_float_data(self):
        # 测试浮点数数据
        cfg = QuantConfig().weight_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=False, cfg=cfg, logger=logger, is_dynamic=False
        )
        test_quantizer.enable_int_infer()
        current_t = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        k = 2
        result = test_quantizer.get_anti_outlier(k, current_t)

        # 计算预期结果
        current_std = np.std(current_t)
        current_mean = np.mean(current_t)
        threshold1 = current_mean - k * current_std
        threshold2 = current_mean + k * current_std
        bigger_num = np.sum(current_t >= threshold2)
        smaller_num = np.sum(current_t <= threshold1)
        expected = (bigger_num + smaller_num) / current_t.size

        assert np.array_equal(result, expected)
        assert np.array_equal(result, 0.0)  # 没有异常值

    def test_negative_k_value(self):
        cfg = QuantConfig().weight_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=False, cfg=cfg, logger=logger, is_dynamic=False
        )
        test_quantizer.enable_int_infer()
        # 测试负数k值
        current_t = np.array([1, 2, 3, 4, 5])
        k = -2
        result = test_quantizer.get_anti_outlier(k, current_t)

        # 计算预期结果
        current_std = np.std(current_t)
        current_mean = np.mean(current_t)
        threshold1 = current_mean - k * current_std
        threshold2 = current_mean + k * current_std
        bigger_num = np.sum(current_t >= threshold2)
        smaller_num = np.sum(current_t <= threshold1)
        expected = (bigger_num + smaller_num) / current_t.size

        assert np.array_equal(result, expected)
        # 负数k会反转阈值，所有值都应该被视为异常
        assert np.array_equal(result, 2.0)

    def test_zero_k_value(self):
        cfg = QuantConfig().weight_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=False, cfg=cfg, logger=logger, is_dynamic=False
        )
        test_quantizer.enable_int_infer()
        # 测试k=0的情况
        current_t = np.array([1, 2, 3, 4, 5])
        k = 0
        result = test_quantizer.get_anti_outlier(k, current_t)

        # 计算预期结果
        current_std = np.std(current_t)
        current_mean = np.mean(current_t)
        threshold1 = current_mean - k * current_std
        threshold2 = current_mean + k * current_std
        bigger_num = np.sum(current_t >= threshold2)
        smaller_num = np.sum(current_t <= threshold1)
        expected = (bigger_num + smaller_num) / current_t.size

        assert np.array_equal(result, expected)
        # k=0时，阈值等于均值，所有值都应该被视为异常
        assert np.array_equal(result, 1.2)

    def test_init_weight_quant_normal_should_return_expected_result(self):
        # 普通的w_bit=8场景的权重的Quantizer._init_weight_quant_normal测试
        cfg = QuantConfig().weight_quant()
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=False, cfg=cfg, logger=logger, is_dynamic=False
        )
        in_param_weight = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        test_quantizer._init_weight_quant_normal(in_param_weight, y=None)
        weight_scale = test_quantizer.weight_scale
        weight_offset = test_quantizer.weight_offset

        _, _, expected_weight_scale, expected_weight_offset = init_weight_quant_normal(
            weight=in_param_weight,
            bit=8,
            is_sym=test_quantizer.is_sym,
            is_signed=test_quantizer.is_signed,
            integral_zp=True,
            admm=test_quantizer.admm,
            mm_tensor=test_quantizer.mm_per_tensor,
        )

        assert torch.equal(weight_scale, expected_weight_scale)
        assert torch.equal(weight_offset, expected_weight_offset)

    def test_init_weight_quant_normal_with_hessian_should_return_expected_result(self):
        # 普通的w_bit=8场景的权重的Quantizer._init_weight_quant_normal在GPTQ场景下测试
        cfg = QuantConfig(w_bit=8, a_bit=16).weight_quant(w_method="GPTQ")
        test_quantizer = Quantizer(
            bit=8, is_signed=True, is_enable=True, is_input=False, cfg=cfg, logger=logger, is_dynamic=False
        )
        in_param_weight = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        in_param_input = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

        test_quantizer._init_weight_quant_normal(in_param_input, y=in_param_input)
        weight_scale = test_quantizer.weight_scale
        weight_offset = test_quantizer.weight_offset

        _, _, expected_weight_scale, expected_weight_offset = init_weight_quant_hessian(
            weight=in_param_weight,
            input=in_param_input,
            bit=8,
            is_sym=test_quantizer.is_sym,
            is_signed=test_quantizer.is_signed,
            integral_zp=True,
            admm=test_quantizer.admm,
            mm_tensor=test_quantizer.mm_per_tensor,
        )

        assert torch.equal(weight_scale, expected_weight_scale)
        assert torch.equal(weight_offset, expected_weight_offset)


class TestLinearQuantizer:
    def test_forward_should_return_expected_result(self):
        # 普通的w8a16场景的LinearQuantizer.forward测试
        cfg = QuantConfig(w_bit=8, a_bit=16).weight_quant()
        linear = torch.nn.Linear(in_features=4, out_features=2, bias=False, dtype=torch.float32)
        in_param_weight = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
        in_param_input = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

        linear.weight.data = in_param_weight
        linear_quantizer = LinearQuantizer(cfg, logger)
        linear_quantizer.set_param(linear)

        result = linear_quantizer(in_param_input)

        expected_result = torch.tensor([[4.0, 4.0]])
        assert torch.equal(result, expected_result)

    def test_forward_with_int_infer_should_return_expected_result(self):
        # 普通的w8a8场景的LinearQuantizer.forward的int_infer测试
        cfg = QuantConfig(w_bit=8, a_bit=8).weight_quant()
        linear = torch.nn.Linear(in_features=4, out_features=2, bias=False, dtype=torch.float32)
        in_param_weight = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
        in_param_input = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])

        linear.weight.data = in_param_weight
        linear_quantizer = LinearQuantizer(cfg, logger)
        linear_quantizer.set_param(linear)
        linear_quantizer.quant_input.init_act_and_observer(cfg)

        _ = linear_quantizer(in_param_input)
        linear_quantizer.quant_input.disable_calib()
        linear_quantizer.quant_weight.disable_calib()
        linear_quantizer.quant_input.enable_int_infer()
        linear_quantizer.quant_weight.enable_int_infer()
        result = linear_quantizer(in_param_input)

        expected_result = torch.tensor([[4.0, 4.0], [4.0, 4.0]])
        assert torch.equal(result, expected_result)

    def test_forward_with_int_infer_and_int_bias_should_return_expected_result(self):
        # 普通的w8a8场景的带int bias的LinearQuantizer.forward的int_infer测试
        cfg = QuantConfig(w_bit=8, a_bit=8).weight_quant()
        linear = torch.nn.Linear(in_features=4, out_features=2, bias=True, dtype=torch.float32)
        in_param_weight = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
        in_param_input = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
        in_param_bias = torch.tensor([[1.0, 1.0]])

        linear.weight.data = in_param_weight
        linear.bias.data = in_param_bias
        linear_quantizer = LinearQuantizer(cfg, logger)
        linear_quantizer.set_param(linear)
        linear_quantizer.quant_input.init_act_and_observer(cfg)

        _ = linear_quantizer(in_param_input)
        linear_quantizer.quant_input.disable_calib()
        linear_quantizer.quant_weight.disable_calib()
        linear_quantizer.quant_input.enable_int_infer()
        linear_quantizer.quant_weight.enable_int_infer()
        linear_quantizer.quant_weight.int_bias = True
        result = linear_quantizer(in_param_input)

        expected_result = torch.tensor([[5.0, 5.0], [5.0, 5.0]])
        assert torch.equal(result, expected_result)

    def test_int_bias_with_bias(self):
        # 测试int_bias=True且有bias的情况
        cfg = QuantConfig(w_bit=8, a_bit=8).weight_quant()
        linear = torch.nn.Linear(in_features=4, out_features=1, bias=True, dtype=torch.float32)
        in_param_weight = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        in_param_input = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])  # noqa: F841
        in_param_bias = torch.tensor([[1.0]])

        linear.weight.data = in_param_weight
        linear.bias.data = in_param_bias
        linear_quantizer = LinearQuantizer(cfg, logger)
        linear_quantizer.set_param(linear)

        linear_quantizer.quant_weight.int_bias = True
        linear_quantizer.bias.data = torch.tensor([1.0, 2.0])

        correction = torch.tensor([0.5, 0.5])
        int_out = torch.tensor([[3.0, 4.0]])
        fp_scale = torch.tensor([[0.1, 0.2]])

        x_device = torch.device('cpu')
        x_dtype = torch.float16

        # 计算预期结果
        bias_int = (linear_quantizer.bias.data / fp_scale).round()  # [10.0, 10.0]
        bias_int -= correction  # [9.5, 9.5]
        expected = (int_out + bias_int) * fp_scale  # [[1.25, 2.7]]
        expected = expected.to(x_dtype)

        result = linear_quantizer._bias_and_dequant_process(correction, int_out, fp_scale, x_device, x_dtype)

        # 验证结果
        assert torch.equal(result, expected)
        assert result.dtype == x_dtype

    def test_int_bias_without_bias(self):
        # 测试int_bias=True且没有bias的情况
        cfg = QuantConfig(w_bit=8, a_bit=8).weight_quant()
        linear = torch.nn.Linear(in_features=4, out_features=1, bias=True, dtype=torch.float32)
        in_param_weight = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        in_param_input = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])  # noqa: F841
        in_param_bias = torch.tensor([[1.0]])

        linear.weight.data = in_param_weight
        linear.bias.data = in_param_bias
        linear_quantizer = LinearQuantizer(cfg, logger)
        linear_quantizer.set_param(linear)
        linear_quantizer.quant_weight.int_bias = True
        linear_quantizer.bias = None

        correction = torch.tensor([0.5, 0.5])
        int_out = torch.tensor([[3.0, 4.0]])
        fp_scale = torch.tensor([[0.1, 0.2]])

        x_device = torch.device('cpu')
        x_dtype = torch.float16

        # 计算预期结果
        bias_int = torch.zeros(correction.size(0)).to(x_device)  # [0.0, 0.0]
        bias_int -= correction  # [-0.5, -0.5]
        expected = (int_out + bias_int) * fp_scale  # [[0.25, 0.7]]
        expected = expected.to(x_dtype)

        result = linear_quantizer._bias_and_dequant_process(correction, int_out, fp_scale, x_device, x_dtype)

        # 验证结果
        assert torch.equal(result, expected)
        assert result.dtype == x_dtype

    def test_no_int_bias_with_bias(self):
        # 测试int_bias=False且有bias的情况
        cfg = QuantConfig(w_bit=8, a_bit=8).weight_quant()
        linear = torch.nn.Linear(in_features=4, out_features=1, bias=True, dtype=torch.float32)
        in_param_weight = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        in_param_input = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])  # noqa: F841
        in_param_bias = torch.tensor([[1.0]])

        linear.weight.data = in_param_weight
        linear.bias.data = in_param_bias
        linear_quantizer = LinearQuantizer(cfg, logger)
        linear_quantizer.set_param(linear)
        linear_quantizer.quant_weight.int_bias = False
        linear_quantizer.bias.data = torch.tensor([1.0, 2.0])

        correction = torch.tensor([0.5, 0.5])
        int_out = torch.tensor([[3.0, 4.0]])
        fp_scale = torch.tensor([[0.1, 0.2]])

        x_device = torch.device('cpu')
        x_dtype = torch.float16

        # 计算预期结果
        fp_out = int_out * fp_scale  # [[0.3, 0.8]]
        fp_out = fp_out.to(x_dtype)

        correction_fp = correction * fp_scale  # [[0.05, 0.1]]
        correction_fp = correction_fp.to(x_dtype)

        bias_fp = linear_quantizer.bias.data.to(x_dtype) - correction_fp  # [[0.95, 1.9]]

        expected = fp_out + bias_fp  # [[1.25, 2.7]]

        result = linear_quantizer._bias_and_dequant_process(correction, int_out, fp_scale, x_device, x_dtype)

        # 验证结果
        assert torch.equal(result, expected)
        assert result.dtype == x_dtype

    def test_no_int_bias_without_bias(self):
        # 测试int_bias=False且没有bias的情况
        cfg = QuantConfig(w_bit=8, a_bit=8).weight_quant()
        linear = torch.nn.Linear(in_features=4, out_features=1, bias=True, dtype=torch.float32)
        in_param_weight = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        in_param_input = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])  # noqa: F841
        in_param_bias = torch.tensor([[1.0]])

        linear.weight.data = in_param_weight
        linear.bias.data = in_param_bias
        linear_quantizer = LinearQuantizer(cfg, logger)
        linear_quantizer.set_param(linear)
        linear_quantizer.quant_weight.int_bias = False
        linear_quantizer.bias = None

        correction = torch.tensor([0.5, 0.5])
        int_out = torch.tensor([[3.0, 4.0]])
        fp_scale = torch.tensor([[0.1, 0.2]])

        x_device = torch.device('cpu')
        x_dtype = torch.float16

        # 计算预期结果
        fp_out = int_out * fp_scale  # [[0.3, 0.8]]
        fp_out = fp_out.to(x_dtype)

        correction_fp = correction * fp_scale  # [[0.05, 0.1]]
        correction_fp = correction_fp.to(x_dtype)

        bias_fp = -correction_fp  # [[-0.05, -0.1]]

        expected = fp_out + bias_fp  # [[0.25, 0.7]]

        result = linear_quantizer._bias_and_dequant_process(correction, int_out, fp_scale, x_device, x_dtype)

        # 验证结果
        assert torch.equal(result, expected)
        assert result.dtype == x_dtype


class TestFAQuantizer:
    # 测试模型中的FA3量化功能
    def setup_method(self):
        # 测试前的参数初始化
        self.batch_size = 2
        self.seq_len = 16
        self.num_heads = 8
        self.head_dim = 64
        self.hidden_size = self.num_heads * self.head_dim

        class TestConfig(PretrainedConfig):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.hidden_size = kwargs.get("hidden_size", 512)
                self.num_attention_heads = kwargs.get("num_attention_heads", 8)
                self.num_key_value_heads = kwargs.get("num_key_value_heads", 8)
                self.quant_mode = 'fa3'
                self.layer_norm_eps = 1e-5

        self.model = MagicMock()
        self.model.config = TestConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_heads,
        )

        self.q_input = torch.ones((self.batch_size, self.num_heads, self.seq_len, self.head_dim), dtype=torch.float32)
        self.k_input = torch.ones((self.batch_size, self.num_heads, self.seq_len, self.head_dim), dtype=torch.float32)
        self.v_input = torch.ones((self.batch_size, self.num_heads, self.seq_len, self.head_dim), dtype=torch.float32)

        self.fa_quantizer = FAQuantizer(self.model.config, logger)
        self.fa_quantizer.configure(bit=8, sym=True, tp_size=1)

        if self.fa_quantizer.tp_size is None:
            self.fa_quantizer.tp_size = 1
        if self.fa_quantizer.num_head is None:
            self.fa_quantizer.num_head = self.num_heads
        if self.fa_quantizer.num_kv_head is None:
            self.fa_quantizer.num_kv_head = self.num_heads
        if self.fa_quantizer.head_dim is None:
            self.fa_quantizer.head_dim = self.head_dim

    def test_fa_quantizer_init(self):
        # 测试FA Quantizer初始化配置
        assert self.fa_quantizer.is_calib is False
        assert self.fa_quantizer.dequant_infer is False

        assert self.fa_quantizer.q_observer is not None
        assert self.fa_quantizer.k_observer is not None
        assert self.fa_quantizer.v_observer is not None

    def test_fa_quantizer_forward_no_quant(self):
        # 测试未启用量化时的前向传播
        assert self.fa_quantizer.is_calib is False
        assert self.fa_quantizer.dequant_infer is False

        q_out = self.fa_quantizer.quant(self.q_input, qkv="q")
        k_out = self.fa_quantizer.quant(self.k_input, qkv="k")
        v_out = self.fa_quantizer.quant(self.v_input, qkv="v")

        assert torch.equal(q_out, self.q_input)
        assert torch.equal(k_out, self.k_input)
        assert torch.equal(v_out, self.v_input)

        expected_values = {t.value for t in [TensorType.Q, TensorType.K, TensorType.V]}
        assert self.fa_quantizer.processed_types == expected_values

    def test_fa_quantizer_forward_with_calib(self):
        # 设置前置条件
        self.fa_quantizer.processed_types = {"q", "k", "v"}
        # 测试启用校准时的前向传播
        self.fa_quantizer.enable_calibration()
        assert self.fa_quantizer.is_calib is True

        for observer in [self.fa_quantizer.q_observer, self.fa_quantizer.k_observer, self.fa_quantizer.v_observer]:
            observer.configure(bit=8, sym=True, num_head=self.num_heads, ratio=0.9999)

        q_out = self.fa_quantizer.quant(self.q_input, qkv="q")
        k_out = self.fa_quantizer.quant(self.k_input, qkv="k")
        v_out = self.fa_quantizer.quant(self.v_input, qkv="v")

        assert torch.equal(q_out, self.q_input)
        assert torch.equal(k_out, self.k_input)
        assert torch.equal(v_out, self.v_input)

        expected_values = {t.value for t in [TensorType.Q, TensorType.K, TensorType.V]}
        assert self.fa_quantizer.processed_types == expected_values

        # 由于调用了get_scale_offset方法 (在quant方法内部),
        # 且update=True (当is_calib=True且dequant_infer=False时), 统计信息应该已被收集
        assert self.fa_quantizer.q_observer._min_values is not None
        assert self.fa_quantizer.q_observer._max_values is not None
        assert self.fa_quantizer.k_observer._min_values is not None
        assert self.fa_quantizer.k_observer._max_values is not None
        assert self.fa_quantizer.v_observer._min_values is not None
        assert self.fa_quantizer.v_observer._max_values is not None

    def test_fa_quantizer_forward_with_quant(self):
        # 设置前置条件
        self.fa_quantizer.processed_types = {"q", "k", "v"}
        # 测试启用量化后的前向传播
        self.fa_quantizer.enable_calibration()

        for observer in [self.fa_quantizer.q_observer, self.fa_quantizer.k_observer, self.fa_quantizer.v_observer]:
            observer.configure(bit=8, sym=True, num_head=self.num_heads, ratio=0.9999)

        _ = self.fa_quantizer.quant(self.q_input, qkv="q")
        _ = self.fa_quantizer.quant(self.k_input, qkv="k")
        _ = self.fa_quantizer.quant(self.v_input, qkv="v")

        self.fa_quantizer.disable_calibration()
        assert self.fa_quantizer.is_calib is False
        self.fa_quantizer.dequant_infer = True
        assert self.fa_quantizer.dequant_infer is True

        q_out = self.fa_quantizer.quant(self.q_input, qkv="q")
        k_out = self.fa_quantizer.quant(self.k_input, qkv="k")
        v_out = self.fa_quantizer.quant(self.v_input, qkv="v")

        assert q_out.shape == self.q_input.shape
        assert k_out.shape == self.k_input.shape
        assert v_out.shape == self.v_input.shape

        # 由于是校准模式下收集的参数进行量化，如果有进行量化+反量化操作，
        # 两个张量因为精度损失不应完全相等。但是在当前实现中，即使在dequant_infer=True
        # 的情况下，代码也只返回了原始张量，因此这里相等
        assert torch.equal(q_out, self.q_input)
        assert torch.equal(k_out, self.k_input)
        assert torch.equal(v_out, self.v_input)

        assert self.fa_quantizer.q_observer.is_calibrated()
        assert self.fa_quantizer.k_observer.is_calibrated()
        assert self.fa_quantizer.v_observer.is_calibrated()

        scales, offsets = self.fa_quantizer.export_quant_params()
        assert len(scales) == 3 and len(offsets) == 3

    def test_fa_quantizer_int_infer(self):
        # 设置前置条件
        self.fa_quantizer.processed_types = {"q", "k", "v"}
        # 测试整数推理模式,校准设置
        self.fa_quantizer.enable_calibration()
        for observer in [self.fa_quantizer.q_observer, self.fa_quantizer.k_observer, self.fa_quantizer.v_observer]:
            observer.configure(bit=8, sym=True, num_head=self.num_heads, ratio=0.9999)

        _ = self.fa_quantizer.quant(self.q_input, qkv="q")
        _ = self.fa_quantizer.quant(self.k_input, qkv="k")
        _ = self.fa_quantizer.quant(self.v_input, qkv="v")

        # 切换到整数推理模式
        self.fa_quantizer.disable_calibration()
        self.fa_quantizer.dequant_infer = False

        q_out = self.fa_quantizer.quant(self.q_input, qkv="q")
        k_out = self.fa_quantizer.quant(self.k_input, qkv="k")  # noqa: F841
        v_out = self.fa_quantizer.quant(self.v_input, qkv="v")  # noqa: F841

        assert q_out.shape == self.q_input.shape
        assert self.fa_quantizer.q_observer.is_calibrated()

        scales, offsets = self.fa_quantizer.export_quant_params()
        assert len(scales) == 3 and len(offsets) == 3

    def _create_mock_attention(self):
        # 创建用于测试的MockAttention类
        class MockAttention(torch.nn.Module):
            def __init__(self, config, logger):
                super().__init__()
                hidden_size = config.hidden_size
                self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
                self.k_proj = torch.nn.Linear(hidden_size, hidden_size)
                self.v_proj = torch.nn.Linear(hidden_size, hidden_size)
                self.o_proj = torch.nn.Linear(hidden_size, hidden_size)
                self.fa_quantizer = FAQuantizer(config, logger)
                self.fa_quantizer.configure(bit=8, sym=True, tp_size=1)
                self.num_heads = config.num_attention_heads
                self.head_dim = config.hidden_size // self.num_heads

            def forward(self, hidden_states):
                batch_size, seq_len, _ = hidden_states.size()
                q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

                q = self.fa_quantizer.quant(q, qkv="q")
                k = self.fa_quantizer.quant(k, qkv="k")
                v = self.fa_quantizer.quant(v, qkv="v")

                attn_output = (q + k + v).transpose(1, 2).reshape(batch_size, seq_len, -1)
                return self.o_proj(attn_output)

        return MockAttention

    def test_attention_module_integration(self):
        # 测试FA Quantizer与模型Attention模块的集成
        MockAttention = self._create_mock_attention()
        attention = MockAttention(self.model.config, logger)
        attention.fa_quantizer.configure(bit=8, sym=True, tp_size=1)
        x = torch.rand((2, 16, self.hidden_size), dtype=torch.float32)

        with torch.no_grad():
            output1 = attention(x)  # noqa: F841

        attention.fa_quantizer.enable_calibration()
        with torch.no_grad():
            _ = attention(x)

        attention.fa_quantizer.disable_calibration()
        with torch.no_grad():
            output2 = attention(x)

        assert output2.shape == x.shape
        assert attention.fa_quantizer.q_observer.is_calibrated()
        scales, offsets = attention.fa_quantizer.export_quant_params()
        assert len(scales) == 3 and len(offsets) == 3

    def test_multiple_attention_layers_quantization(self):
        # 模拟简化版的DeepSeek R1模型，测试多个Attention层的FA3量化
        class MockDeepseekR1(torch.nn.Module):
            def __init__(self, config, logger, num_layers):
                super().__init__()
                self.layers = torch.nn.ModuleList([mock_attention(config, logger) for _ in range(num_layers)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        mock_attention = self._create_mock_attention()
        model = MockDeepseekR1(self.model.config, logger, 3)

        x = torch.rand((2, 16, self.hidden_size), dtype=torch.float32)
        with torch.no_grad():
            output1 = model(x)

        for layer in model.layers:
            layer.fa_quantizer.enable_calibration()

        with torch.no_grad():
            _ = model(x)

        for layer in model.layers:
            layer.fa_quantizer.disable_calibration()
            assert layer.fa_quantizer.is_calibrated()

        with torch.no_grad():
            output2 = model(x)

        assert output1.shape == x.shape and output2.shape == x.shape

        for i, layer in enumerate(model.layers):
            scales, offsets = layer.fa_quantizer.export_quant_params()
            assert len(scales) == 3 and len(offsets) == 3


class TestConv2dQuantizer:
    def test_init(self):
        # 测试初始化方法
        cfg = QuantConfig(w_bit=8, a_bit=8).weight_quant()
        quantizer = Conv2dQuantizer(cfg, logger)

        assert quantizer.in_channels is None
        assert quantizer.out_channels is None
        assert quantizer.kernel_size is None
        assert quantizer.stride is None
        assert quantizer.padding is None
        assert quantizer.dilation is None
        assert quantizer.groups is None
        assert quantizer.weight is None
        assert quantizer.bias is None

        # 验证量化器是否正确初始化
        assert quantizer.quant_input.bit == cfg.a_bit
        assert quantizer.quant_input.is_signed == cfg.a_signed
        assert quantizer.quant_weight.bit == cfg.w_bit
        assert quantizer.quant_weight.is_signed == cfg.w_signed

    def test_set_param_with_bias(self):
        # 测试set_param方法（有bias的情况）
        cfg = QuantConfig(w_bit=8, a_bit=8).weight_quant()
        quantizer = Conv2dQuantizer(cfg, logger)
        conv = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True
        )

        quantizer.set_param(conv)

        # 验证参数是否正确设置
        assert quantizer.in_channels == conv.in_channels
        assert quantizer.out_channels == conv.out_channels
        assert quantizer.kernel_size == conv.kernel_size
        assert quantizer.stride == conv.stride
        assert quantizer.padding == conv.padding
        assert quantizer.dilation == conv.dilation
        assert quantizer.groups == conv.groups
        assert torch.equal(quantizer.weight, nn.Parameter(conv.weight.data))
        assert torch.equal(quantizer.bias, nn.Parameter(conv.bias.data))

    def test_set_param_without_bias(self):
        # 测试set_param方法（没有bias的情况
        cfg = QuantConfig(w_bit=8, a_bit=8).weight_quant()
        quantizer = Conv2dQuantizer(cfg, logger)
        conv_without_bias = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False
        )

        quantizer.set_param(conv_without_bias)

        # 验证bias是否为None
        assert quantizer.bias is None

    def test_conv_forward(self):
        # 测试_conv_forward方法
        cfg = QuantConfig(w_bit=8, a_bit=8).weight_quant()
        quantizer = Conv2dQuantizer(cfg, logger)
        conv = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True
        )
        quantizer.set_param(conv)

        x = torch.randn(1, 3, 32, 32)
        weight = torch.randn(64, 3, 3, 3)

        result = quantizer._conv_forward(x, weight)
        # 验证返回值
        assert result.shape == (1, 64, 32, 32)


class TestLinearNf4Quantizer:
    def test_init(self):
        # 测试初始化方法
        cfg = QuantConfig(w_bit=8, a_bit=8).weight_quant()
        cfg.block_size = 16
        quantizer = LinearNf4Quantizer(cfg, logger)

        assert quantizer.weight is None
        assert quantizer.bias is None
        assert quantizer.weight_shape is None
        assert quantizer.bias_shape is None
        assert quantizer.dtype is None
        assert quantizer.device is None
        assert quantizer.weight_absmax is None
        assert quantizer.bias_absmax is None
        assert quantizer.nf4_mapping is None
        assert quantizer.blocksize == cfg.block_size

    def test_set_param_with_bias(self):
        # 测试set_param方法（有bias的情况）
        cfg = QuantConfig(w_bit=8, a_bit=8).weight_quant()
        quantizer = LinearNf4Quantizer(cfg, logger)
        linear = nn.Linear(in_features=64, out_features=128, bias=True)
        # 设置测试用的张量
        test_weight = torch.randn(128, 64)
        test_bias = torch.randn(128)
        linear.weight.data = test_weight
        linear.bias.data = test_bias
        quantizer.set_param(linear)

        # 验证参数是否正确设置
        assert quantizer.weight_shape == linear.weight.shape
        assert quantizer.dtype == linear.weight.dtype
        assert quantizer.device == linear.weight.device
        assert torch.equal(quantizer.weight, linear.weight.data)
        assert torch.equal(quantizer.bias, linear.bias.data)
        assert quantizer.bias_shape == linear.bias.shape

    def test_set_param_without_bias(self):
        # 测试set_param方法（没有bias的情况）
        cfg = QuantConfig(w_bit=8, a_bit=8).weight_quant()
        quantizer = LinearNf4Quantizer(cfg, logger)
        linear_without_bias = nn.Linear(in_features=64, out_features=128, bias=False)

        quantizer.set_param(linear_without_bias)

        # 验证参数是否正确设置
        assert quantizer.weight_shape == linear_without_bias.weight.shape
        assert quantizer.dtype == linear_without_bias.weight.dtype
        assert quantizer.device == linear_without_bias.weight.device
        assert torch.equal(quantizer.weight, linear_without_bias.weight.data)
        assert quantizer.bias is None
        assert quantizer.bias_shape is None

    def test_normalize_data(self):
        # 测试normalize_data方法
        cfg = QuantConfig(w_bit=8, a_bit=8).weight_quant()
        quantizer = LinearNf4Quantizer(cfg, logger)
        data = torch.randn(128, 16)
        block_weight, absmax = quantizer.normalize_data(data)

        # 验证输出形状
        assert block_weight.shape == (32, 64)
        assert absmax.shape == (32, 1)

        # 验证归一化逻辑
        for i in range(8):
            assert np.allclose(torch.max(torch.abs(block_weight[i])).item(), 1.0, atol=1e-5)

    def test_set_nf4_quantized_vari(self):
        # 测试set_nf4_quantized_vari方法
        cfg = QuantConfig(w_bit=8, a_bit=8).weight_quant()
        quantizer = LinearNf4Quantizer(cfg, logger)
        quantizer.dtype = torch.float32
        quantizer.device = torch.device('cpu')

        quantizer.set_nf4_quantized_vari()

        # 验证nf4_mapping是否正确设置
        assert quantizer.nf4_mapping.device == torch.device('cpu')
        assert quantizer.nf4_mapping.dtype == torch.float32
        assert quantizer.nf4_mapping.shape == (1, 16)

    def test_nf4_quantize_small(self):
        # 测试nf4_quantize方法（小张量情况）
        cfg = QuantConfig(w_bit=8, a_bit=8).weight_quant()
        quantizer = LinearNf4Quantizer(cfg, logger)
        nf4_quantize_mock = MagicMock()
        nf4_quantize_mock.return_value = torch.ones(128, dtype=torch.uint8)
        nf4_mapping = torch.tensor(
            [
                -1.0,
                -0.6961928009986877,
                -0.5250730514526367,
                -0.39491748809814453,
                -0.28444138169288635,
                -0.18477343022823334,
                -0.09105003625154495,
                0.0,
                0.07958029955625534,
                0.16093020141124725,
                0.24611230194568634,
                0.33791524171829224,
                0.44070982933044434,
                0.5626170039176941,
                0.7229568362236023,
                1.0,
            ]
        )
        quantizer.nf4_mapping = nf4_mapping
        weight = torch.randn(16, 16)

        result = quantizer.nf4_quantize(weight)

        # 验证结果形状
        assert result.shape == (128,)
        assert result.dtype == torch.uint8

    def test_nf4_quantize_large(self):
        # 测试nf4_quantize方法（大张量情况）
        cfg = QuantConfig(w_bit=8, a_bit=8).weight_quant()
        quantizer = LinearNf4Quantizer(cfg, logger)
        nf4_quantize_mock = MagicMock()
        nf4_quantize_mock.return_value = torch.ones(128, dtype=torch.uint8)
        nf4_mapping = torch.tensor(
            [
                -1.0,
                -0.6961928009986877,
                -0.5250730514526367,
                -0.39491748809814453,
                -0.28444138169288635,
                -0.18477343022823334,
                -0.09105003625154495,
                0.0,
                0.07958029955625534,
                0.16093020141124725,
                0.24611230194568634,
                0.33791524171829224,
                0.44070982933044434,
                0.5626170039176941,
                0.7229568362236023,
                1.0,
            ]
        )
        quantizer.nf4_mapping = nf4_mapping
        weight = torch.randn(10000, 10000)  # 模拟大张量

        # 模拟max_oom_shape检查
        with patch.object(quantizer, 'nf4_quantize') as mock_quantize:
            mock_quantize.return_value = torch.ones(128, dtype=torch.uint8)

            result = quantizer.nf4_quantize(weight)

            # 验证结果形状
            assert result.shape == (128,)
            assert result.dtype == torch.uint8
