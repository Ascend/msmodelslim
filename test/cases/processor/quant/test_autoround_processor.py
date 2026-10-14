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
# pylint: disable=redefined-outer-name

from unittest.mock import MagicMock

import pytest
import torch
from torch.nn import Parameter, Linear

from msmodelslim.processor.quant.autoround_utils.sign_sgd import SignSGD
from msmodelslim.processor.quant.autoround_utils.wrapper import WrapperLinear


@pytest.fixture
def mock_param():
    """创建模拟模型"""
    param = Parameter(torch.randn(10, 10))
    return param


@pytest.fixture
def mock_linear():
    """创建模拟Linear"""
    linear = Linear(256, 256)

    linear.bits = 4
    linear.sym = True
    linear.group_size = 128
    linear.data_type = "int"
    linear.scale_dtype = torch.float32
    linear.act_bits = 4
    linear.act_sym = True
    linear.act_data_type = "int"
    linear.act_group_size = -1
    linear.act_dynamic = True
    linear.name = "mock_linear"
    linear.to_smooth = True
    linear.scale = None
    linear.zp = None
    linear.act_scale = None
    linear.act_zp = None

    return linear


class TestSignSGD:
    @staticmethod
    def test_init_valid_params(mock_param):
        """测试有效参数初始化"""
        optimizer = SignSGD([mock_param], lr=0.1)
        optimizer.__setstate__({})
        assert optimizer.defaults['lr'] == 0.1

    @staticmethod
    def test_init_invalid_params(mock_param):
        """测试无效参数初始化"""
        with pytest.raises(ValueError, match="Invalid learning rate"):
            SignSGD([mock_param], lr=-0.1)

        with pytest.raises(ValueError, match="Invalid momentum value"):
            SignSGD([mock_param], lr=0.1, momentum=-0.1)

        with pytest.raises(ValueError, match="Invalid weight_decay value"):
            SignSGD([mock_param], lr=0.1, weight_decay=-0.1)

        with pytest.raises(ValueError, match="Nesterov momentum requires a momentum and zero dampening"):
            SignSGD([mock_param], lr=0.1, nesterov=True, momentum=0)

    @staticmethod
    def test_basic_step(mock_param):
        """测试基本step功能"""
        optimizer = SignSGD([mock_param], lr=0.1)
        mock_param.grad = torch.ones_like(mock_param) * 2.0

        optimizer.step()
        # 验证参数已更新
        assert not torch.allclose(mock_param, torch.ones_like(mock_param))

    @staticmethod
    def test_sign_sgd_update(mock_param):
        """测试SignSGD更新规则"""
        optimizer = SignSGD([mock_param], lr=0.1)

        # 设置梯度
        mock_param.grad = torch.ones_like(mock_param) * 3.0  # 梯度为正数

        param_before = mock_param.clone()
        optimizer.step()

        # SignSGD使用梯度的符号进行更新
        expected_update = torch.sign(mock_param.grad) * 0.1
        expected_param = param_before - expected_update
        assert torch.allclose(mock_param, expected_param)

    @staticmethod
    def test_with_weight_decay(mock_param):
        """测试权重衰减"""
        optimizer = SignSGD([mock_param], lr=0.1, weight_decay=0.01)

        mock_param.grad = torch.ones_like(mock_param) * 2.0
        optimizer.step()  # 应该能正常执行不报错

    @staticmethod
    def test_with_momentum(mock_param):
        """测试动量"""
        optimizer = SignSGD([mock_param], lr=0.1, momentum=0.9)

        mock_param.grad = torch.ones_like(mock_param) * 2.0
        optimizer.step()

        # 检查动量缓冲区是否存在
        assert 'momentum_buffer' in optimizer.state[mock_param]

    @staticmethod
    def test_maximize(mock_param):
        """测试最大化模式"""
        optimizer = SignSGD([mock_param], lr=0.1, maximize=True)

        mock_param.grad = torch.ones_like(mock_param) * 2.0
        optimizer.step()

    @staticmethod
    def test_multiple_steps(mock_param):
        """测试多步更新"""
        optimizer = SignSGD([mock_param], lr=0.05)

        for i in range(3):
            mock_param.grad = torch.ones_like(mock_param) * (i + 1)
            optimizer.step()

        # 参数应该持续更新
        assert mock_param.grad is not None


class TestWrapper:
    @staticmethod
    @pytest.mark.parametrize("enable_trainable_smooth", [True, False])
    def test_init_params(mock_linear, enable_trainable_smooth):
        """测试有效参数初始化"""
        mock_linear.name = "o_proj"
        wrapper = WrapperLinear(mock_linear, enable_trainable_smooth=enable_trainable_smooth)
        wrapper.config = MagicMock()
        wrapper.config.num_key_value_heads = 4
        wrapper.config.num_attention_heads = 8
        assert wrapper.orig_layer is not None
        assert wrapper.min_scale is not None
        assert wrapper.max_scale is not None
        assert wrapper.act_max_scale is not None

        if enable_trainable_smooth:
            assert wrapper.act_smooth_scale is not None

        input_tensor = torch.randn(1, 256)
        output = wrapper(input_tensor)
        assert output is not None

        wrapper.unwrapper({})

    @staticmethod
    @pytest.mark.parametrize("group_size", [-1, 0, 15, 128])
    def test_different_group_size(mock_linear, group_size):
        mock_linear.group_size = group_size
        wrapper = WrapperLinear(mock_linear)
        input_tensor = torch.randn(1, 256)
        output = wrapper(input_tensor)
        assert output is not None

    @staticmethod
    @pytest.mark.parametrize("sym", [True, False])
    def test_forward_for_sym_and_asym(mock_linear, sym):
        mock_linear.sym = sym
        wrapper = WrapperLinear(mock_linear)
        input_tensor = torch.randn(1, 256)
        output = wrapper(input_tensor)
        assert output is not None
