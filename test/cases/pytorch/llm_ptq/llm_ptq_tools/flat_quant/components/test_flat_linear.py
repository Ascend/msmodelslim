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
# pylint: disable=consider-using-from-import

import pytest  # noqa: F401
import torch
import torch.nn as nn
from unittest.mock import Mock
from unittest.mock import patch  # noqa: F401

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.components.flat_linear import (
    FakeQuantizedLinear,
    FlatQuantizedLinear,
    FlatNormWrapper,
    ForwardMode,
    FakeQuantizedLinearConfig,
)


class TestFakeQuantizedLinearConfig:
    def test_default_initialization(self):
        config = FakeQuantizedLinearConfig()

        assert config.w_bits == 16
        assert config.a_bits == 16
        assert config.w_asym is False
        assert config.a_asym is False
        assert config.lwc is False
        assert config.lac is False
        assert config.a_groupsize == -1
        assert config.a_per_tensor is False

    def test_custom_initialization(self):
        config = FakeQuantizedLinearConfig(
            w_bits=8, a_bits=4, w_asym=True, a_asym=True, lwc=True, lac=True, a_groupsize=128, a_per_tensor=True
        )

        assert config.w_bits == 8
        assert config.a_bits == 4
        assert config.w_asym is True
        assert config.a_asym is True
        assert config.lwc is True
        assert config.lac is True
        assert config.a_groupsize == 128
        assert config.a_per_tensor is True


class TestForwardMode:
    def test_enum_values(self):
        assert ForwardMode.ORG.value == "org"
        assert ForwardMode.CALIB.value == "calib"
        assert ForwardMode.EVAL.value == "eval"


class TestFakeQuantizedLinear:
    def test_initialization_with_bias(self):
        linear = nn.Linear(128, 64, bias=True)
        config = FakeQuantizedLinearConfig(w_bits=8, a_bits=8)

        fake_linear = FakeQuantizedLinear(config, linear)

        assert fake_linear.weight.shape == (64, 128)
        assert fake_linear.bias.shape == (64,)
        assert fake_linear._mode == ForwardMode.ORG
        assert hasattr(fake_linear, 'weight_quantizer')
        assert hasattr(fake_linear, 'act_quantizer')

    def test_initialization_without_bias(self):
        linear = nn.Linear(128, 64, bias=False)
        config = FakeQuantizedLinearConfig(w_bits=8, a_bits=8)

        fake_linear = FakeQuantizedLinear(config, linear)

        assert fake_linear.weight.shape == (64, 128)
        assert fake_linear.bias is None

    def test_extra_repr(self):
        linear = nn.Linear(128, 64, bias=True)
        config = FakeQuantizedLinearConfig()
        fake_linear = FakeQuantizedLinear(config, linear)

        repr_str = fake_linear.extra_repr()

        assert "weight shape: (64, 128)" in repr_str
        assert "bias=True" in repr_str

    def test_set_act_clip_factor_no_lac(self):
        linear = nn.Linear(128, 64)
        config = FakeQuantizedLinearConfig(lac=False)
        fake_linear = FakeQuantizedLinear(config, linear)

        # Should not raise error even if lac is False
        fake_linear.set_act_clip_factor(torch.nn.Parameter(torch.tensor(0.8)))

    def test_forward_org_mode(self):
        linear = nn.Linear(128, 64)
        config = FakeQuantizedLinearConfig()
        fake_linear = FakeQuantizedLinear(config, linear)
        fake_linear.to_org_mode()

        x = torch.randn(2, 128)
        result = fake_linear(x)

        assert result.shape == (2, 64)

    def test_forward_calib_mode(self):
        linear = nn.Linear(128, 64)
        config = FakeQuantizedLinearConfig(w_bits=8, a_bits=8)
        fake_linear = FakeQuantizedLinear(config, linear)
        fake_linear.to_calib_mode()

        x = torch.randn(2, 128)
        result = fake_linear(x)

        assert result.shape == (2, 64)

    def test_reparameterize(self):
        linear = nn.Linear(128, 64)
        config = FakeQuantizedLinearConfig(w_bits=8, a_bits=8)
        fake_linear = FakeQuantizedLinear(config, linear)

        fake_linear.reparameterize()

        # Should not raise any errors

    def test_mode_switching(self):
        linear = nn.Linear(128, 64)
        config = FakeQuantizedLinearConfig()
        fake_linear = FakeQuantizedLinear(config, linear)

        fake_linear.to_org_mode()
        assert fake_linear._mode == ForwardMode.ORG

        fake_linear.to_calib_mode()
        assert fake_linear._mode == ForwardMode.CALIB

        fake_linear.to_eval_mode()
        assert fake_linear._mode == ForwardMode.EVAL

    def test_fake_quant_weight(self):
        linear = nn.Linear(128, 64)
        config = FakeQuantizedLinearConfig(w_bits=8)
        fake_linear = FakeQuantizedLinear(config, linear)
        original_weight = fake_linear.weight.data.clone()

        fake_linear.fake_quant_weight()

        # Weight should be modified by quantization
        assert not torch.equal(fake_linear.weight.data, original_weight)


class TestFlatQuantizedLinear:
    def test_initialization(self):
        linear = nn.Linear(128, 64)
        config = FakeQuantizedLinearConfig(w_bits=8, a_bits=8)

        flat_linear = FlatQuantizedLinear(config, linear)

        assert flat_linear.weight_in_trans is None
        assert flat_linear.weight_out_trans is None
        assert flat_linear.act_in_trans is None
        assert flat_linear.save_trans is None

    def test_set_trans(self):
        linear = nn.Linear(128, 64)
        config = FakeQuantizedLinearConfig()
        flat_linear = FlatQuantizedLinear(config, linear)

        mock_trans = Mock()
        flat_linear.set_trans(
            weight_in_trans=mock_trans, weight_out_trans=mock_trans, act_in_trans=mock_trans, save_trans=mock_trans
        )

        assert flat_linear.weight_in_trans == mock_trans
        assert flat_linear.weight_out_trans == mock_trans
        assert flat_linear.act_in_trans == mock_trans
        assert flat_linear.save_trans == mock_trans

    def test_set_trans_partial(self):
        linear = nn.Linear(128, 64)
        config = FakeQuantizedLinearConfig()
        flat_linear = FlatQuantizedLinear(config, linear)

        mock_trans = Mock()
        flat_linear.set_trans(weight_in_trans=mock_trans)

        assert flat_linear.weight_in_trans == mock_trans
        assert flat_linear.weight_out_trans is None

    def test_forward_org_mode(self):
        linear = nn.Linear(128, 64)
        config = FakeQuantizedLinearConfig()
        flat_linear = FlatQuantizedLinear(config, linear)
        flat_linear.to_org_mode()

        x = torch.randn(2, 128)
        result = flat_linear(x)

        assert result.shape == (2, 64)

    def test_forward_calib_mode_with_trans(self):
        linear = nn.Linear(128, 64)
        config = FakeQuantizedLinearConfig(w_bits=8, a_bits=8)
        flat_linear = FlatQuantizedLinear(config, linear)
        flat_linear.to_calib_mode()

        # Mock transformation that returns input unchanged
        mock_trans = Mock()
        mock_trans.side_effect = lambda x, inv_t=False: x
        flat_linear.set_trans(act_in_trans=mock_trans)

        x = torch.randn(2, 128)
        result = flat_linear(x)

        assert result.shape == (2, 64)

    def test_forward_calib_mode_with_weight_trans(self):
        linear = nn.Linear(128, 64)
        config = FakeQuantizedLinearConfig(w_bits=8, a_bits=8, lwc=True)
        flat_linear = FlatQuantizedLinear(config, linear)
        flat_linear.to_calib_mode()

        mock_weight_trans = Mock()
        mock_weight_trans.side_effect = lambda x, inv_t=False: x
        flat_linear.set_trans(weight_in_trans=mock_weight_trans, weight_out_trans=mock_weight_trans)

        x = torch.randn(2, 128)
        result = flat_linear(x)

        assert result.shape == (2, 64)

    def test_forward_eval_mode(self):
        linear = nn.Linear(128, 64)
        config = FakeQuantizedLinearConfig(w_bits=8, a_bits=8)
        flat_linear = FlatQuantizedLinear(config, linear)
        flat_linear.to_eval_mode()

        x = torch.randn(2, 128)
        result = flat_linear(x)

        assert result.shape == (2, 64)

    def test_reparameterize_with_transformations(self):
        linear = nn.Linear(128, 64, bias=True)
        config = FakeQuantizedLinearConfig(w_bits=8, a_bits=8, lwc=True)
        flat_linear = FlatQuantizedLinear(config, linear)

        mock_trans = Mock()
        mock_trans.side_effect = lambda x, inv_t=False: x
        flat_linear.set_trans(weight_in_trans=mock_trans, weight_out_trans=mock_trans)

        flat_linear.reparameterize()

        assert flat_linear.weight_in_trans is None
        assert flat_linear.weight_out_trans is None

    def test_reparameterize_already_eval_mode(self):
        linear = nn.Linear(128, 64)
        config = FakeQuantizedLinearConfig(w_bits=8, a_bits=8)
        flat_linear = FlatQuantizedLinear(config, linear)
        flat_linear._mode = ForwardMode.EVAL

        # Should not do anything if already in eval mode
        flat_linear.reparameterize()


class TestFlatNormWrapper:
    def test_initialization_with_bias(self):
        norm = nn.LayerNorm(128)
        mock_trans = Mock()

        wrapper = FlatNormWrapper(norm, trans=mock_trans)

        assert wrapper.norm == norm
        assert wrapper.trans == mock_trans
        assert wrapper._mode == ForwardMode.ORG
        assert hasattr(wrapper.norm, 'weight')
        assert hasattr(wrapper.norm, 'bias')

    def test_initialization_without_bias(self):
        # Create a norm layer without bias
        norm = nn.LayerNorm(128, bias=False)
        wrapper = FlatNormWrapper(norm)

        assert wrapper.norm == norm
        assert wrapper.trans is None

    def test_forward_org_mode(self):
        norm = nn.LayerNorm(128)
        wrapper = FlatNormWrapper(norm)
        wrapper.to_org_mode()

        x = torch.randn(2, 128)
        result = wrapper(x)

        assert result.shape == (2, 128)

    def test_forward_calib_eval_mode_with_trans(self):
        norm = nn.LayerNorm(128)
        mock_trans = Mock()
        mock_trans.side_effect = lambda x: x
        wrapper = FlatNormWrapper(norm, trans=mock_trans)
        wrapper.to_calib_mode()

        x = torch.randn(2, 128)
        result = wrapper(x)

        assert result.shape == (2, 128)
        mock_trans.assert_called_once()

    def test_forward_calib_eval_mode_without_trans(self):
        norm = nn.LayerNorm(128)
        wrapper = FlatNormWrapper(norm, trans=None)
        wrapper.to_calib_mode()

        x = torch.randn(2, 128)
        result = wrapper(x)

        assert result.shape == (2, 128)

    def test_mode_switching(self):
        norm = nn.LayerNorm(128)
        wrapper = FlatNormWrapper(norm)

        wrapper.to_org_mode()
        assert wrapper._mode == ForwardMode.ORG

        wrapper.to_calib_mode()
        assert wrapper._mode == ForwardMode.CALIB

        wrapper.to_eval_mode()
        assert wrapper._mode == ForwardMode.EVAL
