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

import torch

from msmodelslim.ir.const import int4_per_channel_sym, int8_per_token_sym
from msmodelslim.ir.qal import QDType, QParam, QStorage
from msmodelslim.ir.w4a8_dynamic import W4A8DynamicFakeQuantLinear

_IN_FEATURES = 8
_OUT_FEATURES = 4


def _make_module(bias=None) -> W4A8DynamicFakeQuantLinear:
    x_q_param = QParam(scheme=int8_per_token_sym, ext={})
    w_q_param = QParam(scheme=int4_per_channel_sym, ext={"scale": torch.ones(_OUT_FEATURES)})
    w_q = QStorage(QDType.INT4, torch.ones(_OUT_FEATURES, _IN_FEATURES))
    return W4A8DynamicFakeQuantLinear(x_q_param, w_q_param, w_q, bias)


def test_forward_returns_expected_shape_and_value():
    module = _make_module()
    x = torch.ones(2, _IN_FEATURES, dtype=torch.float32)

    output = module(x)

    assert output.shape == (2, _OUT_FEATURES)
    assert output.dtype == torch.float32
    assert torch.allclose(output, torch.full((2, _OUT_FEATURES), 8.0))


def test_forward_adds_bias():
    bias = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    module = _make_module(bias=bias)
    x = torch.ones(1, _IN_FEATURES, dtype=torch.float32)

    output = module(x)

    assert torch.allclose(output, torch.full((_OUT_FEATURES,), 8.0) + bias)


def test_init_stores_weight_and_bias_as_frozen_parameters():
    module = _make_module(bias=torch.ones(_OUT_FEATURES))

    assert module.weight.requires_grad is False
    assert module.bias.requires_grad is False
    assert module.weight_scale.requires_grad is False
