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

import pytest
import torch

from msmodelslim.ir.const import fp8_e4m3_per_channel_sym, fp8_e4m3_per_token_sym
from msmodelslim.ir.qal import QDType, QParam, QStorage
from msmodelslim.ir.w8a8_fp_dynamic import WFP8AFP8DynamicPerChannelFakeQuantLinear
from msmodelslim.utils.exception import SchemaValidateError

_IN_FEATURES = 8
_OUT_FEATURES = 4


def _make_module(bias=None) -> WFP8AFP8DynamicPerChannelFakeQuantLinear:
    x_q_param = QParam(scheme=fp8_e4m3_per_token_sym, ext={})
    w_q_param = QParam(scheme=fp8_e4m3_per_channel_sym, ext={"scale": torch.ones(_OUT_FEATURES)})
    w_q = QStorage(QDType.FP8_E4M3, torch.ones(_OUT_FEATURES, _IN_FEATURES, dtype=torch.float32))
    return WFP8AFP8DynamicPerChannelFakeQuantLinear(x_q_param, w_q_param, w_q, bias)


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


def test_init_raises_when_weight_scale_missing():
    x_q_param = QParam(scheme=fp8_e4m3_per_token_sym, ext={})
    w_q_param = QParam(scheme=fp8_e4m3_per_channel_sym, ext={})
    w_q = QStorage(QDType.FP8_E4M3, torch.ones(_OUT_FEATURES, _IN_FEATURES, dtype=torch.float32))

    with pytest.raises(SchemaValidateError):
        WFP8AFP8DynamicPerChannelFakeQuantLinear(x_q_param, w_q_param, w_q, None)
