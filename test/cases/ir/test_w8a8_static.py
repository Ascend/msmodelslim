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

from msmodelslim.ir.const import int8_per_channel_sym, int8_per_tensor_asym, int8_per_tensor_sym
from msmodelslim.ir.qal import QDType, QParam, QStorage
from msmodelslim.ir.w8a8_static import W8A8StaticFakeQuantLinear

_IN_FEATURES = 8
_OUT_FEATURES = 4


def _make_module(input_scale: float = 1.0, bias=None) -> W8A8StaticFakeQuantLinear:
    x_q_param = QParam(
        scheme=int8_per_tensor_asym,
        ext={"scale": torch.full((1,), input_scale), "offset": torch.zeros(1)},
    )
    w_q_param = QParam(scheme=int8_per_channel_sym, ext={"scale": torch.ones(_OUT_FEATURES)})
    w_q = QStorage(QDType.INT8, torch.ones(_OUT_FEATURES, _IN_FEATURES))
    return W8A8StaticFakeQuantLinear(x_q_param, w_q_param, w_q, bias)


@pytest.mark.parametrize("x_value,input_scale,expected", [(1.0, 1.0, 8.0), (0.6, 1.0, 8.0), (0.6, 0.5, 4.0)])
def test_forward_quantizes_activation_with_input_scale(x_value, input_scale, expected):
    module = _make_module(input_scale=input_scale)
    x = torch.full((1, _IN_FEATURES), x_value, dtype=torch.float32)

    output = module(x)

    assert output.shape == (1, _OUT_FEATURES)
    assert torch.allclose(output, torch.full((1, _OUT_FEATURES), expected), atol=1e-5)


def test_forward_uses_symmetric_per_tensor_scheme_for_activations():
    module = _make_module()
    x = torch.ones(2, _IN_FEATURES, dtype=torch.float32)

    output = module(x)

    assert output.shape == (2, _OUT_FEATURES)
    assert torch.allclose(output, torch.full((2, _OUT_FEATURES), 8.0))


def test_forward_adds_bias():
    bias = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    module = _make_module(bias=bias)
    x = torch.ones(1, _IN_FEATURES, dtype=torch.float32)

    output = module(x)

    assert torch.allclose(output, torch.full((_OUT_FEATURES,), 8.0) + bias)


def test_symmetric_per_tensor_dispatch_is_supported():
    # (int8_per_tensor_sym, int8_per_channel_sym) 也注册到同一实现。
    x_q_param = QParam(
        scheme=int8_per_tensor_sym,
        ext={"scale": torch.ones(1), "offset": torch.zeros(1)},
    )
    w_q_param = QParam(scheme=int8_per_channel_sym, ext={"scale": torch.ones(_OUT_FEATURES)})
    w_q = QStorage(QDType.INT8, torch.ones(_OUT_FEATURES, _IN_FEATURES))

    module = W8A8StaticFakeQuantLinear(x_q_param, w_q_param, w_q, None)
    x = torch.ones(1, _IN_FEATURES, dtype=torch.float32)

    assert torch.allclose(module(x), torch.full((1, _OUT_FEATURES), 8.0))
