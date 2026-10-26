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

from msmodelslim.ir.const import int4_per_channel_sym, int4_per_group_sym, int4_per_token_sym
from msmodelslim.ir.qal import QDType, QParam, QStorage
from msmodelslim.ir.w4a4_dynamic import (
    W4A4DynamicPerChannelFakeQuantLinear,
    W4A4DynamicPerGroupFakeQuantLinear,
)

_IN_FEATURES = 8
_OUT_FEATURES = 4


def _weight(value: float) -> torch.Tensor:
    return torch.full((_OUT_FEATURES, _IN_FEATURES), value, dtype=torch.float32)


def _activation_param() -> QParam:
    return QParam(scheme=int4_per_token_sym, ext={})


def _make_per_channel(scale: float = 1.0, bias=None) -> W4A4DynamicPerChannelFakeQuantLinear:
    w_q_param = QParam(
        scheme=int4_per_channel_sym,
        ext={"scale": torch.full((_OUT_FEATURES,), scale), "offset": torch.zeros(_OUT_FEATURES)},
    )
    return W4A4DynamicPerChannelFakeQuantLinear(
        _activation_param(), w_q_param, QStorage(QDType.INT4, _weight(1.0)), bias
    )


def _make_per_group(scale: float = 1.0, bias=None) -> W4A4DynamicPerGroupFakeQuantLinear:
    w_q_param = QParam(
        scheme=int4_per_group_sym,
        ext={
            "scale": torch.full((_OUT_FEATURES,), scale),
            "offset": torch.zeros(_OUT_FEATURES),
            "group_size": _IN_FEATURES,
        },
    )
    return W4A4DynamicPerGroupFakeQuantLinear(_activation_param(), w_q_param, QStorage(QDType.INT4, _weight(1.0)), bias)


def test_per_channel_forward_returns_expected_shape_and_value():
    module = _make_per_channel()
    x = torch.ones(2, _IN_FEATURES, dtype=torch.float32)

    output = module(x)

    assert output.shape == (2, _OUT_FEATURES)
    assert output.dtype == torch.float32
    assert torch.allclose(output, torch.full((2, _OUT_FEATURES), 8.0))


def test_per_channel_weight_scale_multiplies_dequantized_weight():
    module = _make_per_channel(scale=2.0)
    x = torch.ones(2, _IN_FEATURES, dtype=torch.float32)

    output = module(x)

    assert torch.allclose(output, torch.full((2, _OUT_FEATURES), 16.0))


def test_per_channel_forward_adds_bias():
    bias = torch.arange(_OUT_FEATURES, dtype=torch.float32)
    module = _make_per_channel(bias=bias)
    x = torch.ones(1, _IN_FEATURES, dtype=torch.float32)

    output = module(x)

    assert torch.allclose(output, torch.full((_OUT_FEATURES,), 8.0) + bias)


def test_per_channel_repr_reports_impl_name():
    module = _make_per_channel()

    assert "W4A4DynamicFakeQuantLinear" in repr(module)


def test_per_group_forward_returns_expected_shape_and_value():
    module = _make_per_group()
    x = torch.ones(2, _IN_FEATURES, dtype=torch.float32)

    output = module(x)

    assert output.shape == (2, _OUT_FEATURES)
    assert torch.allclose(output, torch.full((2, _OUT_FEATURES), 8.0))


def test_per_group_stores_group_size_and_applies_scale():
    module = _make_per_group(scale=2.0)
    x = torch.ones(2, _IN_FEATURES, dtype=torch.float32)

    output = module(x)

    assert module.group_size == _IN_FEATURES
    assert torch.allclose(output, torch.full((2, _OUT_FEATURES), 16.0))
