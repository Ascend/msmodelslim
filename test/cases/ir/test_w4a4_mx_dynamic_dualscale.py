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

from msmodelslim.ir.const import mxfp4_dual_scale_sym, mxfp4_per_block_sym
from msmodelslim.ir.qal import QDType, QParam, QStorage
from msmodelslim.ir.w4a4_mx_dynamic_dualscale import W4A4MXDynamicDualScaleFakeQuantLinear

_IN_FEATURES = 8
_OUT_FEATURES = 4


def _make_module(weight_value: float, bias=None) -> W4A4MXDynamicDualScaleFakeQuantLinear:
    x_q_param = QParam(scheme=mxfp4_per_block_sym, ext={"axes": -1, "dual_block_size": _IN_FEATURES})
    w_q_param = QParam(
        scheme=mxfp4_dual_scale_sym,
        ext={
            "axes": -1,
            "scale": torch.ones(1),
            "offset": torch.zeros(1),
            "dual_block_size": _IN_FEATURES,
            "dual_scale": torch.ones(1),
        },
    )
    w_q = QStorage(QDType.MXFP4, torch.full((_OUT_FEATURES, _IN_FEATURES), weight_value))
    return W4A4MXDynamicDualScaleFakeQuantLinear(x_q_param, w_q_param, w_q, bias)


def test_forward_returns_expected_shape_and_value():
    module = _make_module(1.0)
    x = torch.ones(2, _IN_FEATURES, dtype=torch.float32)

    output = module(x)

    assert output.shape == (2, _OUT_FEATURES)
    assert output.dtype == torch.float32
    assert torch.allclose(output, torch.full((2, _OUT_FEATURES), 16.0))


def test_forward_scales_with_weight_magnitude():
    ones = _make_module(1.0)
    doubled = _make_module(2.0)
    x = torch.ones(2, _IN_FEATURES, dtype=torch.float32)

    assert torch.allclose(doubled(x), 2 * ones(x))


def test_forward_adds_bias():
    bias = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    module = _make_module(1.0, bias=bias)
    x = torch.ones(1, _IN_FEATURES, dtype=torch.float32)

    output = module(x)

    assert torch.allclose(output, torch.full((_OUT_FEATURES,), 16.0) + bias)


def test_repr_reports_impl_name():
    module = _make_module(1.0)

    assert "W4A4MXDynamicDualScaleFakeQuantLinear" in repr(module)
