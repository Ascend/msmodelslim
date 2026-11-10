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
from torch.nn import functional as F

from msmodelslim.ir.api import calculate_qparam, fake_quantize
from msmodelslim.ir.auto import AutoFakeQuantLinear
from msmodelslim.ir.const import int4_per_channel_sym, int4_per_group_asym, int4_per_group_sym, int4_per_token_sym
from msmodelslim.ir.qal import QDType, QParam, QScope, QStorage
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


class TestW4A4DynamicPerGroupFakeQuantLinear:
    """验证 W4A4 per-group 动态 IR 的权重分组布局。"""

    def test_forward_uses_input_channel_groups_when_scales_differ(self):
        """场景：每行包含两个不同 scale 的输入通道组。预期：按行分组输出正确。"""
        weight = torch.ones(4, 4, dtype=torch.int8)
        w_q_param = QParam(
            scheme=int4_per_group_sym,
            ext={
                "scale": torch.arange(1, 9, dtype=torch.float32).reshape(4, 2),
                "offset": torch.zeros(4, 2),
                "group_size": 2,
            },
        )
        module = W4A4DynamicPerGroupFakeQuantLinear(None, w_q_param, QStorage(QDType.INT4, weight), None)

        output = module(torch.ones(1, 4))

        torch.testing.assert_close(output, torch.tensor([[6.0, 14.0, 22.0, 30.0]]))

    @pytest.mark.parametrize(
        "out_features,in_features,group_size,symmetric,with_bias,input_shape",
        [
            (4, 4, 2, True, False, (2, 4)),
            (3, 4, 2, True, True, (2, 3, 4)),
            (3, 6, 3, False, True, (2, 6)),
            (4, 8, 4, False, False, (2, 3, 8)),
            (3, 4, 1, True, False, (2, 4)),
            (3, 4, 4, False, True, (2, 4)),
        ],
        ids=["square", "rectangular-3d-bias", "asymmetric", "asymmetric-3d", "element-groups", "row-groups"],
    )
    def test_forward_matches_reference_when_group_layout_varies(
        self, out_features, in_features, group_size, symmetric, with_bias, input_shape
    ):
        """场景：分组布局、offset、bias 和输入维度变化。预期：注册 IR 与按行展开的参考值一致。"""
        weight = (torch.arange(out_features * in_features).reshape(out_features, in_features) % 15 - 7).to(torch.int8)
        num_groups = in_features // group_size
        scale = (
            torch.arange(1, out_features * num_groups + 1, dtype=torch.float32).reshape(out_features, num_groups) / 8
        )
        offset = torch.zeros_like(scale) if symmetric else torch.arange(scale.numel()).reshape_as(scale).float() - 2
        bias = torch.arange(out_features, dtype=torch.float32) / 4 if with_bias else None
        w_q_param = QParam(
            scheme=int4_per_group_sym if symmetric else int4_per_group_asym,
            ext={"scale": scale, "offset": offset, "group_size": group_size},
        )
        module = AutoFakeQuantLinear.create(
            QParam(scheme=int4_per_token_sym), w_q_param, QStorage(QDType.INT4, weight), bias
        )
        assert isinstance(module, W4A4DynamicPerGroupFakeQuantLinear)
        x = torch.arange(torch.tensor(input_shape).prod().item(), dtype=torch.float32).reshape(input_shape) / 7 - 2
        x_flat = x.reshape(-1, in_features)
        x_q_param = calculate_qparam(
            x_flat.amin(dim=1, keepdim=True), x_flat.amax(dim=1, keepdim=True), QDType.INT4, QScope.PER_TOKEN, True
        )
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x_flat), x_q_param).value.reshape(input_shape)
        # Weight groups run along input features within each output row.
        expected_weight = (weight.float() - offset.repeat_interleave(group_size, dim=1)) * scale.repeat_interleave(
            group_size, dim=1
        )
        expected = F.linear(x_q_dq, expected_weight, bias)

        torch.testing.assert_close(module(x), expected)
