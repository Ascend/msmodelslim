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

import torch
import torch.nn.functional as F

from msmodelslim import ir as qir
from msmodelslim.ir.api import dequantize
from msmodelslim.ir.qal import QABCRegistry, QParam, QScheme, QScope, QDType, QStorage


def _w_q_param(out_features: int) -> QParam:
    """构造使用 NEG_OFFSET scheme 的权重量化参数（offset 为取反后的存储值）。"""
    return QParam(
        scheme=QScheme(QScope.PER_CHANNEL_NEG_OFFSET, QDType.INT8, False),
        ext={
            "scale": torch.randn(out_features),
            "offset": torch.randn(out_features),
        },
    )


def _make_module(in_features: int = 16, out_features: int = 32, with_bias: bool = True):
    """实例化 W8A16PerChannelNegOffsetFakeQuantLinear。"""
    w_q = QStorage(QDType.INT8, torch.randint(-128, 127, (out_features, in_features)).to(torch.int8))
    x_q_param = QParam(scheme=QScheme(QScope.PER_TENSOR, QDType.FLOAT, True), ext={})
    bias = torch.randn(out_features) if with_bias else None
    return qir.W8A16PerChannelNegOffsetFakeQuantLinear(x_q_param, _w_q_param(out_features), w_q, bias)


class TestW8A16PerChannelNegOffsetFakeQuantLinear:
    """测试 W8A16 per-channel + offset 取反语义的伪量化 IR（W8A16PerChannelNegOffsetFakeQuantLinear）。"""

    def test_registered_dispatch(self):
        """按 (float_per_tensor_sym, int8_per_channel_asym_neg_offset) 可从注册表创建。"""
        x_q_param = QParam(scheme=QScheme(QScope.PER_TENSOR, QDType.FLOAT, True), ext={})
        w_q = QStorage(QDType.INT8, torch.randint(-128, 127, (32, 16)).to(torch.int8))

        created = QABCRegistry.create(
            qir.AutoFakeQuantLinear,
            (qir.float_per_tensor_sym, qir.int8_per_channel_asym_neg_offset),
            x_q_param,
            _w_q_param(32),
            w_q,
            torch.randn(32),
        )
        assert isinstance(created, qir.W8A16PerChannelNegOffsetFakeQuantLinear)

    def test_initialization(self):
        """构造后参数以 nn.Parameter 保存，shape/dtype 正确。"""
        module = _make_module(in_features=16, out_features=32, with_bias=True)

        assert module.weight.shape == (32, 16), f"weight shape 不正确: {module.weight.shape}"
        assert module.weight.dtype == torch.int8, "weight 应为 int8"
        assert module.weight_scale.shape == (32,), f"weight_scale shape 不正确: {module.weight_scale.shape}"
        assert module.weight_offset.shape == (32,), f"weight_offset shape 不正确: {module.weight_offset.shape}"
        assert module.bias is not None and module.bias.shape == (32,), "bias shape 不正确"

    def test_forward_shape(self):
        """forward 输出 shape 为 [batch, out_features]。"""
        module = _make_module(in_features=16, out_features=32, with_bias=True)
        x = torch.randn(2, 16)

        out = module(x)

        assert out.shape == (2, 32), f"输出 shape 不正确: {out.shape}"

    def test_forward_without_bias(self):
        """无 bias 时 forward 正常。"""
        module = _make_module(in_features=16, out_features=32, with_bias=False)
        assert module.bias is None, "bias 应为 None"

        x = torch.randn(3, 16)
        out = module(x)
        assert out.shape == (3, 32), f"输出 shape 不正确: {out.shape}"

    def test_forward_shape_preserves_batch(self):
        """多维 batch 维度保持。"""
        module = _make_module(in_features=16, out_features=32)
        x = torch.randn(4, 7, 16)

        out = module(x)
        assert out.shape == (4, 7, 32), f"多维 batch 输出 shape 不正确: {out.shape}"

    def test_forward_negates_stored_offset_for_dequantize(self):
        """forward 内部反量化时对存储的取反 offset 再取反还原（等价于用原始 offset 重构）。"""
        module = _make_module(in_features=16, out_features=32, with_bias=True)
        x = torch.randn(2, 16)

        # 用「还原后的原始 offset」重构权重（等价于 forward 内部行为）
        qp_orig = QParam(
            scheme=QScheme(QScope.PER_CHANNEL, QDType.INT8, False),
            ext={
                "scale": module.weight_scale.data,
                "offset": -module.weight_offset.data,
            },
        )
        w_orig = dequantize(QStorage(QDType.INT8, module.weight.data).T, qp_orig).T.value

        out = module(x)
        expected = F.linear(x, w_orig, module.bias)

        assert torch.allclose(out, expected, atol=1e-5), "forward 未按 -weight_offset 还原 offset 重构"
