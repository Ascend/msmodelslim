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

Integer weight edges. Each processor is one hop from FLOAT:

    FLOAT -> W8A16_INT8
    FLOAT -> W8A16_INT8_ASYM
    FLOAT -> W8A8_INT8
    FLOAT -> W4A8_INT4
    FLOAT -> W4A4_INT4

FP8_BLOCK and MXFP4_PACKED reach these targets through their own dequant
processors and then one of the edges above. There is no FLOAT hop inside
this module: the source weight is already floating point.

Quantization is data-free minmax. The channel axis is 0, matching Linear
weights stored as ``[out_features, in_features]``.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

import msmodelslim.core.quantization_schema as qir
from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.convert.types import IRKind, LossLevel
from msmodelslim.core.dtype_ops import (
    int4_quantize_dynamic_per_channel_sym,
    int8_quantize_dynamic_per_channel_asym,
    int8_quantize_dynamic_per_channel_sym,
)
from msmodelslim.core.quantization_algorithm.convert.base import BaseConvertProcessor
from msmodelslim.core.quantization_algorithm.convert.mxfp8_quant import _materialize_linear
from msmodelslim.core.quantization_schema.const import (
    float_per_tensor_sym,
    int4_per_channel_sym,
    int4_per_token_sym,
    int8_per_channel_asym_neg_offset,
    int8_per_channel_sym,
    int8_per_token_sym,
)
from msmodelslim.core.quantization_schema.qal import QParam, QStorage

QuantizeFn = Callable[[nn.Module], nn.Module]


def _channel_param(scale: torch.Tensor) -> torch.Tensor:
    """Per-channel scale/offset from dtype_ops is ``[out, 1]``; modules store ``[out]``."""
    return scale.squeeze(-1)


def _create_fake_quant(
    module: nn.Module,
    w_scheme,
    a_scheme,
    q_weight: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor | None,
) -> nn.Module:
    ext = {"scale": _channel_param(scale)}
    if offset is not None:
        ext["offset"] = _channel_param(offset)
    return qir.AutoFakeQuantLinear.create(
        QParam(scheme=a_scheme),
        QParam(scheme=w_scheme, ext=ext),
        QStorage(dtype=w_scheme.dtype, value=q_weight),
        None if module.bias is None else module.bias.detach(),
    )


def quantize_w8a16_sym(module: nn.Module) -> nn.Module:
    q_weight, scale = int8_quantize_dynamic_per_channel_sym(module.weight.detach(), axis=0, dtype=torch.int8)
    return _create_fake_quant(
        module,
        int8_per_channel_sym,
        float_per_tensor_sym,
        q_weight,
        scale,
        torch.zeros_like(scale),
    )


def quantize_w8a16_asym(module: nn.Module) -> nn.Module:
    q_weight, scale, zp = int8_quantize_dynamic_per_channel_asym(module.weight.detach(), axis=0, dtype=torch.int8)
    # vLLM dequantizes ``(q + offset) * scale``; dtype_ops uses ``(q - zp) * scale``.
    return _create_fake_quant(
        module,
        int8_per_channel_asym_neg_offset,
        float_per_tensor_sym,
        q_weight,
        scale,
        -zp,
    )


def quantize_w8a8(module: nn.Module) -> nn.Module:
    q_weight, scale = int8_quantize_dynamic_per_channel_sym(module.weight.detach(), axis=0, dtype=torch.int8)
    return _create_fake_quant(module, int8_per_channel_sym, int8_per_token_sym, q_weight, scale, None)


def quantize_w4a8(module: nn.Module) -> nn.Module:
    q_weight, scale = int4_quantize_dynamic_per_channel_sym(module.weight.detach(), axis=0, dtype=torch.int8)
    return _create_fake_quant(module, int4_per_channel_sym, int8_per_token_sym, q_weight, scale, None)


def quantize_w4a4(module: nn.Module) -> nn.Module:
    q_weight, scale = int4_quantize_dynamic_per_channel_sym(module.weight.detach(), axis=0, dtype=torch.int8)
    return _create_fake_quant(
        module,
        int4_per_channel_sym,
        int4_per_token_sym,
        q_weight,
        scale,
        torch.zeros_like(scale),
    )


class IntWeightQuantProcessor(BaseConvertProcessor):
    """FLOAT -> one integer IR. ``quantize`` builds the matching fake-quant linear."""

    loss_level = LossLevel.LOSSY.value
    src_ir = IRKind.FLOAT

    def __init__(self, name: str, dst_ir: IRKind, quantize: QuantizeFn):
        self.name = name
        self.dst_ir = dst_ir
        self._quantize = quantize

    def transform(self, module: nn.Module, context: ConvertContext) -> nn.Module:
        linear = _materialize_linear(module, context)
        if linear is None:
            return module
        return self._quantize(linear)


INT_QUANT_PROCESSORS = (
    IntWeightQuantProcessor("W8A16Int8QuantProcessor", IRKind.W8A16_INT8, quantize_w8a16_sym),
    IntWeightQuantProcessor("W8A16Int8AsymQuantProcessor", IRKind.W8A16_INT8_ASYM, quantize_w8a16_asym),
    IntWeightQuantProcessor("W8A8Int8QuantProcessor", IRKind.W8A8_INT8, quantize_w8a8),
    IntWeightQuantProcessor("W4A8Int4QuantProcessor", IRKind.W4A8_INT4, quantize_w4a8),
    IntWeightQuantProcessor("W4A4Int4QuantProcessor", IRKind.W4A4_INT4, quantize_w4a4),
)
