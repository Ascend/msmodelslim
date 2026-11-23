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

IR edge: FLOAT -> HIFP4 (W4A8_HIFP4_MXFP8 export carrier).
"""

from __future__ import annotations

import torch
from torch import nn

from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.convert.types import IRKind, LossLevel
from msmodelslim.core.quant_service.modelslim_convert.virtual_module import ModelFreeLinear
from msmodelslim.core.quantization_schema.w4a8_hifx_mx_dynamic import W4A8HiFxMXDynamicPerBlockFakeQuantLinear
from msmodelslim.core.quantization_algorithm.convert.base import BaseConvertProcessor


def _materialize_linear(module: nn.Module, context: ConvertContext | None = None) -> nn.Linear | None:
    if isinstance(module, ModelFreeLinear):
        if not module.lazy_initialized and context is not None and context.reader is not None:
            module.lazy_init(context.reader, device="cpu")
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2:
            return None
        bias = getattr(module, "bias", None)
        linear = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
        linear.weight = nn.Parameter(weight.detach().to(torch.bfloat16), requires_grad=False)
        if bias is not None:
            linear.bias = nn.Parameter(bias.detach().to(torch.bfloat16), requires_grad=False)
        return linear
    if isinstance(module, nn.Linear):
        return module
    return None


def _hif4_quantize_weight(weight: torch.Tensor, group_size: int = 64):
    """Quantize bf16/fp32 weight into hif4 mantissa + 3-level scales."""
    w = weight.to(torch.float32)
    m, n = w.shape
    if n % group_size != 0:
        raise ValueError(f"weight in_features {n} must be divisible by group_size {group_size}")
    g = n // group_size
    block = w.view(m, g, group_size)

    abs_block = block.abs()
    max_lv1 = abs_block.max(dim=-1, keepdim=True).values
    max_lv2 = abs_block.view(m, g, 2, group_size // 2).amax(dim=-1, keepdim=True)
    max_lv3 = abs_block.view(m, g, 4, group_size // 4).amax(dim=-1, keepdim=True)

    scale_factor = (max_lv1 / 7.0).clamp(min=2 ** (-48), max=49152)
    e_sf = torch.floor(torch.log2(scale_factor))
    scale_factor = torch.round(scale_factor * torch.exp2(2 - e_sf)) * torch.exp2(e_sf - 2)

    rec_sf = (1.0 / scale_factor).to(torch.bfloat16).to(torch.float32)
    rec_sf_ex = rec_sf.unsqueeze(2)
    scale_lv2 = torch.exp2(((max_lv2 * rec_sf_ex).clamp(0, 4) / 4).floor())
    scale_lv2_for_lv3 = scale_lv2.repeat_interleave(2, dim=2)
    scale_lv3 = torch.exp2(((max_lv3 * rec_sf_ex / scale_lv2_for_lv3).clamp(0, 2) / 2).floor())

    scale_lv2_b = scale_lv2.repeat_interleave(group_size // 2, dim=-1).view(m, g, group_size)
    scale_lv3_b = scale_lv3.repeat_interleave(group_size // 4, dim=-1).view(m, g, group_size)
    scale_factor_b = scale_factor.expand_as(scale_lv2_b)

    norm = block.abs() / (scale_factor_b * scale_lv2_b * scale_lv3_b + 1e-12)
    mant = torch.floor(norm * 4 + 0.5) / 4
    mant = mant.clamp(max=1.75)
    q = torch.sign(block) * mant

    scale_lv2_pack = scale_lv2.squeeze(-1).repeat_interleave(4, dim=2).unsqueeze(2).unsqueeze(3)
    scale_lv3_pack = scale_lv3.squeeze(-1).repeat_interleave(4, dim=2).reshape(m, g, 2, 8).unsqueeze(3)

    return (
        q.view(m, n).to(torch.bfloat16),
        scale_factor.squeeze(-1).to(torch.float32),
        scale_lv2_pack.to(torch.float32),
        scale_lv3_pack.to(torch.float32),
    )


class HiFp4QuantProcessor(BaseConvertProcessor):
    name = "HiFp4QuantProcessor"
    src_ir = IRKind.FLOAT
    dst_ir = IRKind.HIFP4
    loss_level = LossLevel.LOSSY.value

    def transform(self, module: nn.Module, context: ConvertContext) -> nn.Module:
        linear = _materialize_linear(module, context)
        if linear is None:
            return module
        q_weight, sf, sl2, sl3 = _hif4_quantize_weight(linear.weight.detach(), group_size=64)
        bias = linear.bias.detach() if linear.bias is not None else None
        return W4A8HiFxMXDynamicPerBlockFakeQuantLinear.from_hif4_export_state_dict(
            {
                "weight": q_weight,
                "weight_scale_factor": sf,
                "weight_scale_lv2": sl2,
                "weight_scale_lv3": sl3,
                "bias": bias,
            }
        )
