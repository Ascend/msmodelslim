#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You may use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------

AscendV1 W8A8 (static) decoder. Inverse of AscendV1Saver.on_w8a8_static.
Reads AscendV1 tensors and produces plain-tensor QuantParamSet for IrBuilder.
"""

from typing import Optional, Type

import torch
from torch import nn

from msmodelslim.format.common.deqscale import int64_deqscale_to_float32
from msmodelslim.format.common.ir_builder.w8a8_static import W8A8StaticParamSet, W8A8StaticStructureSpec
from msmodelslim.format.common.tensor_utils import require_keys, squeeze_shape
from msmodelslim.ir.w8a8_static import W8A8StaticFakeQuantLinear

from .base import LinearIrDecoder

_LABEL = "W8A8"
_REQUIRED = ("weight", "input_scale", "input_offset", "deq_scale")
_MISSING_ACTION = "Please ensure the export was produced by AscendV1Saver.on_w8a8_static."


def _reconstruct_bias(
    prefix: str,
    store,
    weight: torch.Tensor,
    input_offset: torch.Tensor,
    deq_scale: torch.Tensor,
    device: torch.device,
) -> Optional[torch.Tensor]:
    bias_key = f"{prefix}.bias"
    if store.has(bias_key):
        return store.get(bias_key).to(device=device, dtype=torch.float32)

    quant_bias_key = f"{prefix}.quant_bias"
    if not store.has(quant_bias_key):
        return None

    quant_bias = store.get(quant_bias_key).to(device=device, dtype=torch.float32)
    offset = input_offset.to(torch.float32)
    if offset.numel() == 1:
        offset = offset.reshape(())
    correction = weight.to(torch.float32).sum(dim=1) * offset
    deq = deq_scale.to(torch.float32)
    if deq.ndim > 1:
        deq = deq.squeeze()
    bias = (quant_bias + correction) * deq
    return bias.to(torch.float32)


class W8A8StaticDecoder(LinearIrDecoder):
    """Decode AscendV1 W8A8 static tensors into IR-semantic DTOs."""

    label = _LABEL

    def ir_type(self, group_size: int = 0) -> Type[nn.Module]:
        del group_size
        return W8A8StaticFakeQuantLinear

    def structure(self, prefix: str, group_size: int = 0) -> W8A8StaticStructureSpec:
        del group_size
        store = self.store
        require_keys(store, prefix, _REQUIRED, _LABEL, _MISSING_ACTION)
        bias_shape = None
        if store.has(f"{prefix}.bias"):
            bias_shape = store.get_shape(f"{prefix}.bias")
        elif store.has(f"{prefix}.quant_bias"):
            bias_shape = store.get_shape(f"{prefix}.quant_bias")
        return W8A8StaticStructureSpec(
            weight_shape=store.get_shape(f"{prefix}.weight"),
            input_scale_shape=store.get_shape(f"{prefix}.input_scale"),
            input_offset_shape=store.get_shape(f"{prefix}.input_offset"),
            weight_scale_shape=squeeze_shape(store.get_shape(f"{prefix}.deq_scale")),
            bias_shape=bias_shape,
        )

    def params(self, prefix: str, device: torch.device, group_size: int = 0) -> W8A8StaticParamSet:
        del group_size
        store = self.store
        require_keys(store, prefix, _REQUIRED, _LABEL, _MISSING_ACTION)

        weight = store.get(f"{prefix}.weight").to(device=device, dtype=torch.int8)
        input_scale = store.get(f"{prefix}.input_scale").to(device=device, dtype=torch.float32)
        input_offset = store.get(f"{prefix}.input_offset").to(device=device, dtype=torch.float32)
        deq_scale = int64_deqscale_to_float32(store.get(f"{prefix}.deq_scale")).to(device=device)

        weight_scale = deq_scale / input_scale
        if weight_scale.ndim > 1:
            weight_scale = weight_scale.squeeze()

        bias = _reconstruct_bias(prefix, store, weight, input_offset, deq_scale, device)

        return W8A8StaticParamSet(
            weight_int8=weight,
            input_scale=input_scale,
            input_offset=input_offset,
            weight_scale=weight_scale,
            bias=bias,
        )
