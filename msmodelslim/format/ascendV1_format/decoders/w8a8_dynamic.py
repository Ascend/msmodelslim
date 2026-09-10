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

AscendV1 W8A8_DYNAMIC decoder. Inverse of AscendV1Saver W8A8_DYNAMIC handlers.
group_size > 0 selects per-group IR; otherwise per-channel.
One decoder covers both IR classes via ``ir_type(group_size)``.
"""

from typing import Type

import torch
from torch import nn

from msmodelslim.format.common.ir_builder.w8a8_dynamic import W8A8DynamicParamSet, W8A8DynamicStructureSpec
from msmodelslim.format.common.tensor_utils import require_keys, squeeze_last_one
from msmodelslim.ir.w8a8_dynamic import (
    W8A8DynamicPerChannelFakeQuantLinear,
    W8A8DynamicPerGroupFakeQuantLinear,
)

from .base import LinearIrDecoder

_LABEL = "W8A8_DYNAMIC"
_REQUIRED = ("weight", "weight_scale")
_MISSING_ACTION = "Please ensure the export was produced by AscendV1Saver W8A8_DYNAMIC handlers."


class W8A8DynamicDecoder(LinearIrDecoder):
    """Decode AscendV1 W8A8_DYNAMIC tensors; per-channel (gs=0) or per-group (gs>0)."""

    label = _LABEL

    def ir_type(self, group_size: int = 0) -> Type[nn.Module]:
        if group_size and group_size > 0:
            return W8A8DynamicPerGroupFakeQuantLinear
        return W8A8DynamicPerChannelFakeQuantLinear

    def structure(self, prefix: str, group_size: int = 0) -> W8A8DynamicStructureSpec:
        store = self.store
        require_keys(store, prefix, _REQUIRED, _LABEL, _MISSING_ACTION)
        scale_shape = store.get_shape(f"{prefix}.weight_scale")
        bias_shape = store.get_shape(f"{prefix}.bias") if store.has(f"{prefix}.bias") else None
        if group_size and group_size > 0:
            offset_shape = (
                store.get_shape(f"{prefix}.weight_offset") if store.has(f"{prefix}.weight_offset") else scale_shape
            )
            return W8A8DynamicStructureSpec(
                weight_shape=store.get_shape(f"{prefix}.weight"),
                weight_scale_shape=scale_shape,
                group_size=group_size,
                weight_offset_shape=offset_shape,
                bias_shape=bias_shape,
            )
        return W8A8DynamicStructureSpec(
            weight_shape=store.get_shape(f"{prefix}.weight"),
            weight_scale_shape=squeeze_last_one(scale_shape),
            group_size=0,
            bias_shape=bias_shape,
        )

    def params(self, prefix: str, device: torch.device, group_size: int = 0) -> W8A8DynamicParamSet:
        store = self.store
        require_keys(store, prefix, _REQUIRED, _LABEL, _MISSING_ACTION)

        weight = store.get(f"{prefix}.weight").to(device=device, dtype=torch.int8)
        weight_scale = store.get(f"{prefix}.weight_scale").to(device=device, dtype=torch.float32)
        weight_offset = None
        offset_key = f"{prefix}.weight_offset"
        if store.has(offset_key):
            weight_offset = store.get(offset_key).to(device=device, dtype=torch.float32)

        bias = None
        if store.has(f"{prefix}.bias"):
            bias = store.get(f"{prefix}.bias").to(device=device, dtype=torch.float32)

        return W8A8DynamicParamSet(
            weight_int8=weight,
            weight_scale=weight_scale,
            weight_offset=weight_offset,
            bias=bias,
            group_size=group_size if group_size and group_size > 0 else 0,
        )
