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

AscendV1 W8A8_MXFP8 decoder. Inverse of AscendV1Saver.on_w8a8_mx_dynamic_per_block.

  disk weight       : float8_e4m3fn  → IR weight: bfloat16 (MX E4M3 grid)
  disk weight_scale : (scale.squeeze(w_axes) + 127).to(uint8)
                      → IR scale: bfloat16 shared exponent
  disk bias         : FLOAT if present → IR bias: bfloat16
  no weight_offset on disk → zeros_like(decoded scale)
"""

from typing import Tuple, Type

import torch
from torch import nn

from msmodelslim.format.common.ir_builder.w8a8_mx_dynamic import W8A8MXDynamicParamSet, W8A8MXDynamicStructureSpec
from msmodelslim.format.common.tensor_utils import require_keys
from msmodelslim.ir.w8a8_mx_dynamic import W8A8MXDynamicPerBlockFakeQuantLinear
from msmodelslim.utils.exception import SchemaValidateError

from .base import LinearIrDecoder

MXFP8_E8M0_UINT8_BIAS = 127
DEFAULT_MX_AXES = -1

_LABEL = "W8A8_MXFP8"
_REQUIRED = ("weight", "weight_scale")
_MISSING_ACTION = "Please ensure the export was produced by AscendV1Saver.on_w8a8_mx_dynamic_per_block."


def _unsqueeze_shape_at(shape: Tuple[int, ...], dim: int) -> Tuple[int, ...]:
    dims = list(shape)
    insert_at = dim if dim >= 0 else len(dims) + 1 + dim
    dims.insert(insert_at, 1)
    return tuple(dims)


def decode_mxfp8_weight_scale(stored: torch.Tensor, w_axes: int = DEFAULT_MX_AXES) -> torch.Tensor:
    """Invert save: ``(scale.squeeze(w_axes) + 127).to(uint8)``."""
    if not isinstance(w_axes, int):
        raise SchemaValidateError(
            f"MXFP8 w_axes must be int, got {type(w_axes).__name__}",
            action="Product W8A8_MXFP8 Linear uses w_axes=-1.",
        )
    scale = stored.to(dtype=torch.float32) - MXFP8_E8M0_UINT8_BIAS
    return scale.unsqueeze(w_axes)


class W8A8MxDecoder(LinearIrDecoder):
    """Decode AscendV1 W8A8_MXFP8 tensors into IR-semantic DTOs."""

    label = _LABEL
    w_axes = DEFAULT_MX_AXES

    def ir_type(self, group_size: int = 0) -> Type[nn.Module]:
        del group_size
        return W8A8MXDynamicPerBlockFakeQuantLinear

    def structure(self, prefix: str, group_size: int = 0) -> W8A8MXDynamicStructureSpec:
        del group_size
        store = self.store
        require_keys(store, prefix, _REQUIRED, _LABEL, _MISSING_ACTION)
        axes = self.w_axes
        bias_shape = store.get_shape(f"{prefix}.bias") if store.has(f"{prefix}.bias") else None
        return W8A8MXDynamicStructureSpec(
            weight_shape=store.get_shape(f"{prefix}.weight"),
            weight_scale_shape=_unsqueeze_shape_at(store.get_shape(f"{prefix}.weight_scale"), axes),
            w_axes=axes,
            bias_shape=bias_shape,
        )

    def params(self, prefix: str, device: torch.device, group_size: int = 0) -> W8A8MXDynamicParamSet:
        del group_size
        store = self.store
        require_keys(store, prefix, _REQUIRED, _LABEL, _MISSING_ACTION)
        axes = self.w_axes
        weight = store.get(f"{prefix}.weight").to(device=device, dtype=torch.bfloat16)
        weight_scale = decode_mxfp8_weight_scale(store.get(f"{prefix}.weight_scale"), axes).to(
            device=device, dtype=torch.bfloat16
        )
        weight_offset = torch.zeros_like(weight_scale)
        bias = None
        if store.has(f"{prefix}.bias"):
            bias = store.get(f"{prefix}.bias").to(device=device, dtype=torch.bfloat16)

        return W8A8MXDynamicParamSet(
            weight=weight,
            weight_scale=weight_scale,
            weight_offset=weight_offset,
            bias=bias,
            w_axes=axes,
        )
