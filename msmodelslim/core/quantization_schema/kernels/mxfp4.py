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

Unpack MXFP4 checkpoint storage into the inputs of ``mxfp4_dequantize_sym``.

One byte holds two E2M1 values, low nibble first. The E8M0 byte is left as
uint8 storage; dtype_ops decodes the bias and multiplies. This module does
not multiply by the scale.
"""

from __future__ import annotations

import torch

from msmodelslim.utils.exception import SchemaValidateError

# OCP E2M1 codebook. Index is the 4-bit nibble.
_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
_MX_BLOCK = 32


def unpack_fp4_uint8(packed: torch.Tensor) -> torch.Tensor:
    """Unpack ``[N, K/2]`` bytes into float32 ``[N, K]``. Low nibble is first."""
    data = packed.reshape(packed.shape[0], -1).contiguous().view(torch.uint8)
    lut = _E2M1_LUT.to(device=data.device)
    low = lut[(data & 0x0F).long()]
    high = lut[(data >> 4).long()]
    return torch.stack((low, high), dim=-1).reshape(data.shape[0], -1)


def e8m0_to_exponent(scale: torch.Tensor) -> torch.Tensor:
    """E8M0 storage bytes to the unbiased shared exponent."""
    bits = scale.contiguous().view(torch.uint8).to(torch.int32)
    return bits - 127


def prepare_mxfp4_dequant(packed: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Validate shapes and unpack E2M1 into a float32 grid.

    The scale is returned as uint8 E8M0 with shape ``[N, K/32, 1]``, which
    ``mxfp4_dequantize_sym(..., axis=-1)`` broadcasts across each block.
    A mismatched scale is rejected here and is not truncated or padded.
    """
    if packed.ndim != 2 or scale.ndim != 2:
        raise SchemaValidateError(
            f"MXFP4 packed and scale must be 2D, got {tuple(packed.shape)} and {tuple(scale.shape)}",
        )
    rows, packed_cols = packed.shape
    if packed_cols <= 0 or (packed_cols * 2) % _MX_BLOCK != 0:
        raise SchemaValidateError(
            f"MXFP4 packed K/2={packed_cols} does not expand to a multiple of {_MX_BLOCK}",
        )
    groups = (packed_cols * 2) // _MX_BLOCK
    if tuple(scale.shape) != (rows, groups):
        raise SchemaValidateError(
            f"MXFP4 scale shape {tuple(scale.shape)} != ({rows}, {groups}) "
            f"(expected (N, 2 * packed.shape[1] / {_MX_BLOCK}))",
        )
    values = unpack_fp4_uint8(packed)
    shared_exp = scale.contiguous().view(torch.uint8).reshape(rows, groups, 1)
    return values, shared_exp
