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

NVFP4 (ModelOpt) checkpoint kernels.
"""

from __future__ import annotations

import torch

_E2M1_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def unpack_fp4_e2m1(weight_packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 packed FP4(E2M1) into float32 matrix."""
    if weight_packed.dtype is not torch.uint8:
        raise TypeError(f"weight_packed must be uint8, got {weight_packed.dtype}")
    low = (weight_packed & 0x0F).to(torch.long)
    high = ((weight_packed >> 4) & 0x0F).to(torch.long)
    expanded = torch.empty(
        (weight_packed.shape[0], weight_packed.shape[1] * 2),
        dtype=torch.long,
        device=weight_packed.device,
    )
    expanded[:, 0::2] = low
    expanded[:, 1::2] = high
    mag = expanded & 0x7
    sign = (expanded & 0x8) != 0
    table = _E2M1_TABLE.to(weight_packed.device)
    value = table[mag]
    return torch.where(sign, -value, value)


def weight_dequant(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    group_size: int = 16,
) -> torch.Tensor:
    """Dequantize NVFP4 packed checkpoint weight to bf16."""
    value = unpack_fp4_e2m1(weight_packed).to(torch.float32)
    m, n = value.shape
    if weight_scale.ndim != 2 or weight_scale.shape[0] != m:
        raise ValueError(f"weight_scale shape mismatch, expect (M,K), got {tuple(weight_scale.shape)} for M={m}")
    k = weight_scale.shape[1]
    if n != k * group_size:
        raise ValueError(f"unpacked N={n} does not match K*group_size={k * group_size}")
    grouped = value.reshape(m, k, group_size)
    scaled = grouped * weight_scale.to(torch.float32).unsqueeze(-1)
    scaled = scaled * weight_scale_2.to(torch.float32)
    return scaled.reshape(m, n).to(torch.bfloat16)
