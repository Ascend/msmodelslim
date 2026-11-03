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

FP8 per-block checkpoint kernels (E4M3 weight + block ``weight_scale_inv``).

Used by ``processor/convert`` (FP8_BLOCK → FLOAT) and legacy ``model/*/convert_fp8_to_bf16`` scripts.
"""

from __future__ import annotations

import torch

from msmodelslim.utils.exception import SchemaValidateError

# Canonical suffix in HuggingFace / DeepSeek-style FP8 checkpoints.
WEIGHT_SCALE_INV_SUFFIX = ".weight_scale_inv"


def weight_dequant(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """
    Dequantize FP8 block weights to bfloat16.

    Args:
        weight: Quantized weight ``(M, N)``.
        scale: Block scale ``(M // block_size, N // block_size)`` (``weight_scale_inv``).
        block_size: Block size from ``quantization_config.weight_block_size`` (default 128).
    """
    m, n = weight.shape
    dev = weight.device
    fp8_on_npu = str(dev).startswith("npu") and weight.dtype == torch.float8_e4m3fn
    if fp8_on_npu:
        # torch_npu 对 fp8 的 dtype 转换（aclnnInplaceCopy / aclnnCast）支持不完整：
        # 先在 CPU 上解码（fp8->fp32 在 CPU 上可靠），结果 bf16 再回原设备，
        # 供后续 mxfp8 量化步骤继续在 NPU 上计算。
        weight = weight.to("cpu").to(torch.float32)
        scale = scale.to("cpu").to(torch.float32)
    else:
        weight = weight.to(torch.float32)
        scale = scale.to(torch.float32)
    scale_expanded = scale.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    scale_expanded = scale_expanded[:m, :n]
    out = (weight * scale_expanded).to(torch.bfloat16)
    return out.to(dev) if fp8_on_npu else out


def fp8_block_to_mx(
    weight: torch.Tensor,
    scale: torch.Tensor,
    mx_block: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reinterpret an aligned E4M3 + E8M0 block as MXFP8.

    The E4M3 payload is kept. The returned scale is the float32 unbiased shared
    exponent with shape ``[N, K / mx_block, 1]``, the same dtype QAL uses for
    ``2**shared_exp``. ``bn`` must be a multiple of ``mx_block`` so one MX group
    never crosses two source blocks.
    """
    if weight.ndim != 2 or scale.ndim != 2:
        raise SchemaValidateError(
            f"fp8_block_to_mx expects 2D weight and scale, got {tuple(weight.shape)} and {tuple(scale.shape)}",
        )
    if scale.dtype != torch.float8_e8m0fnu:
        raise SchemaValidateError(
            f"fp8_block_to_mx scale must be float8_e8m0fnu, got {scale.dtype}",
        )
    rows, cols = weight.shape
    s0, s1 = scale.shape
    if s0 <= 0 or s1 <= 0 or rows % s0 != 0 or cols % s1 != 0:
        raise SchemaValidateError(
            f"FP8 weight {tuple(weight.shape)} is not divisible by scale {tuple(scale.shape)}",
        )
    bn = cols // s1
    if bn % mx_block != 0:
        raise SchemaValidateError(
            f"FP8 K-block {bn} is not a multiple of MX block {mx_block}",
        )
    if cols % mx_block != 0:
        raise SchemaValidateError(f"K={cols} is not a multiple of MX block {mx_block}")

    scale_bits = scale.view(torch.uint8).to(torch.int32)
    if bool((scale_bits == 0xFF).any().item()):
        raise SchemaValidateError("E8M0 scale contains 0xFF, which is not a finite shared exponent")

    bm = rows // s0
    n_groups = cols // mx_block
    row_index = torch.arange(rows, device=scale_bits.device) // bm
    group_index = (torch.arange(n_groups, device=scale_bits.device) * mx_block) // bn
    exponent = scale_bits[row_index[:, None], group_index[None, :]] - 127
    exponent = exponent.unsqueeze(-1).to(torch.float32)

    if str(weight.device).startswith("npu"):
        weight = weight.to("cpu")
        exponent = exponent.to("cpu")
    return weight, exponent
