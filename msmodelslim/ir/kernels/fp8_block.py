#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
FP8 per-block checkpoint kernels (E4M3 weight + block ``weight_scale_inv``).

Used by ``processor/convert`` (FP8_BLOCK → FLOAT) and legacy ``model/*/convert_fp8_to_bf16`` scripts.
"""

from __future__ import annotations

import torch

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
