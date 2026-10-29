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

CPU calibration kernels for the DeepSeek-V4.1-Flash inference graph (no tilelang / native GEMM).
``model.py`` imports ``act_quant`` / ``fp4_act_quant`` / ``fp4_gemm`` / ``fp8_gemm`` /
``hc_split_sinkhorn`` / ``sparse_attn`` from ``kernel``.  The checkpoint's own
``kernel.py`` is implemented on ``tilelang`` native kernels that do not run on the
calibration host; calibration instead executes the source graph with decoded BF16
weights on CPU, so this module provides the eager equivalent of each API the graph
actually reaches:

* ``act_quant`` / ``fp4_act_quant`` — FP8 / FP4 cache quantize-dequantize, group-32.
* ``hc_split_sinkhorn`` — hyper-connection mixer (pre / post / Sinkhorn comb).
* ``sparse_attn`` — chunked sparse window/compressed attention.
* ``rotary`` — in-place RoPE, the eager twin of ``model.apply_rotary_emb``.
* ``fp4_gemm`` / ``fp8_gemm`` — never reached with BF16 weights; fail loudly.

These are calibration-grade stubs: they decode to full BF16/FP32 on CPU and are not
a native low-bit GEMM oracle.
"""

import torch


def act_quant(x, block_size=32, scale_fmt=None, scale_dtype=None, inplace=False):
    """Eager FP8 (e4m3, group-block) activation quantize in place.

    Mirrors the legacy calibration bridge's fp8_qdq. ``scale_fmt='ue8m0'`` rounds
    scales up to a power of two (UE8M0-compatible) like the native kernel.
    """
    if not inplace:
        raise NotImplementedError('Calibration linears use decoded BF16 weights')
    z = x.float().cpu().unflatten(-1, (-1, block_size))
    amax = z.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = amax / 448
    if scale_fmt is not None:
        s = torch.exp2(torch.ceil(torch.log2(s)))
    y = (z / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s
    x.copy_(y.flatten(-2).to(device=x.device, dtype=x.dtype))
    return x


def fp4_act_quant(x, block_size=32, inplace=False):
    """Eager FP4 (e2m1, group-32) activation quantize in place (cache QDQ)."""
    if not inplace:
        raise NotImplementedError('Only cache QDQ is used')
    z = x.float().cpu().unflatten(-1, (-1, block_size))
    s = torch.exp2(torch.ceil(torch.log2(z.abs().amax(-1, keepdim=True).clamp_min(6 * 2**-126) / 6)))
    v = (z / s).abs().clamp_max(6)
    # Midpoint ties go to an even FP4 mantissa code.
    thresholds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    code = (v.unsqueeze(-1) > thresholds).sum(-1)
    for i in (1, 3, 5):
        code += (v == thresholds[i]).long()
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    y = lut[code] * z.sign() * s
    x.copy_(y.flatten(-2).to(device=x.device, dtype=x.dtype))
    return x


def sinkhorn(mixes, scale, base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """Hyper-connection mixer: sigmoid pre, double-sigmoid post, Sinkhorn comb."""
    h = hc_mult
    pre = torch.sigmoid(mixes[..., :h] * scale[0] + base[:h]) + eps
    post = 2 * torch.sigmoid(mixes[..., h : 2 * h] * scale[1] + base[h : 2 * h])
    comb = (mixes[..., 2 * h :] * scale[2] + base[2 * h :]).unflatten(-1, (h, h))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


# ``model.py`` resolves ``hc_split_sinkhorn`` and ``sinkhorn`` from the kernel module.
hc_split_sinkhorn = sinkhorn


def sparse_attn(q, kv, sink, indices, scale):
    """Chunked sparse attention over the compressed KV (eager, CPU-friendly)."""
    out = torch.empty_like(q)
    # Chunk query positions to bound temporary gather/score memory.
    for b in range(q.shape[0]):
        for begin in range(0, q.shape[1], 32):
            end = min(begin + 32, q.shape[1])
            idx = indices[b, begin:end].to(device=q.device, dtype=torch.long)
            maximum = torch.full((end - begin, q.shape[2]), -1e30, device=q.device, dtype=torch.float32)
            denominator = torch.zeros_like(maximum)
            numerator = torch.zeros_like(q[b, begin:end], dtype=torch.float32)
            for k in range(0, idx.shape[-1], 64):
                ids = idx[:, k : k + 64]
                values = kv[b][ids.clamp_min(0)].float()
                logits = torch.einsum('shd,skd->shk', q[b, begin:end].float(), values) * scale
                logits.masked_fill_((ids < 0).unsqueeze(1), -torch.inf)
                updated = torch.maximum(maximum, logits.amax(-1))
                correction = (maximum - updated).exp()
                probs = (logits - updated.unsqueeze(-1)).exp()
                numerator = numerator * correction.unsqueeze(-1) + torch.einsum(
                    'shk,skd->shd', probs.to(q.dtype).float(), values
                )
                denominator = denominator * correction + probs.sum(-1)
                maximum = updated
            denominator += (sink - maximum).exp()
            out[b, begin:end] = (numerator / denominator.unsqueeze(-1)).to(q.dtype)
    return out


def rotary(x, freqs, inverse=False):
    """Eager in-place RoPE (adjacent-pair complex rotation), matching the graph."""
    f = freqs.cpu()
    c, s = f.real.to(x.device), f.imag.to(x.device)
    if inverse:
        s = -s
    if x.ndim == 4:
        c, s = c[:, None], s[:, None]
    a, b = x.float()[..., 0::2], x.float()[..., 1::2]
    x.copy_(torch.stack((a * c - b * s, a * s + b * c), -1).flatten(-2).to(x.dtype))
    return x


def fp8_gemm(*args, **kwargs):
    raise NotImplementedError('Native FP8 GEMM is not used by the calibration kernel; weights are decoded to BF16.')


def fp4_gemm(*args, **kwargs):
    raise NotImplementedError('Native FP4 GEMM is not used by the calibration kernel; weights are decoded to BF16.')
