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
"""

# MXFP4/MXFP8 per-block 伪量化 driver、Context 子类与 factory。

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Generator, NamedTuple, Optional

import torch
import torch.nn.functional as F

from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.ir.qal import QDType, QParam, QScope, QScheme

from .base import FakeQuantDriver, FakeQuantStage, TLQKernel
from .context import (
    FakeQuantPipelineContext,
    FakeQuantResult,
)
from .registry import register_tlq_kernel


class _MxFormat(NamedTuple):
    ebits: int
    mbits: int
    emax: int
    max_norm: float
    block_size: int
    scale_bits: int


@dataclass
class MxFakeQuantContext(FakeQuantPipelineContext):
    """MXFP per-block 伪量化流水线状态。"""

    block_max: Optional[torch.Tensor] = None
    orig_shape: tuple[int, ...] = ()
    pad_len: int = 0


def build_mx_context(
    config: QConfig,
    float_tensor: torch.Tensor,
    train_tensors: dict[str, torch.Tensor],
) -> MxFakeQuantContext:
    return MxFakeQuantContext(
        config=config,
        float_tensor=float_tensor,
        train_tensors=dict(train_tensors),
    )


def mx_block_size(config: QConfig) -> int:
    q_dtype = QDType(config.dtype)
    return int(q_dtype.mx_finfo.block_size)


def _mx_format(config: QConfig) -> _MxFormat:
    q_dtype = QDType(config.dtype)
    if q_dtype not in (QDType.MXFP4, QDType.MXFP8):
        raise TypeError(f"mxfp driver does not support q_dtype {q_dtype}")
    finfo = q_dtype.mx_finfo
    return _MxFormat(
        ebits=finfo.ebits,
        mbits=finfo.mbits,
        emax=finfo.emax,
        max_norm=float(finfo.max_norm),
        block_size=int(finfo.block_size),
        scale_bits=int(finfo.scale_bits),
    )


_MXFP4_C7_DST_TYPE_MAX = 7.25
_MX_EPS = 9.6e-7
_FP32_MIN_NORMAL = 2 ** (-127 + 1)


def _parse_mx_scale(config: QConfig) -> str:
    """``qconfig.ext["mx_scale"]``: ``base``（默认，对齐 mx_quantization.py）或 ``c7``。"""
    raw = (config.ext or {}).get("mx_scale", "base")
    if raw not in ("base", "c7"):
        raise ValueError(f"Unsupported mx_scale={raw!r}; expected 'base' or 'c7'")
    if raw == "c7" and QDType(config.dtype) != QDType.MXFP4:
        raise ValueError("mx_scale='c7' is only valid for MXFP4")
    return raw


def _floor_ste(x: torch.Tensor) -> torch.Tensor:
    return (x.floor() - x).detach() + x


def _ceil_ste(x: torch.Tensor) -> torch.Tensor:
    return (x.ceil() - x).detach() + x


def _reshape_blocks_last_dim(tensor: torch.Tensor, block_size: int) -> tuple[torch.Tensor, tuple[int, ...], int]:
    orig_shape = tuple(tensor.shape)
    work = tensor
    if work.ndim == 1:
        work = work.unsqueeze(0)
    *batch, n = work.shape
    pad_len = (block_size - n % block_size) % block_size
    if pad_len:
        work = F.pad(work, (0, pad_len))
    n = work.shape[-1]
    blocked = work.reshape(*batch, n // block_size, block_size)
    return blocked, orig_shape, pad_len


def _revert_blocks(blocked: torch.Tensor, orig_shape: tuple[int, ...], pad_len: int) -> torch.Tensor:
    *batch, _n_blocks, block_size = blocked.shape
    flat = blocked.reshape(*batch, -1)
    if pad_len:
        flat = flat[..., :-pad_len]
    if len(orig_shape) == 1:
        return flat.reshape(orig_shape)
    return flat.reshape(orig_shape)


def _block_max_abs(tensor: torch.Tensor, block_size: int) -> tuple[torch.Tensor, tuple[int, ...], int]:
    blocked, orig_shape, pad_len = _reshape_blocks_last_dim(tensor, block_size)
    max_val, _ = torch.max(torch.abs(blocked), dim=-1, keepdim=True)
    return max_val, orig_shape, pad_len


def _quantize_mx_element(
    tensor: torch.Tensor,
    ebits: int,
    mbits: int,
    max_norm: float,
) -> torch.Tensor:
    """元素量化：对齐 mx_quantization._quant（floor(abs + 0.5)）。"""
    bits_ = float(mbits - 2)
    if ebits != 0:
        private_exp = _floor_ste(torch.log2(torch.abs(tensor) + (tensor == 0).type(tensor.dtype)))
        min_exp = -(2 ** (ebits - 1)) + 2
        private_exp = private_exp.clip(min=min_exp)
        tensor = tensor / (2.0 ** private_exp.float()) * (2.0**bits_)
        tensor = torch.sign(tensor) * _floor_ste(torch.abs(tensor) + 0.5)
        tensor = tensor / (2.0**bits_) * (2.0 ** private_exp.float())
    else:
        tensor = tensor * (2.0**bits_)
        tensor = torch.sign(tensor) * _floor_ste(torch.abs(tensor) + 0.5)
        tensor = tensor / (2.0**bits_)
    return torch.clamp(tensor, min=-max_norm, max=max_norm)


def _compute_block_max(ctx: MxFakeQuantContext, block_size: int) -> None:
    max_val, orig_shape, pad_len = _block_max_abs(ctx.working_tensor, block_size)
    ctx.block_max = max_val
    ctx.orig_shape = orig_shape
    ctx.pad_len = pad_len


def _compute_shared_exp(
    ctx: MxFakeQuantContext,
    config: QConfig,
    mx_fmt: _MxFormat,
    mx_scale: str,
) -> None:
    if ctx.block_max is None:
        raise RuntimeError("compute_shared_exp requires block_max")
    max_val = ctx.block_max
    scale_emax = 2.0 ** float(mx_fmt.scale_bits - 1) - 1
    q_dtype = QDType(config.dtype)

    if mx_scale == "c7":
        # MindIE-SD MXFP4 C7：ceil(log2(M / 7.25 + eps))
        shared_exp = _ceil_ste(torch.log2((max_val / _MXFP4_C7_DST_TYPE_MAX).clamp(min=0) + _MX_EPS))
        shared_exp = shared_exp.clamp(
            min=-scale_emax - float(mx_fmt.emax),
            max=scale_emax - float(mx_fmt.emax),
        )
    elif mx_scale == "base":
        if q_dtype == QDType.MXFP4:
            # calculate_mxfp4_qparam：floor(log2(M / 0.875 + eps)) - emax
            man_shift_bit = mx_fmt.mbits - 2
            denom = 1.0 - 0.5 ** (man_shift_bit + 2)
            shared_exp = _floor_ste(torch.log2(max_val / denom + _MX_EPS))
            shared_exp = (shared_exp - float(mx_fmt.emax)).clamp(
                min=-scale_emax - float(mx_fmt.emax),
                max=scale_emax - float(mx_fmt.emax),
            )
        else:
            # calculate_mx_qparam（MXFP8）：floor(log2(M + eps_zero)) - emax
            shared_exp = _floor_ste(torch.log2(max_val + _FP32_MIN_NORMAL * (max_val == 0).to(max_val.dtype)))
            # 训练用 clamp 替代 IR overflow→NaN
            shared_exp = (shared_exp - float(mx_fmt.emax)).clamp(min=-scale_emax, max=scale_emax)
    else:
        raise ValueError(f"Unsupported mx_scale={mx_scale!r}")

    ctx.q_param = QParam(
        scheme=QScheme(
            dtype=q_dtype,
            scope=QScope(config.scope),
            symmetric=config.symmetric,
        ),
        ext={
            "scale": shared_exp,
            "offset": torch.zeros_like(shared_exp),
        },
    )


def _blocked_working_tensor(ctx: MxFakeQuantContext, block_size: int) -> torch.Tensor:
    blocked, _, _ = _reshape_blocks_last_dim(ctx.working_tensor, block_size)
    return blocked


def _affine_normalize_mx(ctx: MxFakeQuantContext, block_size: int, mx_fmt: _MxFormat) -> None:
    """``/ scale`` 与 ``max_norm`` clamp；可训练 ``v`` 由 ``RoundTuneOp`` 在 ``PRE_ROUND`` 注入。"""
    if ctx.q_param is None:
        raise RuntimeError("affine_normalize_mx requires q_param")
    shared_exp = ctx.q_param.ext["scale"]
    scale = torch.pow(2.0, shared_exp.float())
    blocked = _blocked_working_tensor(ctx, block_size)
    normed = blocked / scale
    ctx.normed = torch.clamp(normed, min=-mx_fmt.max_norm, max=mx_fmt.max_norm)


def _quantize_mx_storage(ctx: MxFakeQuantContext, mx_fmt: _MxFormat) -> None:
    if ctx.quantized is not None:
        return
    if ctx.normed is None:
        raise RuntimeError("quantize_mx_storage requires normed")
    if ctx.q_param is None:
        raise RuntimeError("quantize_mx_storage requires q_param")
    ctx.quantized = _quantize_mx_element(
        ctx.normed,
        ebits=mx_fmt.ebits,
        mbits=mx_fmt.mbits,
        max_norm=mx_fmt.max_norm,
    )


def _finalize_mx_output(ctx: MxFakeQuantContext, return_quantized_weight: bool) -> torch.Tensor:
    if ctx.q_param is None or ctx.quantized is None:
        raise RuntimeError("finalize_mx requires q_param and quantized")
    orig_shape = ctx.orig_shape or tuple(ctx.working_tensor.shape)
    if return_quantized_weight:
        return _revert_blocks(ctx.quantized, orig_shape, ctx.pad_len)
    shared_exp = ctx.q_param.ext["scale"]
    scale = torch.pow(2.0, shared_exp.float())
    out_blocked = ctx.quantized * scale
    return _revert_blocks(out_blocked, orig_shape, ctx.pad_len)


def mxfp_fake_quant_driver(
    ctx: MxFakeQuantContext,
    config: QConfig,
    mx_fmt: _MxFormat,
    mx_scale: str,
    return_quantized_weight: bool = False,
) -> Generator[FakeQuantStage, None, FakeQuantResult]:
    yield FakeQuantStage.QUANT_INPUT

    _compute_block_max(ctx, mx_fmt.block_size)
    yield FakeQuantStage.BEFORE_QPARAM

    _compute_shared_exp(ctx, config=config, mx_fmt=mx_fmt, mx_scale=mx_scale)
    yield FakeQuantStage.QPARAM_READY

    _affine_normalize_mx(ctx, block_size=mx_fmt.block_size, mx_fmt=mx_fmt)
    yield FakeQuantStage.PRE_ROUND

    _quantize_mx_storage(ctx, mx_fmt=mx_fmt)
    yield FakeQuantStage.QUANTIZED

    out = _finalize_mx_output(ctx, return_quantized_weight=return_quantized_weight)
    if not return_quantized_weight:
        out = out.to(ctx.float_tensor.dtype)
    yield FakeQuantStage.DEQUANT_OUTPUT
    return FakeQuantResult(
        tensor=out,
        q_param=ctx.q_param,
        quantized=ctx.quantized,
    )


@register_tlq_kernel(
    (QDType.MXFP4, QDType.MXFP4, QScope.PER_BLOCK, True),
    (QDType.MXFP8, QDType.MXFP8, QScope.PER_BLOCK, True),
)
def create_mxfp_kernel(config: QConfig) -> TLQKernel:
    mx_fmt = _mx_format(config)
    mx_scale = _parse_mx_scale(config)
    driver: FakeQuantDriver = partial(
        mxfp_fake_quant_driver,
        config=config,
        mx_fmt=mx_fmt,
        mx_scale=mx_scale,
    )
    return TLQKernel(config, driver, build_mx_context)
