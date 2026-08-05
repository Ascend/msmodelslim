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

# INT4/INT8 对称仿射伪量化 driver、Context 子类与 factory。

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Generator, Optional

import torch

from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.ir.qal import QDType, QParam, QScope, QScheme

from .base import FakeQuantDriver, FakeQuantStage, TLQKernel
from .context import (
    FakeQuantPipelineContext,
    FakeQuantResult,
)
from .registry import register_tlq_kernel

_SCALE_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

_SYMMETRIC_MAX_BOUND: dict[QDType, int] = {
    QDType.INT4: 2 ** (4 - 1) - 1,
    QDType.INT8: 2 ** (8 - 1) - 1,
}


@dataclass
class IntFakeQuantContext(FakeQuantPipelineContext):
    """INT 对称仿射伪量化流水线状态。"""

    min_val: Optional[torch.Tensor] = None
    max_val: Optional[torch.Tensor] = None
    min_abs: Optional[torch.Tensor] = None
    max_abs: Optional[torch.Tensor] = None


def build_int_context(
    config: QConfig,
    float_tensor: torch.Tensor,
    train_tensors: dict[str, torch.Tensor],
) -> IntFakeQuantContext:
    return IntFakeQuantContext(
        config=config,
        float_tensor=float_tensor,
        train_tensors=dict(train_tensors),
    )


def _parse_int_ext(config: QConfig) -> tuple[float, torch.dtype]:
    ext = config.ext
    q_scale_thresh = float(ext.get("q_scale_thresh", 1e-5))
    scale_dtype_name = str(ext.get("scale_dtype", "bfloat16"))
    if scale_dtype_name not in _SCALE_DTYPES:
        raise ValueError(f"Unsupported scale dtype {scale_dtype_name!r}, expected one of {list(_SCALE_DTYPES)}")
    return q_scale_thresh, _SCALE_DTYPES[scale_dtype_name]


def _default_max_bound(config: QConfig) -> int:
    q_dtype = QDType(config.dtype)
    if q_dtype not in _SYMMETRIC_MAX_BOUND:
        raise TypeError(f"q_dtype {q_dtype} is not supported by int fake-quant driver")
    return _SYMMETRIC_MAX_BOUND[q_dtype]


def int_max_bound(_ctx: IntFakeQuantContext, config: QConfig) -> int:
    return _default_max_bound(config)


def _broadcast_scale_offset(
    x: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if scale.numel() == 1:
        return scale, offset
    if scale.ndim == x.ndim - 1 and scale.shape == x.shape[: scale.ndim]:
        return scale.unsqueeze(-1), offset.unsqueeze(-1)
    if scale.ndim == 1 and x.ndim >= 2 and scale.shape[0] == x.shape[0]:
        view = (-1,) + (1,) * (x.ndim - 1)
        return scale.view(view), offset.view(view)
    view = (-1,) + (1,) * (x.dim() - 1)
    return scale.reshape(view), offset.reshape(view)


def _symmetric_dequantize(q: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    s, o = _broadcast_scale_offset(q, scale, offset)
    return (q - o) * s


def _compute_abs_range(ctx: IntFakeQuantContext) -> None:
    t = ctx.working_tensor
    ctx.min_val = torch.clamp(t.min(-1)[0], max=0)
    ctx.max_val = torch.clamp(t.max(-1)[0], min=0)
    ctx.min_abs = -(ctx.min_val)
    ctx.max_abs = ctx.max_val


def _compute_qparam(
    ctx: IntFakeQuantContext,
    config: QConfig,
    max_bound: int,
    q_scale_thresh: float,
    scale_dtype: torch.dtype,
) -> None:
    if ctx.min_abs is None or ctx.max_abs is None:
        raise RuntimeError("compute_qparam requires min_abs/max_abs")
    eps = torch.tensor([torch.finfo(torch.float32).eps]).type_as(ctx.min_abs)
    max_v = torch.max(ctx.min_abs, ctx.max_abs)
    scale = torch.max(max_v / float(max_bound), eps)
    scale = torch.where(
        scale < 0,
        torch.clamp(scale, max=-q_scale_thresh),
        torch.clamp(scale, min=q_scale_thresh),
    )
    scale = scale.to(scale_dtype)
    offset = torch.zeros_like(scale)
    ctx.q_param = QParam(
        scheme=QScheme(
            dtype=QDType(config.dtype),
            scope=QScope(config.scope),
            symmetric=config.symmetric,
        ),
        ext={"scale": scale, "offset": offset},
    )


def _affine_normalize(ctx: IntFakeQuantContext) -> None:
    if ctx.q_param is None:
        raise RuntimeError("affine_normalize requires q_param")
    x = ctx.working_tensor
    scale = ctx.q_param.ext["scale"]
    offset = ctx.q_param.ext.get("offset", torch.zeros_like(scale))
    s, o = _broadcast_scale_offset(x, scale, offset)
    ctx.normed = x / s + o


def _round_to_integer(ctx: IntFakeQuantContext, max_bound: int) -> None:
    if ctx.quantized is not None:
        return
    if ctx.normed is None:
        raise RuntimeError("round_to_integer requires normed")
    if ctx.q_param is None:
        raise RuntimeError("round_to_integer requires q_param")
    min_bound = -max_bound - 1
    quantized = ctx.normed.round().clamp(min=min_bound, max=max_bound)
    ctx.quantized = quantized


def _finalize_output(ctx: IntFakeQuantContext, return_quantized_weight: bool) -> torch.Tensor:
    if ctx.q_param is None:
        raise RuntimeError("fake quantize finalize requires q_param")
    if ctx.quantized is None:
        raise RuntimeError("finalize requires quantized")
    if return_quantized_weight:
        return ctx.quantized
    scale = ctx.q_param.ext["scale"]
    offset = ctx.q_param.ext.get("offset", torch.zeros_like(scale))
    return _symmetric_dequantize(ctx.quantized, scale, offset)


def int_fake_quant_driver(
    ctx: IntFakeQuantContext,
    config: QConfig,
    default_max_bound: int,
    q_scale_thresh: float,
    scale_dtype: torch.dtype,
    return_quantized_weight: bool = False,
) -> Generator[FakeQuantStage, None, FakeQuantResult]:
    max_bound = default_max_bound

    yield FakeQuantStage.QUANT_INPUT

    _compute_abs_range(ctx)
    yield FakeQuantStage.BEFORE_QPARAM

    _compute_qparam(
        ctx,
        config=config,
        max_bound=max_bound,
        q_scale_thresh=q_scale_thresh,
        scale_dtype=scale_dtype,
    )
    yield FakeQuantStage.QPARAM_READY

    _affine_normalize(ctx)
    yield FakeQuantStage.PRE_ROUND

    _round_to_integer(ctx, max_bound)
    yield FakeQuantStage.QUANTIZED

    out = _finalize_output(ctx, return_quantized_weight=return_quantized_weight)
    if not return_quantized_weight:
        out = out.to(ctx.float_tensor.dtype)
    yield FakeQuantStage.DEQUANT_OUTPUT
    return FakeQuantResult(
        tensor=out,
        q_param=ctx.q_param,
        quantized=ctx.quantized,
    )


@register_tlq_kernel(
    (QDType.INT4, QDType.INT4, QScope.PER_TOKEN, True),
    (QDType.INT4, QDType.INT4, QScope.PER_CHANNEL, True),
    (QDType.INT8, QDType.INT8, QScope.PER_CHANNEL, True),
    (QDType.INT8, QDType.INT8, QScope.PER_TOKEN, True),
)
def create_int_kernel(config: QConfig) -> TLQKernel:
    q_scale_thresh, scale_dtype = _parse_int_ext(config)
    default_max_bound = _default_max_bound(config)
    driver: FakeQuantDriver = partial(
        int_fake_quant_driver,
        config=config,
        default_max_bound=default_max_bound,
        q_scale_thresh=q_scale_thresh,
        scale_dtype=scale_dtype,
    )

    return TLQKernel(config, driver, build_int_context)
