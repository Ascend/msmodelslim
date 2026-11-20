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

HiFloat8 conversion and scaled quantization in FP32 on the input device.

Uses variable exponent/mantissa precision and TA rounding (ties away from zero),
including power-of-two subnormals and NaN/Inf. The rounding primitive is
scale-free; the API supports shared per-tensor and static per-head scales. Results
are floating representatives of HiF8 values, not packed bytes.
"""

from functools import lru_cache
from typing import Tuple

import torch

from msmodelslim.core.quantization_schema.qal import QDType, QParam, QScope, QScheme, QStorage, QFuncRegistry
from msmodelslim.utils.exception import SchemaValidateError


HIF8_MAX = 32768.0
HIF8_OVERFLOW = 40960.0
HIF8_MIN_SUBNORMAL = 2.0**-22


def _check_float_input(x):
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise TypeError("HiF8 conversion requires a floating-point torch.Tensor")


def _build_hif8_grid() -> Tuple[float, ...]:
    """Enumerate nonnegative finite HiF8 values once, in increasing order."""
    levels = [0.0] + [HIF8_MIN_SUBNORMAL * 2.0**i for i in range(7)]
    for exponent in range(-15, 16):
        bits = 3 if abs(exponent) <= 3 else 2 if abs(exponent) <= 7 else 1
        for mantissa in range(1 << bits):
            value = 2.0**exponent * (1.0 + mantissa / (1 << bits))
            # The +15 exponent / mantissa=1 code is reserved for infinity.
            if value <= HIF8_MAX:
                levels.append(value)
    return tuple(sorted(levels))


_HIF8_LEVELS = _build_hif8_grid()
# TA switches to the larger magnitude at each midpoint, including 0 / 2**-22.
_HIF8_BOUNDARIES = tuple((lower + upper) / 2.0 for lower, upper in zip(_HIF8_LEVELS, _HIF8_LEVELS[1:])) + (
    HIF8_OVERFLOW,
)


@lru_cache(maxsize=32)
def _hif8_lookup_table(device: str, saturate: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cache FP32 boundaries and outputs for each device and overflow policy."""
    overflow = HIF8_MAX if saturate else float("inf")
    # First use may occur inside inference_mode; keep cached tensors ordinary
    # so later no_grad callers can safely reuse the same immutable tables.
    with torch.inference_mode(False), torch.no_grad():
        boundaries = torch.tensor(_HIF8_BOUNDARIES, dtype=torch.float32, device=device)
        mapped = torch.tensor(_HIF8_LEVELS + (overflow,), dtype=torch.float32, device=device)
    return boundaries, mapped


@torch.no_grad()
def hif8_round(x: torch.Tensor, saturate: bool = False, nan_to_zero: bool = False) -> torch.Tensor:
    """Round to the nearest HiF8 value, with ties away from zero.

    With saturate=True, overflow and infinities map to signed 32768.
    With nan_to_zero=True, NaNs map to positive zero.

    Rounded zeros are positive: the negative-zero code is reserved for NaN.
    """
    _check_float_input(x)
    if not isinstance(saturate, bool) or not isinstance(nan_to_zero, bool):
        raise SchemaValidateError("HiF8 saturate and nan_to_zero must be booleans")
    values = x.to(torch.float32)
    if values.numel() == 0:
        return values.clone()
    is_nan = torch.isnan(values)
    # NaN has no ordered interval. Restore it after looking up ordinary values.
    magnitude = torch.where(is_nan, 0.0, values.abs()).contiguous()
    boundaries, mapped = _hif8_lookup_table(str(values.device), saturate)
    indices = torch.bucketize(magnitude, boundaries, right=True)
    rounded = mapped[indices]
    rounded = torch.where(values < 0, -rounded, rounded)
    rounded = torch.where(rounded == 0, 0.0, rounded)
    return torch.where(is_nan, 0.0 if nan_to_zero else float("nan"), rounded)


@QFuncRegistry.register(dispatch_key=(QDType.HiF8, QScope.PER_TENSOR, True), api_name="calculate_qparam")
@QFuncRegistry.register(dispatch_key=(QDType.HiF8, QScope.PER_HEAD, True), api_name="calculate_qparam")
def calculate_hif8_qparam(
    min_val: torch.Tensor, max_val: torch.Tensor, q_dtype: QDType, q_scope: QScope, symmetric: bool, **kwargs
) -> QParam:
    """Calculate a scale from the observed bounds for each quantization group."""
    absmax = torch.maximum(min_val.float().abs(), max_val.float().abs())
    scale = (absmax / HIF8_MAX).clamp_min(torch.finfo(torch.float32).tiny)
    return QParam(
        scheme=QScheme(scope=q_scope, dtype=q_dtype, symmetric=symmetric),
        ext={"scale": torch.atleast_1d(scale)},
    )


@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.HiF8, QScope.PER_TENSOR, True), api_name="quantize")
@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.HiF8, QScope.PER_CHANNEL, True), api_name="quantize")
def hif8_quantize(tensor: QStorage, q_param: QParam) -> QStorage:
    """Round x / scale to floating representatives of HiF8 values."""
    scale = torch.as_tensor(q_param.ext["scale"], dtype=torch.float32, device=tensor.value.device)
    rounded = hif8_round(tensor.value.float() / scale, saturate=True)
    return QStorage(QDType.HiF8, rounded, ext={"input_dtype": tensor.value.dtype})


@QFuncRegistry.register(dispatch_key=(QDType.HiF8, QDType.HiF8, QScope.PER_TENSOR, True), api_name="dequantize")
@QFuncRegistry.register(dispatch_key=(QDType.HiF8, QDType.HiF8, QScope.PER_CHANNEL, True), api_name="dequantize")
def hif8_dequantize(tensor: QStorage, q_param: QParam) -> QStorage:
    """Dequantize in FP32 and restore the input dtype when quantization recorded it."""
    scale = torch.as_tensor(q_param.ext["scale"], dtype=torch.float32, device=tensor.value.device)
    values = tensor.value.float() * scale
    input_dtype = (tensor.ext or {}).get("input_dtype", torch.float32)
    return QStorage(QDType.FLOAT, values.to(input_dtype))
