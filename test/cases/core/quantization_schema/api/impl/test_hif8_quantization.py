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
"""

import math
from unittest.mock import patch

import pytest
import torch

from msmodelslim.core.quantization_schema.api.impl import hif8_quantization as hif8
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.core.quantization_schema.api import calculate_qparam, quantize, dequantize, fake_quantize
from msmodelslim.core.quantization_schema.const import hif8_per_tensor_sym, hif8_per_head_sym
from msmodelslim.core.quantization_schema.qal import QDType, QParam, QScope, QStorage


def _decode_byte(code):
    """Independent oracle: decode the six HiF8 bit prefixes, not the production grid rule."""
    if code == 0x80:
        return math.nan
    sign = -1 if code & 0x80 else 1
    payload = code & 0x7F
    if payload == 0:
        return 0.0
    if payload == 0x6F:
        return sign * math.inf
    if payload < 0x08:  # S0000abc: nonzero abc encodes a power of two.
        return sign * 2.0 ** (payload - 23)
    if payload < 0x10:  # S0001abc: exponent zero, three fraction bits.
        exponent, bits = 0, 3
    elif payload < 0x20:  # S001tabc: exponent +/-1.
        exponent, bits = -1 if payload & 0x08 else 1, 3
    elif payload < 0x40:  # S01tBabc: exponent +/-(2+B).
        exponent, bits = 2 + ((payload >> 3) & 1), 3
        exponent *= -1 if payload & 0x10 else 1
    elif payload < 0x60:  # S10tABab: exponent +/-(4+2A+B).
        exponent, bits = 4 + ((payload >> 2) & 3), 2
        exponent *= -1 if payload & 0x10 else 1
    else:  # S11tABCa: exponent +/-(8+4A+2B+C).
        exponent, bits = 8 + ((payload >> 1) & 7), 1
        exponent *= -1 if payload & 0x10 else 1
    fraction = payload & ((1 << bits) - 1)
    return sign * math.ldexp(1.0 + fraction / (1 << bits), exponent)


def _nonnegative_levels():
    return tuple(sorted(value for code in range(128) if math.isfinite(value := _decode_byte(code))))


def _assert_exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


def test_hif8_grid_matches_reference_when_all_byte_codes_are_enumerated():
    levels = _nonnegative_levels()
    assert len(levels) == 127
    assert len(set(levels)) == len(levels)
    assert levels[0] == 0.0
    assert levels[1] == 2.0**-22
    assert levels[-1] == 32768.0
    assert hif8._HIF8_LEVELS == levels
    midpoints = tuple((low + high) / 2 for low, high in zip(levels, levels[1:]))
    assert hif8._HIF8_BOUNDARIES == midpoints + (40960.0,)


def test_hif8_round_preserves_values_when_all_byte_codes_are_decoded():
    expected = torch.tensor([_decode_byte(code) for code in range(256)], dtype=torch.float32)
    _assert_exact(hif8.hif8_round(expected), expected)


@pytest.mark.parametrize("sign", [1, -1])
def test_hif8_round_chooses_expected_neighbor_when_values_are_near_midpoints(sign):
    levels = torch.tensor(_nonnegative_levels(), dtype=torch.float32)
    midpoints = (levels[:-1] + levels[1:]) / 2
    below = torch.nextafter(midpoints, torch.full_like(midpoints, -math.inf))
    above = torch.nextafter(midpoints, torch.full_like(midpoints, math.inf))
    values = sign * torch.cat((below, midpoints, above))
    expected = sign * torch.cat((levels[:-1], levels[1:], levels[1:]))
    _assert_exact(hif8.hif8_round(values), expected)


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize(
    "value, expected",
    [
        (1.06, 1.0),
        (1.07, 1.125),
        (1.20, 1.25),  # Three fraction bits, exponent zero.
        (25.0, 24.0),
        (26.0, 28.0),
        (27.0, 28.0),  # Two fraction bits, exponent +4.
        (300.0, 256.0),
        (320.0, 384.0),
        (350.0, 384.0),  # One fraction bit, exponent +8.
        (0.07, 0.0625),
        (0.071, 0.078125),  # Two fraction bits, exponent -4.
        (0.004, 2.0**-8),
        (0.005, 1.5 * 2.0**-8),  # One fraction bit, exponent -8.
    ],
)
def test_hif8_round_matches_expected_values_when_precision_region_changes(value, expected, sign):
    _assert_exact(hif8.hif8_round(torch.tensor(sign * value)), torch.tensor(sign * expected))


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize(
    "value, expected",
    [
        (2.0**-23, 2.0**-22),  # Midpoint between zero and the smallest subnormal.
        (2.0**-22, 2.0**-22),
        (2.0**-16, 2.0**-16),  # Largest subnormal.
        (3.0 * 2.0**-17, 2.0**-15),  # Subnormal/normal transition midpoint.
        (2.0**-15, 2.0**-15),
        (1.25 * 2.0**-15, 1.5 * 2.0**-15),
        (32768.0, 32768.0),
    ],
)
def test_hif8_round_matches_expected_values_when_input_is_at_format_boundary(value, expected, sign):
    _assert_exact(hif8.hif8_round(torch.tensor(sign * value)), torch.tensor(sign * expected))


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("saturate", [False, True])
def test_hif8_round_obeys_saturation_when_values_reach_overflow_threshold(sign, saturate):
    threshold = torch.tensor(40960.0)
    values = sign * torch.stack(
        (
            torch.tensor(32768.0),
            torch.nextafter(threshold, torch.tensor(-math.inf)),
            threshold,
            torch.nextafter(threshold, torch.tensor(math.inf)),
            torch.tensor(math.inf),
        )
    )
    overflow = 32768.0 if saturate else math.inf
    expected = sign * torch.tensor([32768.0, 32768.0, overflow, overflow, overflow])
    _assert_exact(hif8.hif8_round(values, saturate=saturate), expected)


@pytest.mark.parametrize("saturate", [False, True])
@pytest.mark.parametrize("nan_to_zero", [False, True])
def test_hif8_round_obeys_options_when_values_include_nan_infinity_or_zero(saturate, nan_to_zero):
    values = torch.tensor([0.0, -0.0, 2.0**-24, -(2.0**-24), math.nan, math.inf, -math.inf])
    overflow = 32768.0 if saturate else math.inf
    expected = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0 if nan_to_zero else math.nan, overflow, -overflow])
    actual = hif8.hif8_round(values, saturate=saturate, nan_to_zero=nan_to_zero)
    _assert_exact(actual, expected)
    assert not torch.signbit(actual[actual == 0]).any()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_hif8_round_preserves_input_when_converting_floating_dtypes_to_fp32(dtype):
    values = torch.tensor([1.07, -26.0, 350.0], dtype=dtype, requires_grad=True)
    original = values.detach().clone()
    actual = hif8.hif8_round(values)
    assert actual.dtype == torch.float32
    assert actual.device == values.device
    assert not actual.requires_grad
    _assert_exact(actual, torch.tensor([1.125, -28.0, 384.0]))
    _assert_exact(values.detach(), original)


def test_hif8_round_preserves_results_when_input_is_empty_scalar_or_noncontiguous():
    empty = torch.empty(2, 0, dtype=torch.float16)
    result = hif8.hif8_round(empty)
    assert result.shape == empty.shape
    assert result.dtype == torch.float32
    _assert_exact(hif8.hif8_round(torch.tensor(26.0)), torch.tensor(28.0))
    values = torch.tensor([[1.07, 25.0], [-26.0, 350.0]]).T
    assert not values.is_contiguous()
    expected = torch.tensor([[1.125, -28.0], [24.0, 384.0]])
    _assert_exact(hif8.hif8_round(values), expected)


def test_hif8_round_reuses_cache_when_initialized_in_inference_mode():
    hif8._hif8_lookup_table.cache_clear()
    try:
        with torch.inference_mode():
            first = hif8.hif8_round(torch.tensor([26.0]))
        with torch.no_grad():
            second = hif8.hif8_round(torch.tensor([26.0]))
        _assert_exact(first, torch.tensor([28.0]))
        _assert_exact(second, first)
        boundaries, mapped = hif8._hif8_lookup_table("cpu", False)
        assert not torch.is_inference(boundaries)
        assert not torch.is_inference(mapped)
    finally:
        hif8._hif8_lookup_table.cache_clear()


@pytest.mark.parametrize("value", [[1.0], torch.tensor([1]), torch.tensor([1 + 2j])])
def test_hif8_round_raises_error_when_input_is_not_floating(value):
    with pytest.raises(TypeError, match="floating-point"):
        hif8.hif8_round(value)


@pytest.mark.parametrize(
    "options",
    [
        {"saturate": 1},
        {"saturate": "false"},
        {"nan_to_zero": 0},
        {"nan_to_zero": None},
    ],
)
def test_hif8_round_raises_error_when_conversion_flags_are_not_booleans(options):
    with pytest.raises(SchemaValidateError, match="must be booleans"):
        hif8.hif8_round(torch.ones(1), **options)


@pytest.mark.parametrize("scale", [1.0, 2.0, 3.0, 0.3, torch.tensor([3.0]), torch.tensor(3.0)])
def test_hif8_qal_applies_scale_once_when_quantizing_and_dequantizing(scale):
    param = QParam(hif8_per_tensor_sym, ext={"scale": scale})
    values = torch.tensor([-104.0, -26.0, 1.07, 25.0, 26.0, 104.0])
    resolved = torch.as_tensor(scale, dtype=torch.float32)
    expected_quantized = hif8.hif8_round(values / resolved.reshape(()))
    quantized = quantize(QStorage(QDType.FLOAT, values), param)
    _assert_exact(quantized.value, expected_quantized)
    expected = expected_quantized * resolved.reshape(())
    _assert_exact(dequantize(quantized, param).value, expected)
    _assert_exact(fake_quantize(QStorage(QDType.FLOAT, values), param).value, expected)


@pytest.mark.parametrize(
    "minimum,maximum,expected",
    [
        (-32768.0, 16384.0, 1.0),
        (-65536.0, 32768.0, 2.0),
        (0.0, 0.0, torch.finfo(torch.float32).tiny),
        (-98304.0, 32768.0, 3.0),
        (0.0, 1e-40, torch.finfo(torch.float32).tiny),
    ],
)
def test_hif8_minmax_resolves_shared_scale_when_bounds_are_finite_and_ordered(minimum, maximum, expected):
    param = calculate_qparam(torch.tensor(minimum), torch.tensor(maximum), QDType.HiF8, QScope.PER_TENSOR, True)
    assert param.ext.keys() == {"scale"}
    assert param.ext["scale"].shape == (1,)
    assert param.ext["scale"].dtype == torch.float32
    _assert_exact(param.ext["scale"], torch.tensor([expected]))


@pytest.mark.parametrize(
    "operation,dtype",
    [
        (quantize, QDType.FLOAT),
        (dequantize, QDType.HiF8),
        (fake_quantize, QDType.FLOAT),
    ],
)
@pytest.mark.parametrize("ext", [None, {}, {"saturate": True}])
def test_hif8_qal_raises_error_when_scale_is_omitted(operation, dtype, ext):
    param = QParam(hif8_per_tensor_sym, ext=ext)
    values = torch.tensor([-26.0, 25.0, 26.0])
    with pytest.raises(TypeError if ext is None else KeyError):
        operation(QStorage(dtype, values), param)


def test_hif8_round_skips_quantization_parameters_when_converting_directly():
    values = torch.tensor([-26.0, 25.0, 26.0])
    with patch.object(hif8.QFuncRegistry, "dispatch", side_effect=AssertionError("unexpected scaled conversion")):
        _assert_exact(hif8.hif8_round(values), torch.tensor([-28.0, 24.0, 28.0]))


def test_hif8_per_head_minmax_resolves_independent_scales_when_head_bounds_differ():
    param = calculate_qparam(
        torch.tensor([-32768.0, -65536.0, 0.0]),
        torch.tensor([16384.0, 98304.0, 0.0]),
        QDType.HiF8,
        QScope.PER_HEAD,
        True,
    )
    assert param.scheme == hif8_per_head_sym
    assert param.ext["scale"].shape == (3,)
    assert param.ext["scale"].dtype == torch.float32
    assert param.ext.keys() == {"scale"}
    _assert_exact(param.ext["scale"], torch.tensor([1.0, 3.0, torch.finfo(torch.float32).tiny]))


def test_hif8_per_head_minmax_clamps_scale_when_nonzero_value_underflows():
    param = calculate_qparam(
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.0, 1e-40]),
        QDType.HiF8,
        QScope.PER_HEAD,
        True,
    )
    _assert_exact(param.ext["scale"], torch.full((2,), torch.finfo(torch.float32).tiny))


def test_hif8_per_head_minmax_keeps_vector_scale_when_there_is_one_head():
    param = calculate_qparam(
        torch.tensor(-65536.0),
        torch.tensor(32768.0),
        QDType.HiF8,
        QScope.PER_HEAD,
        True,
    )
    _assert_exact(param.ext["scale"], torch.tensor([2.0]))
