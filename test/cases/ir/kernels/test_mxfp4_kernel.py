#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""MXFP4 unpack tests. Shape errors are raised before dtype_ops dequantize."""

import torch

import pytest

from msmodelslim.ir.kernels.mxfp4 import e8m0_to_exponent, prepare_mxfp4_dequant, unpack_fp4_uint8
from msmodelslim.utils.exception import SchemaValidateError


def test_unpack_fp4_uint8_matches_for_uint8_and_int8_storage():
    bits = torch.tensor([[0x10, 0x2F], [0x83, 0x06]], dtype=torch.uint8)
    as_int8 = bits.view(torch.int8)
    unpacked = unpack_fp4_uint8(bits)
    assert torch.equal(unpacked, unpack_fp4_uint8(as_int8))
    assert unpacked.shape == (2, 4)
    assert unpacked[0, 0].item() == 0.0
    assert unpacked[0, 1].item() == 0.5
    assert unpacked[0, 2].item() == -6.0
    assert unpacked[0, 3].item() == 1.0


def test_e8m0_to_exponent_matches_uint8_and_float8_storage():
    bits = torch.tensor([[127, 130], [120, 127]], dtype=torch.uint8)
    from_uint8 = e8m0_to_exponent(bits)
    from_e8m0 = e8m0_to_exponent(bits.view(torch.float8_e8m0fnu))
    assert torch.equal(from_uint8, from_e8m0)
    assert torch.equal(from_uint8, bits.to(torch.int32) - 127)


def test_prepare_mxfp4_dequant_rejects_bad_shape_before_dequant():
    packed = torch.zeros(2, 16, dtype=torch.uint8)
    scale = torch.zeros(2, 2, dtype=torch.uint8)
    with pytest.raises(SchemaValidateError, match="scale shape"):
        prepare_mxfp4_dequant(packed, scale)
