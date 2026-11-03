#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""FP8 block kernel tests: direct MXFP8 reinterpretation."""

import pytest
import torch

from msmodelslim.ir.kernels.fp8_block import fp8_block_to_mx
from msmodelslim.utils.exception import SchemaValidateError

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "float8_e8m0fnu"),
    reason="torch build has no float8 e4m3/e8m0",
)


def _e8m0(bits: torch.Tensor) -> torch.Tensor:
    return bits.to(torch.uint8).view(torch.float8_e8m0fnu)


def _e4m3(rows: int, cols: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    # 0x7F is the E4M3 NaN code; keep the payload finite so the two dequant paths compare.
    bits = torch.randint(0, 127, (rows, cols), generator=gen, dtype=torch.uint8)
    return bits.view(torch.float8_e4m3fn)


def test_fp8_block_to_mx_keeps_e4m3_bytes_and_matches_block_dequant():
    for rows, cols, scale_shape in ((64, 128, (2, 4)), (128, 128, (1, 1))):
        weight = _e4m3(rows, cols, seed=rows)
        gen = torch.Generator().manual_seed(cols)
        bits = torch.randint(110, 140, scale_shape, generator=gen, dtype=torch.uint8)
        scale = _e8m0(bits)
        out_w, exponent = fp8_block_to_mx(weight, scale)
        assert torch.equal(out_w.view(torch.uint8), weight.view(torch.uint8))
        assert exponent.dtype == torch.float32

        bm, bn = rows // scale_shape[0], cols // scale_shape[1]
        block = weight.to(torch.float32) * scale.to(torch.float32).repeat_interleave(bm, 0).repeat_interleave(bn, 1)
        expanded = exponent.squeeze(-1).to(torch.float32).repeat_interleave(32, dim=1)
        mx = weight.to(torch.float32) * torch.exp2(expanded)
        assert torch.equal(block, mx)

        saved = (exponent.squeeze(-1) + 127).to(torch.uint8)
        row_index = torch.arange(rows) // bm
        group_index = (torch.arange(cols // 32) * 32) // bn
        assert torch.equal(saved, bits[row_index[:, None], group_index[None, :]])


def test_fp8_block_to_mx_rejects_bad_scale():
    weight = _e4m3(32, 64)
    bad = _e8m0(torch.full((1, 2), 0xFF, dtype=torch.uint8))
    with pytest.raises(SchemaValidateError, match="0xFF"):
        fp8_block_to_mx(weight, bad)

    odd_block = _e8m0(torch.full((1, 1), 120, dtype=torch.uint8))
    with pytest.raises(SchemaValidateError, match="multiple of MX"):
        fp8_block_to_mx(_e4m3(32, 40), odd_block)

    not_divisible = _e8m0(torch.full((1, 3), 120, dtype=torch.uint8))
    with pytest.raises(SchemaValidateError, match="not divisible"):
        fp8_block_to_mx(_e4m3(32, 40), not_divisible)
