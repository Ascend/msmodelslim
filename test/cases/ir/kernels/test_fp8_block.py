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

import pytest
import torch

from msmodelslim.ir.kernels.fp8_block import WEIGHT_SCALE_INV_SUFFIX, weight_dequant


def test_weight_scale_inv_suffix_matches_checkpoint_convention():
    assert WEIGHT_SCALE_INV_SUFFIX == ".weight_scale_inv"


def test_weight_dequant_scales_blocks_and_returns_bfloat16():
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    scale = torch.tensor([[2.0]], dtype=torch.float32)

    output = weight_dequant(weight, scale, block_size=2)

    assert output.dtype == torch.bfloat16
    assert output.shape == weight.shape
    assert torch.equal(output, torch.tensor([[2.0, 4.0], [6.0, 8.0]], dtype=torch.bfloat16))


def test_weight_dequant_with_unit_scale_is_identity():
    weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    scale = torch.ones(1, 1, dtype=torch.float32)

    output = weight_dequant(weight, scale, block_size=2)

    assert torch.equal(output, weight.to(torch.bfloat16))


@pytest.mark.parametrize("num_blocks", [1, 2])
def test_weight_dequant_accepts_float8_e4m3_weight(num_blocks):
    block_size = 2
    size = block_size * num_blocks
    weight = torch.ones(size, size, dtype=torch.float8_e4m3fn)
    scale = torch.full((num_blocks, num_blocks), 3.0, dtype=torch.float32)

    output = weight_dequant(weight, scale, block_size=block_size)

    assert output.dtype == torch.bfloat16
    assert output.shape == (size, size)
    assert torch.equal(output, torch.full((size, size), 3.0, dtype=torch.bfloat16))


def test_weight_dequant_supports_non_divisible_shape():
    weight = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    scale = torch.ones(2, 2, dtype=torch.float32)

    output = weight_dequant(weight, scale, block_size=2)

    assert output.shape == (3, 3)
    assert output.dtype == torch.bfloat16
