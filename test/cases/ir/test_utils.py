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

from msmodelslim.ir.utils import reshape_to_blocks, undo_reshape_to_blocks
from msmodelslim.utils.exception import SchemaValidateError


def test_reshape_to_blocks_splits_axis_into_tiles():
    a = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    reshaped, axes, orig_shape, padded_shape = reshape_to_blocks(a, [1], 2)

    assert reshaped.shape == (2, 2, 2)
    assert axes == [1]
    assert list(orig_shape) == [2, 4, 1]
    assert list(padded_shape) == [2, 4, 1]
    assert torch.equal(reshaped.reshape(2, 4), a)


def test_reshape_and_undo_round_trip_for_divisible_shape():
    a = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    reshaped, axes, orig_shape, padded_shape = reshape_to_blocks(a, [2], 2)
    restored = undo_reshape_to_blocks(reshaped.clone(), padded_shape, orig_shape, axes)

    assert torch.equal(restored, a)


def test_reshape_pads_and_undo_restores_non_divisible_shape():
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    reshaped, axes, orig_shape, padded_shape = reshape_to_blocks(a, [1], 2)

    assert reshaped.shape == (2, 2, 2)
    assert list(padded_shape) == [2, 4, 1]
    assert torch.equal(undo_reshape_to_blocks(reshaped.clone(), padded_shape, orig_shape, axes), a)


def test_reshape_normalizes_negative_axis():
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    reshaped, axes, _, _ = reshape_to_blocks(a, [-1], 4)

    assert axes == [1]
    assert reshaped.shape == (2, 1, 4)


def test_reshape_raises_when_axes_missing():
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    with pytest.raises(SchemaValidateError):
        reshape_to_blocks(a, None, 2)


@pytest.mark.parametrize("block_size", [0, -1])
def test_reshape_raises_when_block_size_not_positive(block_size):
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    with pytest.raises(SchemaValidateError):
        reshape_to_blocks(a, [1], block_size)
