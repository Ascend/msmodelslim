#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------

Unit tests for msmodelslim.format.common.pack.
"""

from __future__ import annotations

import pytest
import torch

from msmodelslim.format.common.pack import _pack_int4, process_scale, w4a8_pack_int4


class TestPackInt4:
    """Tests for _pack_int4."""

    def test_pack_int4_2d_when_n_even(self):
        weight = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int8)

        packed = _pack_int4(weight)

        assert packed.shape == (2, 2)
        assert packed.dtype == torch.int8

    def test_pack_int4_3d_when_expert_weights(self):
        weight = torch.arange(16, dtype=torch.int8).reshape(2, 2, 4)

        packed = _pack_int4(weight)

        assert packed.shape == (2, 2, 2)

    def test_pack_int4_raise_value_error_when_shape_invalid(self):
        with pytest.raises(ValueError, match="Unexpected weight shape"):
            _pack_int4(torch.zeros(4, dtype=torch.int8))

    def test_pack_int4_raise_assertion_error_when_n_odd(self):
        weight = torch.zeros(2, 3, dtype=torch.int8)

        with pytest.raises(AssertionError, match="n dimension should be even"):
            _pack_int4(weight)


class TestW4A8PackInt4:
    """Tests for w4a8_pack_int4."""

    def test_w4a8_pack_int4_transpose_and_pack_when_valid(self):
        weight = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int8)

        packed = w4a8_pack_int4(weight)

        assert packed.dtype == torch.int8
        assert packed.numel() == weight.numel() // 2


class TestProcessScale:
    """Tests for process_scale."""

    def test_process_scale_up_proj_branch_when_name_contains_up_proj(self):
        bias = torch.ones(2, 4)

        result = process_scale("model.layers.0.mlp.up_proj", bias, tp_num=1)

        assert result.shape == (2, 1)
        assert torch.allclose(result, torch.tensor([[32.0], [32.0]]))

    def test_process_scale_down_proj_branch_when_name_contains_down_proj(self):
        bias = torch.ones(4, 8)

        result = process_scale("model.layers.0.mlp.down_proj", bias, tp_num=2)

        assert result.shape == (4, 2)

    def test_process_scale_return_unchanged_when_name_unmatched(self):
        bias = torch.ones(2, 4)

        result = process_scale("model.layers.0.other", bias, tp_num=1)

        assert torch.equal(result, bias)
