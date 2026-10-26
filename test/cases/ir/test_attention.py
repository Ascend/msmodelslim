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

from msmodelslim.ir.attention import FakeQuantDynamicCache
from msmodelslim.ir.const import int8_per_channel_sym
from msmodelslim.ir.qal import QParam


def _make_cache(scale, offset=None):
    if offset is None:
        offset = torch.zeros_like(scale)
    q_param = QParam(scheme=int8_per_channel_sym, ext={"scale": scale, "offset": offset})
    return FakeQuantDynamicCache(q_param)


def test_is_atomic_returns_true_for_dynamic_cache():
    assert FakeQuantDynamicCache.is_atomic() is True


def test_init_keeps_scale_and_offset_as_frozen_parameters():
    scale = torch.full((2,), 0.5, dtype=torch.float32)
    offset = torch.zeros(2, dtype=torch.float32)

    module = _make_cache(scale, offset)

    assert isinstance(module.kv_cache_scale, torch.nn.Parameter)
    assert module.kv_cache_scale.requires_grad is False
    assert module.kv_cache_offset.requires_grad is False
    assert torch.equal(module.kv_cache_scale, scale)
    assert torch.equal(module.kv_cache_offset, offset)


def test_forward_preserves_shape_and_dtype_of_input():
    module = _make_cache(torch.full((2,), 0.5, dtype=torch.float32))
    x = torch.ones(1, 1, 2, 2, dtype=torch.float32)

    output = module(x)

    assert output.shape == x.shape
    assert output.dtype == torch.float32


def test_forward_quantizes_values_with_configured_scale():
    # scale=0.5: value 1.0/0.5=2 -> round 2 -> 2*0.5 = 1.0
    module = _make_cache(torch.full((2,), 0.5, dtype=torch.float32))
    x = torch.ones(1, 1, 2, 2, dtype=torch.float32)

    output = module(x)

    assert torch.allclose(output, torch.ones_like(x))


def test_forward_clamps_small_values_to_zero_with_large_scale():
    # scale=3.0: value 1.0/3.0=0.33 -> round 0 -> 0.0
    module = _make_cache(torch.full((2,), 3.0, dtype=torch.float32))
    x = torch.ones(1, 1, 2, 2, dtype=torch.float32)

    output = module(x)

    assert torch.equal(output, torch.zeros_like(x))


def test_forward_rounds_value_up_to_single_step_for_unit_scale():
    # scale=1.0: value 0.6 -> round 1 -> 1.0
    module = _make_cache(torch.ones(2, dtype=torch.float32))
    x = torch.full((1, 1, 2, 2), 0.6, dtype=torch.float32)

    output = module(x)

    assert torch.allclose(output, torch.ones_like(x))


def test_forward_raises_when_scale_size_does_not_match_input_channels():
    module = _make_cache(torch.ones(3, dtype=torch.float32))
    x = torch.ones(1, 1, 2, 2, dtype=torch.float32)

    with pytest.raises(RuntimeError):
        _ = module(x)
