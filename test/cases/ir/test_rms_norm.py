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

from msmodelslim.ir.rms_norm import RMSNorm


def test_init_sets_unit_weight_and_epsilon():
    module = RMSNorm(4, eps=1e-5)

    assert module.variance_epsilon == 1e-5
    assert torch.equal(module.weight, torch.ones(4))
    assert isinstance(module.weight, torch.nn.Parameter)


def test_forward_normalizes_by_root_mean_square():
    # eps=0: x=[3,4,0,0] -> rms=sqrt(25/4)=2.5 -> x/rms=[1.2,1.6,0,0]
    module = RMSNorm(4, eps=0.0)
    x = torch.tensor([[3.0, 4.0, 0.0, 0.0]], dtype=torch.float32)

    output = module(x)

    assert output.dtype == torch.float32
    assert torch.allclose(output, torch.tensor([[1.2, 1.6, 0.0, 0.0]]), atol=1e-6)


def test_forward_scales_normalized_output_by_weight():
    module = RMSNorm(4, eps=0.0)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([2.0, 0.5, 1.0, 1.0]))
    x = torch.tensor([[3.0, 4.0, 0.0, 0.0]], dtype=torch.float32)

    output = module(x)

    assert torch.allclose(output, torch.tensor([[2.4, 0.8, 0.0, 0.0]]), atol=1e-6)


def test_forward_returns_normalized_constant_input_unchanged():
    module = RMSNorm(4, eps=0.0)
    x = torch.full((2, 4), 2.0, dtype=torch.float32)

    output = module(x)

    assert torch.allclose(output, torch.ones_like(x), atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_forward_rounds_through_weight_dtype_when_weight_is_half(dtype):
    module = RMSNorm(4, eps=0.0).to(dtype)
    x = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)

    output = module(x)

    assert torch.equal(output, torch.tensor([[1.4140625, 1.4140625, 0.0, 0.0]], dtype=torch.float32))
