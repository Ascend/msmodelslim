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

from msmodelslim.ir.norm_bias import RMSNormBias


def test_init_sets_unit_weight_zero_bias_and_epsilon():
    module = RMSNormBias(4, eps=1e-5)

    assert module.variance_epsilon == 1e-5
    assert torch.equal(module.weight, torch.ones(4))
    assert torch.equal(module.bias, torch.zeros(4))
    assert isinstance(module.weight, torch.nn.Parameter)
    assert isinstance(module.bias, torch.nn.Parameter)


def test_forward_normalizes_by_root_mean_square():
    # eps=0: x=[3,4,0,0] -> rms=sqrt(25/4)=2.5 -> x/rms=[1.2,1.6,0,0]
    module = RMSNormBias(4, eps=0.0)
    x = torch.tensor([[3.0, 4.0, 0.0, 0.0]], dtype=torch.float32)

    output = module(x)

    assert torch.allclose(output, torch.tensor([[1.2, 1.6, 0.0, 0.0]]), atol=1e-6)


def test_forward_applies_weight_and_bias_after_normalization():
    module = RMSNormBias(4, eps=0.0)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        module.bias.copy_(torch.full((4,), 0.5))
    x = torch.tensor([[3.0, 4.0, 0.0, 0.0]], dtype=torch.float32)

    output = module(x)

    assert torch.allclose(output, torch.tensor([[1.7, 3.7, 0.5, 0.5]]), atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_forward_returns_half_dtype_when_parameters_are_half(dtype):
    module = RMSNormBias(4, eps=0.0).to(dtype)

    output = module(torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32))

    assert output.dtype == dtype
