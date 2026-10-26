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

import torch
import torch.nn.functional as F

from msmodelslim.ir.w16a16s import W16A16sLinear


def test_forward_matches_linear_when_bias_given():
    torch.manual_seed(0)
    weight = torch.randn(3, 5, dtype=torch.float32)
    bias = torch.randn(3, dtype=torch.float32)
    module = W16A16sLinear(weight, bias)
    x = torch.randn(2, 5, dtype=torch.float32)

    output = module(x)

    assert output.shape == (2, 3)
    assert torch.allclose(output, F.linear(x, weight, bias), atol=1e-6)


def test_forward_matches_linear_when_bias_is_none():
    torch.manual_seed(1)
    weight = torch.randn(3, 5, dtype=torch.float32)
    module = W16A16sLinear(weight, None)
    x = torch.randn(4, 5, dtype=torch.float32)

    output = module(x)

    assert module.bias is None
    assert torch.allclose(output, F.linear(x, weight), atol=1e-6)


def test_parameters_are_frozen():
    module = W16A16sLinear(torch.randn(3, 5, dtype=torch.float32), torch.randn(3, dtype=torch.float32))

    assert isinstance(module.weight, torch.nn.Parameter)
    assert module.weight.requires_grad is False
    assert module.bias.requires_grad is False
