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
from torch import nn


@pytest.fixture
def summary_state() -> dict:
    """注入给 SummaryContext 的承载位置：普通 dict 即可，无需 context。"""
    return {}


@pytest.fixture
def fake_linear_factory():
    """构造一个类名即量化 IR 名、权重按给定 dtype 存储的假线性 IR。"""

    def factory(class_name, out_features, in_features, dtype=torch.int8):
        class _FakeLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(out_features, in_features, dtype=dtype), requires_grad=False)
                self.weight_scale = nn.Parameter(torch.ones(out_features), requires_grad=False)

        _FakeLinear.__name__ = class_name
        return _FakeLinear()

    return factory
