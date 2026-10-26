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
from torch import nn

# 导入包内所有 IR，触发 QABCRegistry 注册后再使用 create 分派。
import msmodelslim.ir  # noqa: F401
from msmodelslim.ir.auto import (
    AutoFakeQuantActivation,
    AutoFakeQuantDynamicCache,
    AutoFakeQuantLinear,
)
from msmodelslim.ir.const import int8_per_channel_sym, int8_per_token_sym
from msmodelslim.ir.qal import QDType, QParam, QStorage
from msmodelslim.ir.w8a8_dynamic import W8A8DynamicPerChannelFakeQuantLinear


class _AtomicLinear(AutoFakeQuantLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _CompositeLinear(AutoFakeQuantLinear):
    def __init__(self):
        super().__init__()
        self.inner = nn.Linear(2, 2, bias=False)

    @staticmethod
    def is_atomic() -> bool:
        return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)


def test_is_atomic_returns_true_for_base_ir_classes():
    assert AutoFakeQuantLinear.is_atomic() is True
    assert AutoFakeQuantActivation.is_atomic() is True
    assert AutoFakeQuantDynamicCache.is_atomic() is True


def test_named_modules_yields_only_self_when_atomic():
    module = _AtomicLinear()

    entries = list(module.named_modules())

    assert entries == [("", module)]


def test_named_modules_traverses_children_when_not_atomic():
    module = _CompositeLinear()

    names = [name for name, _ in module.named_modules()]

    assert names == ["", "inner"]
    assert dict(module.named_modules())["inner"] is module.inner


def test_create_dispatches_to_registered_fake_quant_linear_by_scheme_pair():
    x_q_param = QParam(scheme=int8_per_token_sym, ext={"scale": torch.ones(1, dtype=torch.float32)})
    w_q_param = QParam(scheme=int8_per_channel_sym, ext={"scale": torch.ones(4, dtype=torch.float32)})
    w_q = QStorage(dtype=QDType.INT8, value=torch.ones(4, 8, dtype=torch.int8))

    module = AutoFakeQuantLinear.create(x_q_param, w_q_param, w_q, None)

    assert isinstance(module, W8A8DynamicPerChannelFakeQuantLinear)
    assert torch.equal(module.weight_scale, torch.ones(4, dtype=torch.float32))
