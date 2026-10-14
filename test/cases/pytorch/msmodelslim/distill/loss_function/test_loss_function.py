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
# pylint: disable=redefined-outer-name

try:
    from torch.tensor import Tensor
except ModuleNotFoundError:
    from torch import Tensor
import pytest

from ascend_utils.pytorch.knowledge_distill.distill_losses_func_torch import DISTILL_LOSS_FUNC_TORCH


@pytest.fixture()
def tensor_s():
    yield Tensor([[1.0], [1.0]])


@pytest.fixture()
def tensor_t():
    yield Tensor([[1.0], [1.0]])


def test_kd_cross_entropy_given_valid_when_any_then_pass(tensor_s, tensor_t):
    loss_func_kd_cross_entropy = DISTILL_LOSS_FUNC_TORCH["KDCrossEntropy"]()
    loss_func_kd_cross_entropy(tensor_s, tensor_t)
