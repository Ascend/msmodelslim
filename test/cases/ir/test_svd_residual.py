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
import torch.nn.functional as F

from msmodelslim.ir.svd_residual import SVDResidualWrapper, SVDResidualHookIR


def test_SVDResidualWrapper_is_atomic_given_no_parameter_when_called_then_return_true():
    assert SVDResidualWrapper.is_atomic() is True


def test_SVDResidualWrapper_forward_given_residual_and_lowrank_when_called_then_equal_original_linear():
    torch.manual_seed(0)
    in_features = 5
    out_features = 3
    rank = 2

    # 构造“真实”权重
    original_weight = torch.randn(out_features, in_features, dtype=torch.float32)

    # 构造低秩分解 U S V^T（与当前实现一致）：
    # svd_lowrank_l1 = V^T [rank, in_features]
    # svd_lowrank_l2 = U*S [out_features, rank]
    svd_lowrank_l1 = torch.randn(rank, in_features, dtype=torch.float32)
    svd_lowrank_l2 = torch.randn(out_features, rank, dtype=torch.float32)
    lowrank_weight = svd_lowrank_l2 @ svd_lowrank_l1

    # 让 wrapped_module 的权重 = 原始权重 - 低秩部分，即 residual
    residual_weight = original_weight - lowrank_weight
    linear = nn.Linear(in_features, out_features, bias=False)
    with torch.no_grad():
        linear.weight.copy_(residual_weight)

    wrapper = SVDResidualWrapper(linear, svd_lowrank_l1, svd_lowrank_l2)

    x = torch.randn(4, in_features, dtype=torch.float32)
    expected = F.linear(x, original_weight, bias=None)
    out = wrapper(x)

    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)


def test_SVDResidualWrapper_forward_given_mismatched_lowrank_shape_when_forward_then_raise_runtime_error():
    in_features = 4
    out_features = 3
    rank = 2

    linear = nn.Linear(in_features, out_features, bias=False)

    # 当前实现语义：left=V^T[rank, in_features], right=U*S[out_features, rank]
    # 这里让 left 的 in_features 维度故意错一维，触发形状不匹配
    svd_lowrank_l1 = torch.randn(rank, in_features + 1, dtype=torch.float32)
    svd_lowrank_l2 = torch.randn(out_features, rank, dtype=torch.float32)

    wrapper = SVDResidualWrapper(linear, svd_lowrank_l1, svd_lowrank_l2)
    x = torch.randn(2, in_features, dtype=torch.float32)

    with pytest.raises(RuntimeError):
        _ = wrapper(x)


def test_SVDResidualWrapper_forward_given_zero_lowrank_when_called_then_equal_residual_output():
    in_features = 6
    out_features = 4
    rank = 2

    residual_weight = torch.randn(out_features, in_features, dtype=torch.float32)
    # 当前实现语义：left=V^T, right=U*S
    svd_lowrank_l1 = torch.zeros(rank, in_features, dtype=torch.float32)
    svd_lowrank_l2 = torch.zeros(out_features, rank, dtype=torch.float32)

    linear = nn.Linear(in_features, out_features, bias=False)
    with torch.no_grad():
        linear.weight.copy_(residual_weight)

    wrapper = SVDResidualWrapper(linear, svd_lowrank_l1, svd_lowrank_l2)

    x = torch.randn(3, in_features, dtype=torch.float32)
    out_linear = linear(x)
    out_wrapper = wrapper(x)

    assert torch.allclose(out_wrapper, out_linear, atol=1e-6, rtol=1e-6)


def test_SVDResidualHookIR___init___given_lowrank_tensors_when_created_then_store_internal_state():
    # 与当前实现语义一致：left=V^T [rank, in]，right=U*S [out, rank]
    svd_lowrank_l1 = torch.randn(2, 5, dtype=torch.float32)
    svd_lowrank_l2 = torch.randn(3, 2, dtype=torch.float32)

    hook = SVDResidualHookIR(svd_lowrank_l1, svd_lowrank_l2)

    assert torch.allclose(hook.svd_lowrank_l1, svd_lowrank_l1)
    assert torch.allclose(hook.svd_lowrank_l2, svd_lowrank_l2)


def test_SVDResidualHookIR___call___given_module_and_args_when_invoked_then_return_same_args():
    linear = nn.Linear(4, 3, bias=False)
    hook = SVDResidualHookIR(
        torch.randn(1, 4, dtype=torch.float32),
        torch.randn(3, 1, dtype=torch.float32),
    )

    x = torch.randn(5, 4, dtype=torch.float32)
    y = torch.randn(5, 4, dtype=torch.float32)
    args = (x, y)

    returned = hook(linear, args)

    # 不修改输入，只是 passthrough
    assert returned is args
    assert returned[0] is x
    assert returned[1] is y


def test_SVDResidualHookIR_wrapper_module_given_linear_module_when_called_then_return_wrapper_and_remove_hook(
    monkeypatch,
):
    svd_lowrank_l1 = torch.randn(2, 4, dtype=torch.float32)
    svd_lowrank_l2 = torch.randn(3, 2, dtype=torch.float32)
    hook = SVDResidualHookIR(svd_lowrank_l1, svd_lowrank_l2)

    # 监控 remove_hook 是否被调用
    called = {"flag": False}

    def fake_remove_hook():
        called["flag"] = True

    monkeypatch.setattr(hook, "remove_hook", fake_remove_hook)

    linear = nn.Linear(4, 3, bias=False)
    wrapper = hook.wrapper_module(linear)

    assert isinstance(wrapper, SVDResidualWrapper)
    assert wrapper.wrapped_module is linear
    assert torch.allclose(wrapper.svd_lowrank_l1.data, svd_lowrank_l1)
    assert torch.allclose(wrapper.svd_lowrank_l2.data, svd_lowrank_l2)
    assert called["flag"] is True
