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
# pylint: disable=redefined-outer-name,super-with-arguments

import torch
import pytest

try:
    import torch_npu
except Exception:
    torch_npu = None

try:
    from msmodelslim.pytorch.sparse import sparse_tools
except Exception:
    sparse_tools = None


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H, bias=True)
        self.linear2 = torch.nn.Linear(H, D_out, bias=True)

    def forward(self, x):
        _ = self.linear1(x)
        y_pred = self.linear2(_)
        return y_pred


@pytest.fixture(scope="function")
def generate_model():
    model = TwoLayerNet(100, 10, 1)
    yield model


def get_sparsity(model):
    total_params = 0
    total_zero_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            total_zero_params += torch.sum(param == 0).item()
    try:
        sparsity = total_zero_params / total_params
    except ZeroDivisionError as error:
        raise ZeroDivisionError("divisor cannot be zero!") from error
    return sparsity


@pytest.mark.skipif(torch_npu is None or sparse_tools is None, reason="requires torch_npu and sparse_tools")
def test_sparse_tools_given_model_then_pass(generate_model):
    # 修改 test_dataset 为一个包含字符串的列表
    test_dataset = ["example_string_1", "example_string_2"]
    sparse_config = sparse_tools.SparseConfig(method='magnitude')
    prune_compressor = sparse_tools.Compressor(generate_model, sparse_config)
    prune_compressor.compress(dataset=test_dataset)
