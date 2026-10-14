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
# pylint: disable=consider-using-from-import

import torch
import torch.nn as nn
from unittest.mock import patch

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.utils.function_utils import (
    get_init_scale,
    get_decompose_dim,
    get_init_weight,
    get_inverse,
    get_n_set_parameters_byname,
    set_require_grad_all,
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.utils.function_utils import (
    get_random_orthg,  # noqa: F401
)  # noqa: F401
from unittest.mock import Mock  # noqa: F401
import numpy as np  # noqa: F401
import pytest  # noqa: F401


class TestMathematicalFunctions:
    def test_get_init_scale(self):
        w_smax = torch.tensor([2.0, 4.0])
        x_smax = torch.tensor([1.0, 2.0])
        alpha = 0.5

        result = get_init_scale(w_smax, x_smax, alpha)

        assert result.shape == w_smax.shape
        assert torch.all(result >= 1e-5)

    def test_get_decompose_dim(self):
        # Test perfect square
        result = get_decompose_dim(16)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] <= result[1]

        # Test non-perfect square
        result = get_decompose_dim(15)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_init_weight(self):
        dim = 4

        result = get_init_weight(dim)

        assert result.shape == (dim, dim)
        assert isinstance(result, torch.Tensor)

    def test_get_inverse(self):
        # Test regular case
        matrix = torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=torch.float32)

        result = get_inverse(matrix)

        assert result.shape == matrix.shape
        assert result.dtype == matrix.dtype

        # Check that result is actually the inverse
        identity_check = torch.mm(matrix, result)
        expected_identity = torch.eye(2)
        assert torch.allclose(identity_check, expected_identity, atol=1e-6)

    @patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.utils.function_utils.npu_available', True)
    def test_get_inverse_npu_path(self):
        matrix = torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=torch.float32)

        result = get_inverse(matrix)

        assert result.shape == matrix.shape
        assert result.dtype == matrix.dtype


class TestModelParameterFunctions:
    def test_get_n_set_parameters_byname(self):
        model = nn.Sequential(nn.Linear(10, 5, bias=True), nn.Linear(5, 2, bias=True))
        required_names = ["weight"]

        params = list(get_n_set_parameters_byname(model, required_names))

        # Should find 2 weight parameters
        assert len(params) == 2
        for param in params:
            assert param.requires_grad is True

    def test_get_n_set_parameters_byname_partial_match(self):
        model = nn.Sequential(nn.Linear(10, 5, bias=True), nn.Linear(5, 2, bias=True))
        required_names = ["0.weight"]  # Only first layer weight

        params = list(get_n_set_parameters_byname(model, required_names))

        # Should find 1 weight parameter
        assert len(params) == 1
        assert params[0].requires_grad is True

    def test_get_n_set_parameters_byname_no_match(self):
        model = nn.Sequential(nn.Linear(10, 5, bias=True))
        required_names = ["nonexistent"]

        params = list(get_n_set_parameters_byname(model, required_names))

        assert len(params) == 0

    def test_set_require_grad_all_true(self):
        model = nn.Sequential(nn.Linear(10, 5, bias=True), nn.Linear(5, 2, bias=True))

        # First set all to False
        for param in model.parameters():
            param.requires_grad = False

        set_require_grad_all(model, True)

        for param in model.parameters():
            assert param.requires_grad is True

    def test_set_require_grad_all_false(self):
        model = nn.Sequential(nn.Linear(10, 5, bias=True), nn.Linear(5, 2, bias=True))

        # All parameters should start with requires_grad=True by default
        set_require_grad_all(model, False)

        for param in model.parameters():
            assert param.requires_grad is False
