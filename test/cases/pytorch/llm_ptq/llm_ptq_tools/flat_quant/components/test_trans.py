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
# pylint: disable=consider-using-from-import,use-implicit-booleaness-not-comparison

import torch
import torch.nn as nn
from unittest.mock import patch

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.components.trans import (
    SVDSingleTransMatrix,
    InvSingleTransMatrix,
    DiagonalTransMatrix,
    GeneralMatrixTrans,
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.components.trans import (
    SingleTransMatrix,  # noqa: F401
)  # noqa: F401
from unittest.mock import Mock  # noqa: F401
import pytest  # noqa: F401


class TestSVDSingleTransMatrix:
    def test_init_default_values(self):
        matrix = SVDSingleTransMatrix(size=10)

        assert matrix.size == 10
        assert matrix.deriction == "right"
        assert hasattr(matrix, 'linear_u')
        assert hasattr(matrix, 'linear_v')
        assert hasattr(matrix, 'linear_diag')
        assert matrix._diag_relu is None

    def test_init_with_custom_direction(self):
        matrix = SVDSingleTransMatrix(size=10, deriction="left")

        assert matrix.deriction == "left"

    def test_init_with_diag_relu_enabled(self):
        matrix = SVDSingleTransMatrix(size=10, diag_relu=True)

        assert matrix._diag_relu is not None
        assert isinstance(matrix._diag_relu, nn.Softplus)

    def test_get_diag_without_relu(self):
        matrix = SVDSingleTransMatrix(size=10, diag_relu=False)

        diag = matrix.get_diag()

        assert diag.shape == (10,)

    def test_get_diag_with_relu(self):
        matrix = SVDSingleTransMatrix(size=10, diag_relu=True)

        diag = matrix.get_diag()

        assert diag.shape == (10,)
        assert torch.all(diag > 0)  # Softplus ensures positive values

    def test_get_matrix_not_eval_mode(self):
        matrix = SVDSingleTransMatrix(size=10)

        result = matrix.get_matrix()

        assert result.shape == (10, 10)

    def test_get_matrix_with_inv_t(self):
        matrix = SVDSingleTransMatrix(size=10)

        result = matrix.get_matrix(inv_t=True)

        assert result.shape == (10, 10)

    def test_get_matrix_eval_mode(self):
        matrix = SVDSingleTransMatrix(size=10)
        matrix.to_eval_mode()

        result = matrix.get_matrix()

        assert result.shape == (10, 10)
        assert matrix._eval_mode is True

    def test_get_matrix_eval_mode_inv_t(self):
        matrix = SVDSingleTransMatrix(size=10)
        matrix.to_eval_mode()

        result = matrix.get_matrix(inv_t=True)

        assert result.shape == (10, 10)

    def test_reparameterize(self):
        matrix = SVDSingleTransMatrix(size=10)

        matrix.reparameterize()

        assert matrix._eval_mode is True
        assert hasattr(matrix, 'matrix')
        assert hasattr(matrix, 'matrix_inv_t')

    def test_forward_basic(self):
        matrix = SVDSingleTransMatrix(size=10)
        inp = torch.randn(3, 10)

        result = matrix(inp)

        assert result.shape == inp.shape

    def test_forward_left_direction(self):
        matrix = SVDSingleTransMatrix(size=5, deriction="left")
        inp = torch.randn(2, 10)  # 10 = 5 * 2

        result = matrix(inp)

        assert result.shape == inp.shape


class TestInvSingleTransMatrix:
    def test_init(self):
        matrix = InvSingleTransMatrix(size=10)

        assert matrix.size == 10
        assert matrix.deriction == "right"
        assert hasattr(matrix, 'trans_linear')

    def test_get_matrix_not_eval_mode(self):
        matrix = InvSingleTransMatrix(size=10)

        result = matrix.get_matrix()

        assert result.shape == (10, 10)

    def test_get_matrix_with_inv_t(self):
        matrix = InvSingleTransMatrix(size=10)

        result = matrix.get_matrix(inv_t=True)

        assert result.shape == (10, 10)

    def test_get_matrix_eval_mode(self):
        matrix = InvSingleTransMatrix(size=10)
        matrix.to_eval_mode()

        result = matrix.get_matrix()

        assert result.shape == (10, 10)

    def test_reparameterize(self):
        matrix = InvSingleTransMatrix(size=10)

        matrix.reparameterize()

        assert matrix._eval_mode is True
        assert hasattr(matrix, 'matrix')
        assert hasattr(matrix, 'matrix_inv_t')


class TestDiagonalTransMatrix:
    def test_init_default_params(self):
        matrix = DiagonalTransMatrix(size=10)

        assert matrix.size == 10
        assert hasattr(matrix, 'diag_scale')
        assert matrix.diag_scale.shape == (10,)

    def test_init_with_custom_params(self):
        init_para = torch.ones(5) * 2.0
        matrix = DiagonalTransMatrix(size=5, init_para=init_para)

        assert torch.allclose(matrix.diag_scale, init_para)

    def test_repr(self):
        matrix = DiagonalTransMatrix(size=10)
        repr_str = repr(matrix)

        assert "DiagonalTransMatrix" in repr_str
        assert "size=10" in repr_str

    def test_forward_normal(self):
        matrix = DiagonalTransMatrix(size=10)
        inp = torch.randn(3, 10)

        result = matrix(inp)

        assert result.shape == inp.shape

    def test_forward_with_inv_t(self):
        matrix = DiagonalTransMatrix(size=10)
        inp = torch.randn(3, 10)

        result = matrix(inp, inv_t=True)

        assert result.shape == inp.shape

    def test_forward_with_none_diag_scale(self):
        matrix = DiagonalTransMatrix(size=10)
        matrix.diag_scale = None
        inp = torch.randn(3, 10)

        result = matrix(inp)

        assert torch.equal(result, inp)

    def test_reparameterize(self):
        matrix = DiagonalTransMatrix(size=10)

        matrix.reparameterize()

        assert matrix.diag_scale is None

    def test_to_eval_mode(self):
        matrix = DiagonalTransMatrix(size=10)

        matrix.to_eval_mode()

        assert matrix.diag_scale is None

    def test_get_save_params(self):
        matrix = DiagonalTransMatrix(size=10)

        params = matrix.get_save_params()

        assert params == {}


class TestGeneralMatrixTrans:
    def test_init_default(self):
        trans = GeneralMatrixTrans(left_size=8, right_size=16)

        assert hasattr(trans, 'left_trans')
        assert hasattr(trans, 'right_trans')
        assert trans.diag_trans is None

    def test_init_with_diag(self):
        diag_init = torch.ones(8 * 16) * 0.5
        trans = GeneralMatrixTrans(left_size=8, right_size=16, add_diag=True, diag_init_para=diag_init)

        assert trans.diag_trans is not None
        assert isinstance(trans.diag_trans, DiagonalTransMatrix)

    def test_init_with_inv_type(self):
        trans = GeneralMatrixTrans(left_size=8, right_size=16, tran_type="inv")

        assert isinstance(trans.left_trans, InvSingleTransMatrix)
        assert isinstance(trans.right_trans, InvSingleTransMatrix)

    def test_forward_without_diag(self):
        trans = GeneralMatrixTrans(left_size=4, right_size=8)
        inp = torch.randn(2, 32)  # 32 = 4 * 8

        result = trans(inp)

        assert result.shape == inp.shape

    def test_forward_with_diag(self):
        trans = GeneralMatrixTrans(left_size=4, right_size=8, add_diag=True)
        inp = torch.randn(2, 32)

        result = trans(inp)

        assert result.shape == inp.shape

    def test_forward_with_inv_t(self):
        trans = GeneralMatrixTrans(left_size=4, right_size=8)
        inp = torch.randn(2, 32)

        result = trans(inp, inv_t=True)

        assert result.shape == inp.shape

    def test_to_eval_mode(self):
        trans = GeneralMatrixTrans(left_size=4, right_size=8, add_diag=True)

        trans.to_eval_mode()

        assert trans.left_trans._eval_mode is True
        assert trans.right_trans._eval_mode is True

    def test_get_save_params(self):
        trans = GeneralMatrixTrans(left_size=4, right_size=8)

        with patch.object(trans.left_trans, 'get_save_params', return_value={'left': 'test'}):
            with patch.object(trans.right_trans, 'get_save_params', return_value={'right': 'test'}):
                params = trans.get_save_params()

                assert 'left' in params
                assert 'right' in params
