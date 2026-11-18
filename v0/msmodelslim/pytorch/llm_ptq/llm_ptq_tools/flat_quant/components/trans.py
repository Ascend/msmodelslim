#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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
import torch.nn as nn

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.utils.function_utils import get_init_weight, get_inverse


class SingleTransMatrix(nn.Module):
    """
    Base class for a single transformation matrix.
    This matrix can be applied from the left or right to an input tensor.
    """
    def __init__(self, size, deriction="right"):
        """
        Initializes the SingleTransMatrix.

        Args:
            size (int): The size of the square transformation matrix.
            deriction (str, optional): The direction of multiplication, "right" or "left".
                                      Defaults to "right".
        """
        super(SingleTransMatrix, self).__init__()
        self.size = size
        self._eval_mode = False
        self.register_buffer("matrix", torch.empty(0))
        self.register_buffer("matrix_inv_t", torch.empty(0))
        self.deriction = deriction

    def __repr__(self):
        res = f"{self.__class__.__name__}(eval_mode={self._eval_mode}"
        res += f", matrix.shape={self.size})"
        return res

    def get_matrix(self, inv_t=False):
        """
        Returns the transformation matrix.

        Args:
            inv_t (bool, optional): If True, returns the inverse transpose of the matrix.
                                   Defaults to False.
        
        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def reparameterize(self):
        """
        Reparameterizes the matrix, typically for evaluation mode.
        This might involve pre-computing the matrix if it's defined by learnable parameters.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def to_eval_mode(self):
        """
        Switches the module to evaluation mode.
        This typically involves reparameterizing the matrix.
        """
        if not self._eval_mode:
            with torch.no_grad():
                self.reparameterize()
            self._eval_mode = True

    def forward(self, inp, inv_t=False):
        """
        Applies the transformation matrix to the input tensor.

        Args:
            inp (torch.Tensor): The input tensor.
            inv_t (bool, optional): If True, applies the inverse transpose of the matrix.
                                   Defaults to False.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        if self.deriction == "right":
            init_shape = inp.shape
            matirx = self.get_matrix(inv_t=inv_t).to(inp)
            inp = inp.reshape(-1, matirx.shape[0])
            output = inp @ matirx
            return output.reshape(init_shape)
        elif self.deriction == "left":
            if self.size == 0:
                raise ValueError("size can not be zero.")
            init_shape = inp.shape
            matirx = self.get_matrix(inv_t=inv_t).T.to(inp)
            inp = inp.reshape(-1, self.size, init_shape[-1] // self.size)
            output = matirx @ inp
            return output.reshape(init_shape)
        else:
            raise ValueError(f"Invalid deriction")

    def get_save_params(self):
        """
        Returns parameters to be saved.
        """
        return {self.deriction + "_trans": self.get_matrix()}


class SVDSingleTransMatrix(SingleTransMatrix):
    """
    A single transformation matrix parameterized by Singular Value Decomposition (SVD).
    The matrix is represented as U @ diag(S) @ V.T, where U and V are orthogonal matrices
    and S is a vector of singular values.
    """
    def __init__(self, size, deriction="right", diag_relu=False):
        """
        Initializes the SVDSingleTransMatrix.

        Args:
            size (int): The size of the square transformation matrix.
            deriction (str, optional): The direction of multiplication. Defaults to "right".
            diag_relu (bool, optional): If True, applies a Softplus activation 
                                        to the diagonal elements (singular values)
                                        to ensure they are positive. Defaults to False.
        """
        super(SVDSingleTransMatrix, self).__init__(size, deriction)
        self.linear_u = nn.Linear(size, size, bias=False)
        self.linear_u.weight.data = get_init_weight(size).to(self.linear_u.weight)
        self.linear_u = nn.utils.parametrizations.orthogonal(self.linear_u, 
                                                            orthogonal_map="cayley", 
                                                            use_trivialization=False)
        self.linear_v = nn.Linear(size, size, bias=False)
        self.linear_v.weight.data = get_init_weight(size).to(self.linear_v.weight)
        self.linear_v = nn.utils.parametrizations.orthogonal(self.linear_v, 
                                                            orthogonal_map="cayley", 
                                                            use_trivialization=False)
        if diag_relu:
            beta = 1
            init_diag = torch.log(torch.exp(torch.tensor(beta)) - 1.0) / beta
            self.linear_diag = torch.nn.Parameter(init_diag * torch.ones(size), requires_grad=True)
            self._diag_relu = nn.Softplus(beta=beta, threshold=20)
        else:
            self.linear_diag = torch.nn.Parameter(torch.ones(size), requires_grad=True)
            self._diag_relu = None

    def get_diag(self):
        if self._diag_relu is not None:
            return self._diag_relu(self.linear_diag)
        else:
            return self.linear_diag

    def get_matrix(self, inv_t=False):
        if not self._eval_mode:
            self.linear_u.cpu()
            self.linear_v.cpu()
            orthog_u, orthog_v = self.linear_u.weight, self.linear_v.weight
            linear_diag = self.get_diag()
            orthog_u = orthog_u.to(linear_diag)
            orthog_v = orthog_v.to(linear_diag)
            if inv_t:
                linear_diag = 1 / linear_diag
            return orthog_u @ torch.diag(linear_diag) @ orthog_v.t()
        else:
            if inv_t:
                return self.matrix_inv_t
            return self.matrix

    def reparameterize(self):
        matrix = self.get_matrix()
        matrix_inv_t = self.get_matrix(inv_t=True)
        self.matrix = matrix
        self.matrix_inv_t = matrix_inv_t
        self._eval_mode = True
        del self.linear_u, self.linear_diag, self.linear_v


class InvSingleTransMatrix(SingleTransMatrix):
    """
    A single transformation matrix that is directly learnable and invertible.
    The matrix itself is a parameter.
    """
    def __init__(self, size, deriction="right", **kwargs):
        """
        Initializes the InvSingleTransMatrix.

        Args:
            size (int): The size of the square transformation matrix.
            deriction (str, optional): The direction of multiplication. Defaults to "right".
        """
        super(InvSingleTransMatrix, self).__init__(size, deriction)
        trans_linear = nn.Linear(size, size, bias=False)
        trans_linear.weight.data = get_init_weight(size).to(trans_linear.weight)
        self.trans_linear = trans_linear

    def get_matrix(self, inv_t=False):
        if not self._eval_mode:
            matrix = self.trans_linear.weight
            if inv_t:
                matrix = get_inverse(matrix).T
            return matrix
        else:
            if inv_t:
                return self.matrix_inv_t
            return self.matrix

    def reparameterize(self):
        if not self._eval_mode:
            matrix = self.trans_linear.weight
            matrix_inv_t = get_inverse(matrix).T
            self.matrix = matrix
            self.matrix_inv_t = matrix_inv_t
            self._eval_mode = True
            del self.trans_linear


class DiagonalTransMatrix(nn.Module):
    """
    A transformation matrix that is purely diagonal.
    It scales the input element-wise.
    """
    def __init__(self, size, init_para=None):
        """
        Initializes the DiagonalTransMatrix.

        Args:
            size (int): The number of diagonal elements.
            init_para (torch.Tensor, optional): Initial values for the diagonal elements.
                                               If None, defaults to ones.
        """
        super(DiagonalTransMatrix, self).__init__()
        self.size = size
        if init_para is None:
            self.diag_scale = torch.nn.Parameter(torch.ones((size)), requires_grad=True)
        else:
            self.diag_scale = torch.nn.Parameter(init_para, requires_grad=True)

    def __repr__(self):
        res = f"{self.__class__.__name__}(size={self.size})"
        return res

    def forward(self, inp, inv_t=False):
        if self.diag_scale is None:
            return inp
        if inv_t:
            inp = inp / self.diag_scale.to(inp)
        else:
            inp = inp * self.diag_scale.to(inp)
        return inp

    def reparameterize(self):
        self.diag_scale = None

    def to_eval_mode(self):
        self.reparameterize()

    def get_save_params(self):
        return {}


class GeneralMatrixTrans(nn.Module):
    """
    A general matrix transformation module that applies left and right transformations,
    and optionally a diagonal transformation.
    The transformation is of the form: L @ (diag_trans(X)) @ R.
    """
    def __init__(self, left_size, 
                 right_size, 
                 add_diag=False, 
                 diag_init_para=None, 
                 tran_type="svd",
                 diag_relu=False):
        super(GeneralMatrixTrans, self).__init__()
        TranMatrix = SVDSingleTransMatrix if tran_type == "svd" else InvSingleTransMatrix
        self.left_trans = TranMatrix(left_size, deriction="left", diag_relu=diag_relu)
        self.right_trans = TranMatrix(right_size, deriction="right", diag_relu=diag_relu)
        
        if add_diag:
            self.diag_trans = DiagonalTransMatrix(left_size * right_size, diag_init_para)
        else:
            self.diag_trans = None
        
        
    def forward(self, inp, inv_t=False):
        if self.diag_trans is not None:
            inp = self.diag_trans(inp, inv_t=inv_t)
        if self.right_trans is not None:
            inp = self.right_trans(inp, inv_t=inv_t)
        if self.left_trans is not None:
            inp = self.left_trans(inp, inv_t=inv_t)
        return inp

    def to_eval_mode(self):
        self.left_trans.to_eval_mode()
        self.right_trans.to_eval_mode()
        if self.diag_trans is not None:
            self.diag_trans.to_eval_mode()

    def get_save_params(self):
        return {**self.left_trans.get_save_params(), **self.right_trans.get_save_params()}
        