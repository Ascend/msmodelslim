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
# pylint: disable=attribute-defined-outside-init,consider-using-from-import

import torch
import torch.nn as nn
from msmodelslim.utils.exception import UnexpectedError, UnsupportedError, SchemaValidateError
from msmodelslim.processor.flat_quant.flat_quant_utils.utils import get_init_weight, get_inverse


class SingleTransMatrix(nn.Module):
    """基础变换矩阵类，支持左乘或右乘输入张量，用于实现可学习的线性变换。"""

    def __init__(self, size, direction="right"):
        super().__init__()
        self.size = size
        self._eval_mode = False
        self.direction = direction

    def __repr__(self):
        return f"{self.__class__.__name__}(eval_mode={self._eval_mode}, matrix.shape={self.size})"

    def get_matrix(self, inv_t=False):
        """获取当前变换矩阵（子类必须实现），支持返回逆转置用于反向变换。"""
        raise UnsupportedError("Subclasses must implement the `get_matrix` method.")

    def reparameterize(self):
        """重参数化：将可学习参数固化为固定缓冲区，释放动态参数以节省内存。"""
        raise UnsupportedError("Subclasses must implement the `reparameterize` method.")

    def to_eval_mode(self):
        """切换至评估模式：触发重参数化，冻结参数，提升推理效率。"""
        if not self._eval_mode:
            with torch.no_grad():
                self.reparameterize()
            self._eval_mode = True

    def forward(self, inp, inv_t=False):
        """对输入张量应用变换矩阵，支持左乘或右乘方向。"""
        if self.direction == "right":
            init_shape = inp.shape
            matrix = self.get_matrix(inv_t=inv_t).to(inp)
            inp = inp.reshape(-1, matrix.shape[0])
            output = inp @ matrix
            return output.reshape(init_shape)
        elif self.direction == "left":
            if self.size == 0:
                raise UnexpectedError("The dimension size for transformation cannot be zero.")
            init_shape = inp.shape
            matrix = self.get_matrix(inv_t=inv_t).T.to(inp)
            inp = inp.reshape(-1, self.size, init_shape[-1] // self.size)
            output = matrix @ inp
            return output.reshape(init_shape)
        else:
            raise SchemaValidateError(
                f"Invalid transformation direction: {self.direction}. Only 'left' or 'right' are supported."
            )

    def get_save_params(self):
        """返回需要保存的参数字典，用于模型序列化，默认返回方向对应的变换矩阵。"""
        return {self.direction + "_trans": self.get_matrix()}


class SVDSingleTransMatrix(SingleTransMatrix):
    """基于 SVD 分解的可学习变换矩阵：U @ diag(S) @ V^T，使用 Cayley 变换保证正交性。"""

    def __init__(self, size, direction="right", diag_relu=False):
        super().__init__(size, direction)

        # 左正交矩阵 U（左乘方向）
        self.linear_u = nn.Linear(size, size, bias=False)
        self.linear_u.weight.data = get_init_weight(size).to(self.linear_u.weight)
        self.linear_u = nn.utils.parametrizations.orthogonal(
            self.linear_u, orthogonal_map="cayley", use_trivialization=False
        )

        # 右正交矩阵 V（右乘方向）
        self.linear_v = nn.Linear(size, size, bias=False)
        self.linear_v.weight.data = get_init_weight(size).to(self.linear_v.weight)
        self.linear_v = nn.utils.parametrizations.orthogonal(
            self.linear_v, orthogonal_map="cayley", use_trivialization=False
        )

        # 对角缩放因子 S（可学习）
        if diag_relu:
            beta = 1
            init_diag = torch.log(torch.exp(torch.tensor(beta)) - 1.0) / beta
            self.linear_diag = torch.nn.Parameter(init_diag * torch.ones(size), requires_grad=True)
            self._diag_relu = nn.Softplus(beta=beta, threshold=20)
        else:
            self.linear_diag = torch.nn.Parameter(torch.ones(size), requires_grad=True)
            self._diag_relu = None

    def get_diag(self):
        """获取经过激活函数处理后的对角元素（即奇异值），确保为正数。"""
        return self._diag_relu(self.linear_diag) if self._diag_relu is not None else self.linear_diag

    def get_matrix(self, inv_t=False):
        """动态计算变换矩阵（训练模式）或从缓冲区读取（评估模式）。"""
        if not self._eval_mode:
            orthog_u = self.linear_u.weight.to(self.get_diag())
            orthog_v = self.linear_v.weight.to(self.get_diag())
            diag = self.get_diag()
            if inv_t:
                diag = 1 / diag  # 逆变换：对角取倒数
            return orthog_u @ torch.diag(diag) @ orthog_v.t()
        else:
            return self.matrix_inv_t if inv_t else self.matrix

    def reparameterize(self):
        """重参数化：将可学习参数固化为固定矩阵并释放动态参数，以节省内存。"""
        matrix = self.get_matrix()
        matrix_inv_t = self.get_matrix(inv_t=True)
        self.matrix = matrix
        self.matrix_inv_t = matrix_inv_t
        self._eval_mode = True
        del self.linear_u, self.linear_v, self.linear_diag


class InvSingleTransMatrix(SingleTransMatrix):
    """直接可学习的可逆变换矩阵：参数即为变换矩阵本身，适用于轻量级可逆变换。"""

    def __init__(self, size, direction="right", **kwargs):
        super().__init__(size, direction)
        trans_linear = nn.Linear(size, size, bias=False)
        trans_linear.weight.data = get_init_weight(size).to(trans_linear.weight)
        self.trans_linear = trans_linear

    def get_matrix(self, inv_t=False):
        """获取变换矩阵或其逆转置，支持逆变换操作。"""
        if not self._eval_mode:
            matrix = self.trans_linear.weight
            if inv_t:
                matrix = get_inverse(matrix).T
            return matrix
        else:
            return self.matrix_inv_t if inv_t else self.matrix

    def reparameterize(self):
        """重参数化：将可学习权重固化为缓冲区，并删除原参数。"""
        matrix = self.trans_linear.weight
        matrix_inv_t = get_inverse(matrix).T
        self.matrix = matrix
        self.matrix_inv_t = matrix_inv_t
        self._eval_mode = True
        del self.trans_linear


class DiagonalTransMatrix(nn.Module):
    """纯对角变换矩阵：仅对输入进行逐元素缩放，用于实现对角缩放因子。"""

    def __init__(self, size, init_para=None):
        super().__init__()
        self.size = size
        if init_para is None:
            self.diag_scale = torch.nn.Parameter(torch.ones(size), requires_grad=True)
        else:
            self.diag_scale = torch.nn.Parameter(init_para, requires_grad=True)

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"

    def forward(self, inp, inv_t=False):
        """对输入张量进行逐元素缩放，支持逆变换（除法）。"""
        if self.diag_scale is None:
            return inp
        scale = self.diag_scale.to(inp)
        return inp / scale if inv_t else inp * scale

    def reparameterize(self):
        """评估模式下释放对角参数，避免内存占用。"""
        self.diag_scale = None

    def to_eval_mode(self):
        """切换至评估模式，触发参数释放。"""
        self.reparameterize()

    def get_save_params(self):
        """对角矩阵在评估模式下已释放参数，无需保存，返回空字典。"""
        return {}


class GeneralMatrixTrans(nn.Module):
    """通用矩阵变换模块：L @ diag_trans(X) @ R，组合左变换、对角缩放、右变换。"""

    def __init__(self, left_size, right_size, add_diag=False, diag_init_para=None, tran_type="svd", diag_relu=False):
        super().__init__()
        TranMatrix = SVDSingleTransMatrix if tran_type == "svd" else InvSingleTransMatrix
        self.left_trans = TranMatrix(left_size, direction="left", diag_relu=diag_relu)
        self.right_trans = TranMatrix(right_size, direction="right", diag_relu=diag_relu)
        self.diag_trans = DiagonalTransMatrix(left_size * right_size, diag_init_para) if add_diag else None

    def forward(self, inp, inv_t=False):
        """按顺序应用变换：对角缩放 → 右乘 → 左乘，确保与量化流程一致。"""
        if self.diag_trans is not None:
            inp = self.diag_trans(inp, inv_t=inv_t)
        if self.right_trans is not None:
            inp = self.right_trans(inp, inv_t=inv_t)
        if self.left_trans is not None:
            inp = self.left_trans(inp, inv_t=inv_t)
        return inp

    def to_eval_mode(self):
        """统一切换所有子模块至评估模式，触发重参数化，用于推理前优化。"""
        self.left_trans.to_eval_mode()
        self.right_trans.to_eval_mode()
        if self.diag_trans is not None:
            self.diag_trans.to_eval_mode()

    def get_save_params(self):
        """返回需要保存的变换参数（左、右矩阵），支持模型回退模式。"""
        return {**self.left_trans.get_save_params(), **self.right_trans.get_save_params()}
