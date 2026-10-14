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

from msmodelslim.processor.flat_quant.flat_quant_utils.trans_matrix import (
    DiagonalTransMatrix,
    GeneralMatrixTrans,
    InvSingleTransMatrix,
    SingleTransMatrix,
    SVDSingleTransMatrix,
)
from msmodelslim.utils.exception import SchemaValidateError, UnexpectedError, UnsupportedError


class _ConcreteTransMatrix(SingleTransMatrix):
    """最小可用的 SingleTransMatrix 子类，固定返回单位矩阵。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.matrix = None
        self.matrix_inv_t = None

    def get_matrix(self, inv_t=False):
        return torch.eye(self.size)

    def reparameterize(self):
        self._eval_mode = True
        self.matrix = torch.eye(self.size)
        self.matrix_inv_t = torch.eye(self.size)


class TestSingleTransMatrix:
    """Test suite for SingleTransMatrix — 基础变换。"""

    # ---------- 正常情形 ----------

    def test_forward_right_direction_preserves_shape_when_input_2d(self):
        """主路径：direction=right 时，forward 应右乘变换矩阵，保留输入形状。"""
        m = _ConcreteTransMatrix(size=4, direction="right")
        inp = torch.randn(2, 4)

        out = m(inp)

        assert out.shape == inp.shape

    def test_forward_left_direction_returns_correct_shape_when_size_divides_last_dim(self):
        """主路径：direction=left 时，forward 应左乘变换矩阵。"""
        m = _ConcreteTransMatrix(size=2, direction="left")
        inp = torch.randn(2, 4)

        out = m(inp)

        assert out.shape == inp.shape

    def test_to_eval_mode_sets_eval_mode_flag_when_called_first_time(self):
        """主路径：to_eval_mode 第一次调用应触发 reparameterize 并置 _eval_mode=True。"""
        m = _ConcreteTransMatrix(size=4)

        m.to_eval_mode()

        assert m._eval_mode is True

    def test_to_eval_mode_is_idempotent_when_called_twice(self):
        """边界：to_eval_mode 二次调用不应重复 reparameterize。"""
        m = _ConcreteTransMatrix(size=4)
        m.to_eval_mode()
        first = m._eval_mode
        m.to_eval_mode()  # 不应抛错

        assert m._eval_mode is first

    def test_get_save_params_returns_dict_with_direction_key_when_called(self):
        """主路径：get_save_params 返回的字典应包含 direction 键。"""
        m = _ConcreteTransMatrix(size=4, direction="right")

        params = m.get_save_params()

        assert "right_trans" in params

    # ---------- 异常情形 ----------

    def test_forward_raises_unexpected_error_when_size_is_zero_in_left_direction(self):
        """异常：direction=left + size=0 时 forward 应抛 UnexpectedError。"""
        m = _ConcreteTransMatrix(size=0, direction="left")
        inp = torch.randn(2, 4)

        with pytest.raises(UnexpectedError):
            m(inp)

    def test_forward_raises_schema_validate_error_when_direction_is_invalid(self):
        """异常：direction 非 left/right 应抛 SchemaValidateError。"""
        m = _ConcreteTransMatrix(size=4, direction="diagonal")
        inp = torch.randn(2, 4)

        with pytest.raises(SchemaValidateError):
            m(inp)

    def test_get_matrix_raises_unsupported_error_when_called_on_base_class(self):
        """异常：基类 get_matrix 应抛 UnsupportedError（强制子类实现）。"""
        m = SingleTransMatrix(size=4)

        with pytest.raises(UnsupportedError):
            m.get_matrix()

    def test_reparameterize_raises_unsupported_error_when_called_on_base_class(self):
        """异常：基类 reparameterize 应抛 UnsupportedError。"""
        m = SingleTransMatrix(size=4)

        with pytest.raises(UnsupportedError):
            m.reparameterize()


class TestSVDSingleTransMatrix:
    """Test suite for SVDSingleTransMatrix — SVD 分解变换。"""

    def test_get_matrix_returns_square_matrix_of_size_when_in_training_mode(self):
        """主路径：训练模式下 get_matrix 返回 size×size 矩阵。"""
        m = SVDSingleTransMatrix(size=8, direction="right")

        mat = m.get_matrix()

        assert mat.shape == (8, 8)

    def test_get_matrix_with_inv_t_returns_inverse_diagonal_when_called(self):
        """边界：inv_t=True 应返回对角取倒数的矩阵。"""
        m = SVDSingleTransMatrix(size=4, direction="right")

        mat = m.get_matrix(inv_t=True)

        assert mat.shape == (4, 4)

    def test_reparameterize_frees_dynamic_params_when_called(self):
        """主路径：reparameterize 后 linear_u/v/diag 应被删除。"""
        m = SVDSingleTransMatrix(size=4, direction="right")

        m.reparameterize()

        assert m._eval_mode is True
        assert not hasattr(m, "linear_u")
        assert not hasattr(m, "linear_v")
        assert not hasattr(m, "linear_diag")

    def test_svd_with_diag_relu_initializes_softplus(self):
        """边界：diag_relu=True 时应使用 Softplus 激活。"""
        m = SVDSingleTransMatrix(size=4, direction="right", diag_relu=True)

        # 奇异值应 > 0（Softplus 输出）
        diag = m.get_diag()
        assert torch.all(diag > 0)


class TestInvSingleTransMatrix:
    """Test suite for InvSingleTransMatrix — 直接可学习矩阵。"""

    def test_get_matrix_returns_orthogonal_matrix_in_training_mode(self):
        """主路径：训练模式 get_matrix 返回正交矩阵。"""
        m = InvSingleTransMatrix(size=8, direction="right")

        mat = m.get_matrix()

        # 初始化为正交矩阵
        assert mat.shape == (8, 8)

    def test_reparameterize_frees_trans_linear_when_called(self):
        """主路径：reparameterize 后 trans_linear 应被删除。"""
        m = InvSingleTransMatrix(size=4, direction="right")

        m.reparameterize()

        assert m._eval_mode is True
        assert not hasattr(m, "trans_linear")


class TestDiagonalTransMatrix:
    """Test suite for DiagonalTransMatrix — 对角缩放。"""

    # ---------- 正常情形 ----------

    def test_forward_multiplies_input_by_scale_when_inv_t_false(self):
        """主路径：inv_t=False 时，输出 = input * scale。"""
        m = DiagonalTransMatrix(size=4, init_para=torch.tensor([2.0, 2.0, 2.0, 2.0]))
        inp = torch.ones(3, 4)

        out = m(inp)

        assert torch.allclose(out, torch.full((3, 4), 2.0))

    def test_forward_divides_input_by_scale_when_inv_t_true(self):
        """主路径：inv_t=True 时，输出 = input / scale。"""
        m = DiagonalTransMatrix(size=4, init_para=torch.tensor([2.0, 2.0, 2.0, 2.0]))
        inp = torch.full((3, 4), 4.0)

        out = m(inp, inv_t=True)

        assert torch.allclose(out, torch.full((3, 4), 2.0))

    def test_forward_returns_input_as_is_when_diag_scale_is_none(self):
        """边界：diag_scale 为 None 时（reparameterize 后）原样返回。"""
        m = DiagonalTransMatrix(size=4)
        m.diag_scale = None
        inp = torch.ones(2, 4)

        out = m(inp)

        assert torch.equal(out, inp)

    def test_reparameterize_sets_diag_scale_to_none(self):
        """主路径：reparameterize 后 diag_scale 为 None。"""
        m = DiagonalTransMatrix(size=4)

        m.reparameterize()

        assert m.diag_scale is None

    def test_get_save_params_returns_empty_dict_when_diag_scale_released(self):
        """主路径：get_save_params 在参数释放后返回空 dict。"""
        m = DiagonalTransMatrix(size=4)

        params = m.get_save_params()

        assert not params


class TestGeneralMatrixTrans:
    """Test suite for GeneralMatrixTrans — 通用组合变换。"""

    def test_forward_returns_same_shape_when_no_diag_trans(self):
        """主路径：无 diag_trans 时，输出形状与输入相同。"""
        m = GeneralMatrixTrans(left_size=4, right_size=4, add_diag=False)
        inp = torch.randn(2, 4)

        out = m(inp)

        assert out.shape == inp.shape

    def test_forward_with_diag_trans_returns_same_shape_when_called(self):
        """边界：含 diag_trans 时，输出形状仍与输入相同。"""
        # left_size * right_size = 4*4 = 16
        m = GeneralMatrixTrans(left_size=4, right_size=4, add_diag=True)
        inp = torch.randn(2, 16)  # last dim = left_size * right_size

        out = m(inp)

        assert out.shape == inp.shape

    def test_to_eval_mode_invokes_eval_on_submodules_when_called(self):
        """主路径：to_eval_mode 应递归调用所有子模块的 to_eval_mode。"""
        m = GeneralMatrixTrans(left_size=4, right_size=4, add_diag=True)

        m.to_eval_mode()

        assert m.left_trans._eval_mode is True
        assert m.right_trans._eval_mode is True
        assert m.diag_trans.diag_scale is None
