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
# pylint: disable=pointless-string-statement

"""
msmodelslim.processor.adapt_rotation.iterative_hadamard_optimization 模块的单元测试
"""
import unittest  # noqa: E402

import torch  # noqa: E402

from msmodelslim.processor.adapt_rotation.iterative_hadamard_optimization import (  # noqa: E402
    orthogonal_transform,
    reconstruction_loss,
    quant_tensor_sym,
    quant_tensor_sym_batched,
    HadamardOptimizer,
)


class TestOrthogonalTransform(unittest.TestCase):
    """测试 orthogonal_transform 函数"""

    def test_orthogonal_transform_return_orthogonal_matrix_when_inputs_valid(self):
        """测试输入有效时返回正交矩阵"""
        A = torch.randn(4, 4)
        B = torch.randn(4, 4)
        X = orthogonal_transform(A, B, iters=50)
        XtX = X.T @ X
        I = torch.eye(4)  # noqa: E741
        self.assertTrue(torch.allclose(XtX, I, atol=1e-4))

    def test_orthogonal_transform_return_shape_d_by_d_when_inputs_are_n_by_d(self):
        """测试输入 (N,D) 时返回 (D,D) 形状"""
        A = torch.randn(10, 4)
        B = torch.randn(10, 4)
        X = orthogonal_transform(A, B, iters=20)
        self.assertEqual(X.shape, (4, 4))


class TestReconstructionLoss(unittest.TestCase):
    """测试 reconstruction_loss 函数"""

    def test_reconstruction_loss_return_zero_when_a_equals_b(self):
        """测试 A==B 时损失为 0"""
        A = torch.randn(4, 4)
        loss = reconstruction_loss(A, A.clone())
        self.assertAlmostEqual(loss, 0.0, places=6)

    def test_reconstruction_loss_return_positive_when_a_differs_from_b(self):
        """测试 A!=B 时损失为正"""
        A = torch.randn(4, 4)
        B = A + 1.0
        loss = reconstruction_loss(A, B)
        self.assertGreater(loss, 0)


class TestQuantTensorSym(unittest.TestCase):
    """测试 quant_tensor_sym 函数"""

    def test_quant_tensor_sym_return_same_shape_as_input(self):
        """测试量化后形状与输入一致"""
        tensor = torch.randn(8, 4)
        quantized = quant_tensor_sym(tensor, quant_dtype="int4")
        self.assertEqual(quantized.shape, tensor.shape)

    def test_quant_tensor_sym_return_float_dtype(self):
        """测试量化结果为 float 类型"""
        tensor = torch.randn(4, 4)
        quantized = quant_tensor_sym(tensor, quant_dtype="int4")
        self.assertTrue(quantized.dtype in (torch.float32, torch.float16))


class TestQuantTensorSymBatched(unittest.TestCase):
    """测试 quant_tensor_sym_batched 函数"""

    def test_quant_tensor_sym_batched_return_same_shape_as_input(self):
        """测试批量化化后形状与输入一致"""
        tensor = torch.randn(20, 4)
        quantized = quant_tensor_sym_batched(tensor, batch_size=8, quant_dtype="int4")
        self.assertEqual(quantized.shape, tensor.shape)


class TestHadamardOptimizer(unittest.TestCase):
    """测试 HadamardOptimizer 类"""

    def test_init_set_default_values_when_no_args_provided(self):
        """测试未提供参数时使用默认值"""
        opt = HadamardOptimizer()
        self.assertEqual(opt.quant_dtype, "int4")
        self.assertEqual(opt.batch_size, 128)
        self.assertEqual(opt.steps, 20)
        self.assertEqual(opt.patience, 5)
        self.assertEqual(opt.min_steps, 6)
        self.assertEqual(opt.max_samples, 2048)

    def test_optimize_return_matrix_shape_d_by_d_when_hadamard_is_d_by_d(self):
        """测试 optimize 返回 (D,D) 形状矩阵"""
        opt = HadamardOptimizer(steps=2, min_steps=1, patience=10)
        D = 4
        hadamard = torch.eye(D)
        acts = {"layer1": torch.randn(32, D)}
        result = opt.optimize(acts, hadamard)
        self.assertEqual(result.shape, (D, D))

    def test_optimize_return_tensor_on_same_device_as_hadamard_when_device_not_specified(self):
        """测试未指定 device 时结果与 hadamard 同设备"""
        opt = HadamardOptimizer(steps=2, min_steps=1, patience=10)
        D = 4
        hadamard = torch.eye(D)
        acts = {"layer1": torch.randn(32, D)}
        result = opt.optimize(acts, hadamard)
        self.assertEqual(result.device, hadamard.device)

    def test_optimize_handle_multiple_layers_in_activations_dict(self):
        """测试 activations_dict 含多个 layer 时正常优化"""
        opt = HadamardOptimizer(steps=2, min_steps=1, patience=10)
        D = 4
        hadamard = torch.eye(D)
        acts = {
            "layer1": torch.randn(16, D),
            "layer2": torch.randn(16, D),
        }
        result = opt.optimize(acts, hadamard)
        self.assertEqual(result.shape, (D, D))
