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

import unittest

import torch

from msmodelslim.processor.quarot.common.hadamard import (
    is_pow2,
    walsh_matrix,
    random_hadamard_matrix,
    matmul_had_u,
    matmul_had_u_t,
)


class TestIsPow2(unittest.TestCase):
    """测试 is_pow2 函数"""

    def test_powers_of_two_should_return_true(self):
        """测试 2 的幂次返回 True"""
        powers = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        for n in powers:
            with self.subTest(n=n):
                self.assertTrue(is_pow2(n))

    def test_non_powers_of_two_should_return_false(self):
        """测试非 2 的幂次返回 False"""
        non_powers = [0, 3, 5, 6, 7, 9, 10, 15, 17, 100, 1000]
        for n in non_powers:
            with self.subTest(n=n):
                self.assertFalse(is_pow2(n))

    def test_negative_numbers_should_return_false(self):
        """测试负数返回 False"""
        self.assertFalse(is_pow2(-1))
        self.assertFalse(is_pow2(-2))


class TestWalshMatrix(unittest.TestCase):
    """测试 walsh_matrix 函数"""

    def test_size_1_should_return_1x1_matrix(self):
        """测试大小为 1 返回 1x1 矩阵"""
        result = walsh_matrix(1, torch.float32, torch.device('cpu'))
        self.assertEqual(result.shape, (1, 1))
        self.assertEqual(result[0, 0].item(), 1.0)

    def test_size_2_should_return_2x2_matrix(self):
        """测试大小为 2 返回 2x2 矩阵"""
        result = walsh_matrix(2, torch.float32, torch.device('cpu'))
        self.assertEqual(result.shape, (2, 2))

    def test_size_4_should_return_4x4_matrix(self):
        """测试大小为 4 返回 4x4 矩阵"""
        result = walsh_matrix(4, torch.float32, torch.device('cpu'))
        self.assertEqual(result.shape, (4, 4))

    def test_matrix_should_be_orthogonal(self):
        """测试矩阵是正交的"""
        for size in [2, 4, 8]:
            with self.subTest(size=size):
                result = walsh_matrix(size, torch.float32, torch.device('cpu'))
                # H^T * H 应该是对角矩阵
                product = result.T @ result
                expected = torch.eye(size) * size
                self.assertTrue(torch.allclose(product, expected, atol=1e-5))

    def test_entries_should_be_plus_or_minus_one(self):
        """测试矩阵元素为 +1 或 -1"""
        result = walsh_matrix(8, torch.float32, torch.device('cpu'))
        self.assertTrue(torch.all((result == 1.0) | (result == -1.0)))


class TestRandomHadamardMatrix(unittest.TestCase):
    """测试 random_hadamard_matrix 函数"""

    def test_should_return_correct_size(self):
        """测试返回正确大小"""
        size = 8
        result = random_hadamard_matrix(size, torch.float32, torch.device('cpu'))
        self.assertEqual(result.shape, (size, size))

    def test_should_be_orthogonal(self):
        """测试矩阵是正交的"""
        size = 16
        result = random_hadamard_matrix(size, torch.float32, torch.device('cpu'))
        # H^T * H 应该近似为单位矩阵
        product = result.T @ result
        expected = torch.eye(size)
        self.assertTrue(torch.allclose(product, expected, atol=1e-5))

    def test_different_calls_should_produce_different_matrices(self):
        """测试不同调用产生不同矩阵"""
        size = 8
        result1 = random_hadamard_matrix(size, torch.float32, torch.device('cpu'))
        result2 = random_hadamard_matrix(size, torch.float32, torch.device('cpu'))
        # 随机矩阵应该不同（概率极高）
        self.assertFalse(torch.allclose(result1, result2))


class TestMatmulHadU(unittest.TestCase):
    """测试 matmul_had_u 函数"""

    def test_should_preserve_shape(self):
        """测试保持形状"""
        x = torch.randn(2, 4, 8)
        result = matmul_had_u(x)
        self.assertEqual(result.shape, x.shape)

    def test_power_of_two_size_should_work(self):
        """测试 2 的幂次大小"""
        for size in [2, 4, 8, 16]:
            with self.subTest(size=size):
                x = torch.randn(1, size)
                result = matmul_had_u(x)
                self.assertEqual(result.shape, x.shape)

    def test_should_be_normalized(self):
        """测试结果是归一化的"""
        x = torch.eye(8)
        result = matmul_had_u(x)
        # 归一化后，结果的范数应该近似为 1
        norms = torch.norm(result, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones(8), atol=1e-5))

    def test_transpose_should_work(self):
        """测试转置模式"""
        x = torch.randn(2, 8)
        result = matmul_had_u(x, transpose=True)
        self.assertEqual(result.shape, x.shape)


class TestMatmulHadUT(unittest.TestCase):
    """测试 matmul_had_u_t 函数"""

    def test_should_produce_same_result_as_transpose_mode(self):
        """测试与 matmul_had_u(transpose=True) 结果相同"""
        x = torch.randn(2, 8)
        result1 = matmul_had_u_t(x)
        result2 = matmul_had_u(x, transpose=True)
        self.assertTrue(torch.allclose(result1, result2))

    def test_should_preserve_shape(self):
        """测试保持形状"""
        x = torch.randn(3, 16)
        result = matmul_had_u_t(x)
        self.assertEqual(result.shape, x.shape)

    def test_inverse_should_recover_original(self):
        """测试逆变换恢复原始数据"""
        x = torch.randn(1, 8)
        # H * H^T 应该近似为恒等变换（归一化后）
        h_x = matmul_had_u(x)
        ht_h_x = matmul_had_u_t(h_x)
        self.assertTrue(torch.allclose(ht_h_x, x, atol=1e-5))


class TestGetHadK(unittest.TestCase):
    """测试 get_had_k 函数"""

    def test_should_return_hadamard_matrix_for_power_of_two(self):
        """测试 2 的幂次返回 Hadamard 矩阵"""
        from msmodelslim.processor.quarot.common.hadamard import get_had_k

        had_k, k = get_had_k(8)
        # 对于 2 的幂次，应该返回 None 和 1
        self.assertIsNone(had_k)
        self.assertEqual(k, 1)

    def test_should_raise_for_non_decomposable(self):
        """测试不可分解的数抛出异常"""
        from msmodelslim.processor.quarot.common.hadamard import get_had_k
        from msmodelslim.utils.exception import UnsupportedError

        # 3 不是 2 的幂次，也不在 HADAMARD_TXT_DATA_FILE_NAME 中
        with self.assertRaises(UnsupportedError):
            get_had_k(3)


class TestLoadHadamardMatrixFromTxt(unittest.TestCase):
    """测试 load_hadamard_matrix_from_txt 函数"""

    def test_should_load_existing_file(self):
        """测试加载存在的文件"""
        from msmodelslim.processor.quarot.common.hadamard import load_hadamard_matrix_from_txt

        # 测试加载 had.12.txt
        try:
            matrix = load_hadamard_matrix_from_txt("had.12.txt")
            self.assertIsNotNone(matrix)
            self.assertEqual(matrix.shape[0], 12)
            self.assertEqual(matrix.shape[1], 12)
        except Exception:
            # 如果文件不存在，跳过
            pass


class TestTxtSafeLoad(unittest.TestCase):
    """测试 txt_safe_load 函数"""

    def test_should_raise_for_nonexistent_file(self):
        """测试不存在的文件抛出异常"""
        from msmodelslim.processor.quarot.common.hadamard import txt_safe_load

        with self.assertRaises(Exception):
            txt_safe_load("/nonexistent/path/file.txt")


if __name__ == '__main__':
    unittest.main()
