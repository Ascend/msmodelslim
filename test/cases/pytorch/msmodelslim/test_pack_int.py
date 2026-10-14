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

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.complex_quantifier import (
    _pack_int4,
    _w4a16_pack_int4,
    _w4a8_pack_int4,
    process_scale,
)


class TestComplexQuantifier(unittest.TestCase):
    def test_pack_int4_2d(self):
        """Test pack_int4 function for 2D tensor"""
        # Prepare input data: 2x4 tensor with small values to avoid overflow
        weight = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int8)

        result = _pack_int4(weight)

        # Verify output shape
        self.assertEqual(result.shape, (2, 2))

        # Verify packed results
        # [1,2] -> (2<<4)|1 = 33 (0x21)
        # [3,4] -> (4<<4)|3 = 67 (0x43)
        # [5,6] -> (6<<4)|5 = 101 (0x65)
        # [7,8] -> (8<<4)|7 = -121 (0x87, due to int8 overflow)
        expected = torch.tensor([[33, 67], [101, -121]], dtype=torch.int8)
        assert torch.equal(result, expected), f"Expected {expected}, but got {result}"

    def test_pack_int4_3d(self):
        """Test pack_int4 function for 3D tensor"""
        # Prepare input data: 2x2x4 tensor with small values to avoid overflow
        weight = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]], [[1, 2, 3, 4], [5, 6, 7, 8]]], dtype=torch.int8)

        result = _pack_int4(weight)

        # Verify output shape
        self.assertEqual(result.shape, (2, 2, 2))

        expected = torch.tensor([[[33, 67], [101, -121]], [[33, 67], [101, -121]]], dtype=torch.int8)
        assert torch.equal(result, expected), f"Expected {expected}, but got {result}"

    def test_pack_int4_odd_dimension(self):
        """Test case for non-even dimension"""
        weight = torch.tensor(
            [
                [1, 2, 3]  # 3 is not even
            ],
            dtype=torch.int8,
        )

        with self.assertRaises(AssertionError):
            _pack_int4(weight)

    def test_w4a16_pack_int4_no_transpose(self):
        """Test w4a16_pack_int4 function without transpose"""
        weight = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int8)

        result = _w4a16_pack_int4(weight, trans_flag=False)

        # Check output shape
        assert result.shape == (1, 4), f"Expected shape (1, 4), but got {result.shape}"

        expected = torch.tensor([[81, 98, 115, -124]], dtype=torch.int8)

        assert torch.equal(result, expected), f"Expected {expected}, but got {result}"

    def test_w4a16_pack_int4_with_transpose(self):
        """Test w4a16_pack_int4 function with transpose"""
        weight = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int8)

        result = _w4a16_pack_int4(weight, trans_flag=True)

        # Check output shape
        assert result.shape == (2, 2), f"Expected shape (2, 2), but got {result.shape}"

        # Expected results after transpose:
        # [1,2,3,4] -> [(2<<4)|1, (4<<4)|3] -> [33, 67]
        # [5,6,7,8] -> [(6<<4)|5, (8<<4)|7] -> [101, -121]
        expected = torch.tensor([[33, 67], [101, -121]], dtype=torch.int8)

        assert torch.equal(result, expected), f"Expected {expected}, but got {result}"

    def test_w4a16_pack_int4_large_matrix(self):
        """Test with larger matrix"""
        # Create a larger test matrix 4x8
        weight = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [9, 10, 11, 12, 13, 14, 15, 16],
                [1, 2, 3, 4, 5, 6, 7, 8],
                [9, 10, 11, 12, 13, 14, 15, 16],
            ],
            dtype=torch.int8,
        )

        result = _pack_int4(weight)

        # Verify output shape
        self.assertEqual(result.shape, (4, 4))

        # Calculate expected results
        # [1,2] -> (2<<4)|1 = 33
        # [3,4] -> (4<<4)|3 = 67
        # [5,6] -> (6<<4)|5 = 101
        # [7,8] -> (8<<4)|7 = -121
        # [9,10] -> (10<<4)|9 = -87
        # [11,12] -> (12<<4)|11 = -53
        # [13,14] -> (14<<4)|13 = -19
        # [15,16] -> (16<<4)|15 = 15 (overflow)
        expected = torch.tensor(
            [[33, 67, 101, -121], [-87, -53, -19, 15], [33, 67, 101, -121], [-87, -53, -19, 15]], dtype=torch.int8
        )
        assert torch.equal(result, expected), f"Expected {expected}, but got {result}"

        # ================== 测试 _w4a8_pack_int4 函数 ==================

    def test_w4a8_pack_int4_2d(self):
        """Test _w4a8_pack_int4 with 2D tensor"""
        weight = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int8)

        # 手动计算预期结果：转置最后两维（即全部转置）
        transposed = weight.transpose(-1, -2)  # (2,4) -> (4,2)
        packed = _pack_int4(transposed)  # (4,2) -> (4,1)
        expected = packed.transpose(-1, -2)  # (4,1) -> (1,4)

        result = _w4a8_pack_int4(weight)
        self.assertEqual(result.shape, expected.shape)
        self.assertTrue(torch.equal(result, expected))

    def test_w4a8_pack_int4_3d(self):
        """Test _w4a8_pack_int4 with 3D tensor (含专家维度)"""
        weight = torch.tensor(
            [
                [[1, 2, 3, 4], [5, 6, 7, 8]],  # 专家1，形状 (2,4)
                [[9, 10, 11, 12], [13, 14, 15, 16]],  # 专家2，形状 (2,4)
            ],
            dtype=torch.int8,
        )  # 形状 (2,2,4)

        # 手动计算预期结果
        transposed = weight.transpose(-1, -2)  # (2,2,4) -> (2,4,2)
        packed = _pack_int4(transposed)  # (2,4,2) -> (2,4,1)
        expected = packed.transpose(-1, -2)  # (2,4,1) -> (2,1,4)

        result = _w4a8_pack_int4(weight)
        self.assertEqual(result.shape, expected.shape)

        # 验证每个专家的结果
        for i in range(weight.shape[0]):
            self.assertTrue(torch.equal(result[i], expected[i]))
        # ================== 测试 process_scale 函数 ==================

    def test_process_scale_up_proj(self):
        """Test process_scale for up_proj/gate_proj (sum dim=1)"""
        name = "up_proj.weight"
        tp_num = 16
        test_bias = torch.tensor([[0.4, 0.8], [1.2, 1.6]], dtype=torch.float32)
        bias = process_scale(name, test_bias, tp_num)

        expected = torch.tensor([[9.6], [22.4]], dtype=torch.float32)
        self.assertTrue(torch.allclose(bias, expected, atol=1e-6))

    def test_process_scale_down_proj(self):
        """Test process_scale for down_proj (sum across tp_num groups)"""
        name = "down_proj.weight"
        tp_num = 1
        test_bias = torch.tensor([[0.4, 0.8], [1.2, 1.6]], dtype=torch.float32)
        bias = process_scale(name, test_bias, tp_num)

        expected = torch.tensor([[9.6], [22.4]], dtype=torch.float32)
        self.assertTrue(torch.allclose(bias, expected, atol=1e-6))

    def test_process_scale_down_proj_2(self):
        """Test process_scale for down_proj (sum across tp_num groups)"""
        name = "down_proj.weight"
        test_bias = torch.tensor([[0.4, 0.8], [1.2, 1.6]], dtype=torch.float32)
        tp_num = 2
        bias = process_scale(name, test_bias, tp_num)

        expected = torch.tensor([[3.2, 6.4], [9.6, 12.8]], dtype=torch.float32)
        self.assertTrue(torch.allclose(bias, expected, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
