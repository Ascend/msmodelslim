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

import struct
import unittest

import pytest
import torch

from msmodelslim.core.quant_service.modelslim_v1.save.utils.pack import (
    pack_fp4_to_uint8,
    pack_hif4_weight_to_uint8,
    pack_hif4_scale_e1_to_uint8,
    pack_hif4_scale_to_fp32,
)


class TestPackUtils(unittest.TestCase):
    def test_pack_fp4_to_uint8_matches_expected_when_exact_values(self):
        x = torch.tensor([[0.0, 0.5, -1.0, -6.0]], dtype=torch.float32)

        packed = pack_fp4_to_uint8(x)

        expected = torch.tensor([[16, 250]], dtype=torch.uint8)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.tolist(), expected.tolist())

    def test_pack_fp4_to_uint8_picks_lower_index_when_tie_values(self):
        # 0.75 / 1.25 are tie cases and should pick lower index due to argmin behavior.
        x = torch.tensor([[0.75, 1.25, 2.6, -3.4]], dtype=torch.float32)

        packed = pack_fp4_to_uint8(x)

        expected = torch.tensor([[33, 213]], dtype=torch.uint8)
        self.assertEqual(packed.tolist(), expected.tolist())

    def test_pack_fp4_to_uint8_preserves_shape_when_multi_row(self):
        x = torch.tensor(
            [
                [4.2, -0.4, 1.49, -1.51],
                [-0.1, 6.1, -2.2, 2.9],
            ],
            dtype=torch.float32,
        )

        packed = pack_fp4_to_uint8(x)

        expected = torch.tensor(
            [
                [150, 179],
                [120, 92],
            ],
            dtype=torch.uint8,
        )
        self.assertEqual(tuple(packed.shape), (2, 2))
        self.assertEqual(packed.tolist(), expected.tolist())

    def test_pack_fp4_to_uint8_raises_runtime_error_when_odd_n(self):
        x = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float32)

        with self.assertRaises(RuntimeError):
            pack_fp4_to_uint8(x)


class TestPackHiF4Utils(unittest.TestCase):
    float_to_e1m2 = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

    def test_exact_value_and_sign_mapping(self):
        positives = torch.tensor(self.float_to_e1m2, dtype=torch.float32).reshape(1, -1)
        negatives = -positives
        x = torch.cat([positives, negatives], dim=1)
        packed = pack_hif4_weight_to_uint8(x)

        pos_indices = torch.arange(8, dtype=torch.long)
        neg_indices = pos_indices + 8
        all_indices = torch.stack([pos_indices, neg_indices]).reshape(-1, 2)
        expected = (all_indices[:, 0] | (all_indices[:, 1] << 4)).to(torch.uint8).reshape(1, 8)

        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.tolist(), expected.tolist())

    def test_nearest_value_mapping(self):
        x = torch.tensor([[0.125, 0.375, 1.125, -0.125]], dtype=torch.float32)
        packed = pack_hif4_weight_to_uint8(x)

        expected = torch.tensor([[16, 132]], dtype=torch.uint8)
        self.assertEqual(packed.tolist(), expected.tolist())

    def test_multi_row_shape_and_values(self):
        x = torch.tensor(
            [
                [0.0, 1.75, -1.0, 0.25],
                [-0.5, 1.5, 0.75, -0.75],
            ],
            dtype=torch.float32,
        )
        packed = pack_hif4_weight_to_uint8(x)
        self.assertEqual(packed.shape, (2, 2))
        expected = torch.tensor([[112, 28], [106, 179]], dtype=torch.uint8)
        self.assertEqual(packed.tolist(), expected.tolist())

    def test_odd_n_raises_runtime_error(self):
        x = torch.tensor([[0.0, 0.25, 0.5]], dtype=torch.float32)  # 3 columns, odd
        with pytest.raises(RuntimeError):
            pack_hif4_weight_to_uint8(x)


class TestPackHif4ScaleE1(unittest.TestCase):
    def test_1d_exact_multiple_of_8(self):
        """Aggregate 8 elements into one uint8 with big-endian bit order."""
        vals = torch.tensor([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0], dtype=torch.float32)
        out = pack_hif4_scale_e1_to_uint8(vals)
        self.assertEqual(out.dtype, torch.uint8)
        self.assertEqual(out.shape, (1,))
        self.assertEqual(out.item(), 85)

    def test_3d_shape(self):
        """Test 3D tensor aggregation."""
        vals = torch.full((2, 3, 8), 2.0, dtype=torch.float32)
        out = pack_hif4_scale_e1_to_uint8(vals)
        self.assertEqual(out.shape, (6,))
        assert torch.equal(out, torch.full((6,), 255, dtype=torch.uint8))

    def test_non_multiple_of_8_raises_runtime_error(self):
        """If last dimension is not divisible by 8, view(-1,8) raises RuntimeError."""
        vals = torch.ones(9, dtype=torch.float32)
        with pytest.raises(RuntimeError):
            pack_hif4_scale_e1_to_uint8(vals)

    def test_mixed_values_2d(self):
        """Verify packing of known pattern in 2D."""
        vals = torch.tensor(
            [
                [1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            ],
            dtype=torch.float32,
        )
        out = pack_hif4_scale_e1_to_uint8(vals)
        expected = torch.tensor([99, 255], dtype=torch.uint8)
        self.assertEqual(out.tolist(), expected.tolist())


# ---------- pack_hif4_scale_to_fp32 tests ----------
class TestPackHif4ScaleToFp32(unittest.TestCase):
    @staticmethod
    def int_bits_to_float32(bits: int) -> float:
        """将 32 位无符号整数位模式转换为 float32"""
        return struct.unpack('f', struct.pack('I', bits & 0xFFFFFFFF))[0]

    def test_basic_packing(self):
        """Test the whole packing chain: scale_factor, lv2, lv3 -> float32 bits."""
        scale_factor = torch.tensor([1.0, 2.0], dtype=torch.float32).reshape(2, 1, 1, 1)
        lv2 = torch.tensor(
            [[[[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]]], [[[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]]]],
            dtype=torch.float32,
        )
        lv3 = torch.ones(2, 1, 1, 16, dtype=torch.float32)
        result = pack_hif4_scale_to_fp32(scale_factor, lv2, lv3)
        result_bits = result.view(torch.int32)

        expected_bits = torch.tensor(
            [
                -1068171264,
                -989921280,
            ],
            dtype=torch.int32,
        )
        self.assertTrue(
            torch.equal(result_bits, expected_bits),
            msg=(
                f"\nExpected bits : {[hex(i & 0xFFFFFFFF) for i in expected_bits.tolist()]}"
                f"\nActual bits   : {[hex(i & 0xFFFFFFFF) for i in result_bits.tolist()]}"
            ),
        )

    def test_e6m2_boundary_values_when_bias48(self):
        """Verify the to_e6m2 conversion including carry and clamp."""
        scale_factor = torch.tensor([0.25, 1.96875, 1e-10, 1e10], dtype=torch.float32).reshape(4, 1, 1, 1)
        lv2 = torch.ones(4, 1, 1, 8, dtype=torch.float32)
        lv3 = torch.ones(4, 1, 1, 16, dtype=torch.float32)
        result = pack_hif4_scale_to_fp32(scale_factor, lv2, lv3).squeeze()
        self.assertEqual(result.shape, (4,))
        expected_bits = [-1207959552, -1006632960, 989855744, -33554432]
        expected_float = torch.tensor([self.int_bits_to_float32(b) for b in expected_bits], dtype=torch.float32)
        assert torch.equal(result, expected_float)

    def test_e6m2_clamp_to_min_max_when_out_of_range(self):
        scale_factor = torch.tensor([2 ** (-50), 2**20], dtype=torch.float32).reshape(2, 1, 1, 1)

        lv2 = torch.ones(2, 1, 1, 8)
        lv3 = torch.ones(2, 1, 1, 16)

        result = pack_hif4_scale_to_fp32(scale_factor, lv2, lv3)

        expected_bits = [
            0,
            -33554432,
        ]
        expected = torch.tensor([self.int_bits_to_float32(x) for x in expected_bits], dtype=torch.float32)

        assert torch.equal(result, expected)

    def test_lv2_lv3_squeeze_alignment(self):
        """When scale_factor has singleton dim, squeeze must propagate correctly."""
        scale_factor = torch.tensor([[1.5]], dtype=torch.float32)
        lv2 = torch.tensor([[1.0] * 8], dtype=torch.float32)
        lv3 = torch.tensor([[1.0] * 16], dtype=torch.float32)
        result = pack_hif4_scale_to_fp32(scale_factor, lv2, lv3)
        self.assertEqual(result.numel(), 1)

    def test_negative_scale_factor_raises(self):
        """Log2 of negative values yields nan/inf; function may not guard."""
        scale_factor = torch.tensor([-1.0], dtype=torch.float32)
        lv2 = torch.ones(1, 8, dtype=torch.float32)
        lv3 = torch.ones(1, 16, dtype=torch.float32)
        result = pack_hif4_scale_to_fp32(scale_factor, lv2, lv3)
        assert isinstance(result, torch.Tensor)


class TestPackInt4Helpers(unittest.TestCase):
    """Tests for _pack_int4 / w4a8_pack_int4 / process_scale."""

    def test_pack_int4_3d_expert_shape_when_three_dims(self):
        from msmodelslim.core.quant_service.modelslim_v1.save.utils.pack import _pack_int4

        weight = torch.arange(8, dtype=torch.int8).reshape(1, 2, 4)
        packed = _pack_int4(weight)
        self.assertEqual(packed.shape, (1, 2, 2))

    def test_w4a8_pack_int4_transposes_and_packs_when_2d(self):
        from msmodelslim.core.quant_service.modelslim_v1.save.utils.pack import w4a8_pack_int4

        w = torch.zeros(4, 4, dtype=torch.int8)
        out = w4a8_pack_int4(w)
        self.assertEqual(out.dtype, torch.int8)

    def test_process_scale_up_proj_branch_when_name_contains_up_proj(self):
        from msmodelslim.core.quant_service.modelslim_v1.save.utils.pack import process_scale

        bias = torch.ones(2, 4)
        out = process_scale("model.layers.0.up_proj", bias, tp_num=2)
        self.assertEqual(out.shape[0], 2)

    def test_process_scale_down_proj_branch_when_name_contains_down_proj(self):
        from msmodelslim.core.quant_service.modelslim_v1.save.utils.pack import process_scale

        bias = torch.ones(4, 8)
        out = process_scale("model.layers.0.down_proj", bias, tp_num=2)
        self.assertEqual(out.shape, (4, 2))
