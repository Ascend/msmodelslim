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

import unittest

import torch

from msmodelslim.core.quant_service.modelslim_v1.save.utils.pack import pack_fp4_to_uint8


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
