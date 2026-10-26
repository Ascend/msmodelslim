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

Unit tests for MSELayerWiseAnalysisMethod scoring on synthetic tensors.
"""

import unittest

import torch

from msmodelslim.processor.analysis.binary_operator_layer_wise.metrics.mse import (
    MSELayerWiseAnalysisMethod,
)


class TestMSELayerWiseAnalysisMethod(unittest.TestCase):
    def setUp(self):
        self.method = MSELayerWiseAnalysisMethod()

    def test_name_and_supports_distributed(self):
        self.assertEqual(self.method.name, "mse_layer_wise")
        self.assertTrue(self.method.supports_distributed)

    def test_compute_score_unwraps_nested_tuple_and_averages(self):
        ref = [((torch.zeros(1, 2),),)]
        cand = [((torch.ones(1, 2),),)]
        self.assertAlmostEqual(self.method.compute_score(ref, cand), 1.0, places=6)

    def test_compute_score_returns_zero_when_no_tensor_pair(self):
        self.assertEqual(self.method.compute_score([("x",)], [("y",)]), 0.0)
        self.assertEqual(self.method.compute_score([], []), 0.0)

    def test_get_hook_is_none_for_layer_wise_method(self):
        self.assertIsNone(self.method.get_hook())


if __name__ == "__main__":
    unittest.main()
