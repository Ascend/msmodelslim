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

Unit tests for StdAnalysisMethod scoring, zero-std handling and merge.
"""

import unittest

import torch

from msmodelslim.processor.analysis.unary_operator.metrics.std import StdAnalysisMethod, _as_float


class TestAsFloat(unittest.TestCase):
    def test_tensor_and_scalar_are_converted(self):
        self.assertEqual(_as_float(torch.tensor(3)), 3.0)
        self.assertEqual(_as_float(2), 2.0)


class TestStdAnalysisMethod(unittest.TestCase):
    def setUp(self):
        self.method = StdAnalysisMethod()

    def test_name_and_supports_distributed(self):
        self.assertEqual(self.method.name, "std")
        self.assertTrue(self.method.supports_distributed)

    def test_compute_score_divides_abs_max_by_std(self):
        score = self.method.compute_score({"t_max": 4.0, "t_min": -2.0, "std": 2.0})
        self.assertEqual(score, 2.0)

    def test_compute_score_zero_std_returns_abs_max_then_zero(self):
        self.assertEqual(self.method.compute_score({"t_max": -5.0, "t_min": 0.0, "std": 0}), 5.0)
        self.assertEqual(self.method.compute_score({"t_max": 0.0, "t_min": 0.0, "std": 0}), 0.0)

    def test_pack_and_merge_reduce_with_max_min_max(self):
        first = self.method.pack_stats_for_distributed_merge({"t_max": 4.0, "t_min": -2.0, "std": 1.0})
        second = self.method.pack_stats_for_distributed_merge({"t_max": 3.0, "t_min": -5.0, "std": 2.0})
        self.assertEqual(first, {"t_max": 4.0, "t_min": -2.0, "std": 1.0})

        merged = self.method.merge_distributed_stats([first, second])
        self.assertEqual(merged, {"t_max": 4.0, "t_min": -5.0, "std": 2.0})


if __name__ == "__main__":
    unittest.main()
