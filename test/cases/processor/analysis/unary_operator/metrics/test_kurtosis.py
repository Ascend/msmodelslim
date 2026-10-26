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

Unit tests for the kurtosis helper and KurtosisAnalysisMethod.
"""

import unittest

import torch
from torch import nn

from msmodelslim.processor.analysis.unary_operator.metrics.kurtosis import (
    KurtosisAnalysisMethod,
    kurtosis,
)


class TestKurtosisFunction(unittest.TestCase):
    def test_two_point_distribution_has_excess_kurtosis_minus_two(self):
        self.assertAlmostEqual(float(kurtosis(torch.tensor([0.0, 1.0]))), -2.0, places=6)


class TestKurtosisAnalysisMethod(unittest.TestCase):
    def test_name_and_supports_distributed(self):
        method = KurtosisAnalysisMethod()
        self.assertEqual(method.name, "kurtosis")
        self.assertTrue(method.supports_distributed)

    def test_compute_score_concatenates_collected_tensors(self):
        method = KurtosisAnalysisMethod()
        layer_data = {"tensor": [torch.tensor([0.0]), torch.tensor([1.0])]}
        self.assertAlmostEqual(method.compute_score(layer_data), -2.0, places=6)

    def test_hook_sorts_full_input_when_shorter_than_sample_step(self):
        method = KurtosisAnalysisMethod(sample_step=100)
        stats = {}
        method.get_hook()(nn.Identity(), (torch.tensor([[3.0, 1.0, 2.0]]),), None, "block.linear", stats)
        sampled = stats["block.linear"]["tensor"][0]
        self.assertTrue(torch.equal(sampled.reshape(-1), torch.tensor([1.0, 2.0, 3.0])))

    def test_matches_linear_and_conv2d_only(self):
        method = KurtosisAnalysisMethod()
        self.assertTrue(method._matches(nn.Linear(2, 2)))
        self.assertTrue(method._matches(nn.Conv2d(1, 1, 1)))
        self.assertFalse(method._matches(nn.ReLU()))


if __name__ == "__main__":
    unittest.main()
