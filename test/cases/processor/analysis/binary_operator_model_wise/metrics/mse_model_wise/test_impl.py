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

Unit tests for MSEModelWiseAnalysisMethod pair skipping and zip truncation.
"""

import unittest

import torch

from msmodelslim.processor.analysis.binary_operator_model_wise.metrics.mse_model_wise.impl import (
    MSEModelWiseAnalysisMethod,
)


class TestMSEModelWiseAnalysisMethodExtra(unittest.TestCase):
    def setUp(self):
        self.method = MSEModelWiseAnalysisMethod()

    def test_compute_score_skips_unsupported_pair_but_keeps_valid_one(self):
        ref = [{"hidden_states": torch.zeros(1, 2)}, "not-a-block-output"]
        cand = [{"hidden_states": torch.ones(1, 2)}, "not-a-block-output"]
        self.assertAlmostEqual(self.method.compute_score(ref, cand), 1.0, places=6)

    def test_compute_score_truncates_to_shortest_output_list(self):
        ref = [torch.zeros(1, 2), torch.full((1, 2), 1.0e3)]
        cand = [torch.ones(1, 2)]
        self.assertAlmostEqual(self.method.compute_score(ref, cand), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
