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

Unit tests for QuantileAnalysisMethod numeric score and pack/merge.
"""

import unittest

import torch

from msmodelslim.processor.analysis.unary_operator.metrics.quantile import QuantileAnalysisMethod

_ACT = torch.arange(8.0)
_EXPECTED = 2 * 7.0 / 254 / (6.0 - 2.0 + 1e-10)


class TestQuantileAnalysisMethod(unittest.TestCase):
    def test_name_and_supports_distributed(self):
        method = QuantileAnalysisMethod()
        self.assertEqual(method.name, "quantile")
        self.assertTrue(method.supports_distributed)

    def test_get_quantile_score_matches_hand_computed_value(self):
        score = QuantileAnalysisMethod.get_quantile_score(_ACT, torch.device("cpu"))
        self.assertAlmostEqual(score, _EXPECTED, places=6)

    def test_compute_score_accepts_device_string(self):
        method = QuantileAnalysisMethod()
        score = method.compute_score({"tensor": [_ACT], "device": "cpu"})
        self.assertAlmostEqual(score, _EXPECTED, places=6)

    def test_pack_and_merge_preserve_tensors_and_device(self):
        method = QuantileAnalysisMethod()
        packed = method.pack_stats_for_distributed_merge({"tensor": [torch.arange(4.0)], "device": torch.device("cpu")})
        self.assertEqual(packed["device"], "cpu")

        merged = method.merge_distributed_stats([packed])
        self.assertEqual(merged["device"], "cpu")
        self.assertEqual(len(merged["tensor"]), 1)
        self.assertTrue(torch.equal(merged["tensor"][0], torch.arange(4.0)))


if __name__ == "__main__":
    unittest.main()
