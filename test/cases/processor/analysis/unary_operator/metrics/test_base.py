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

Unit tests for the abstract UnaryAnalysisMethod contract and defaults.
"""

# pylint: disable=abstract-class-instantiated

import inspect
import unittest

from msmodelslim.processor.analysis.unary_operator.metrics.base import UnaryAnalysisMethod


class _Method(UnaryAnalysisMethod):
    @property
    def name(self):
        return "delta_unary"

    def get_hook(self):
        return lambda *args, **kwargs: None

    def compute_score(self, layer_data):
        return float(layer_data["v"])


class TestUnaryAnalysisMethodContract(unittest.TestCase):
    def test_base_class_is_abstract(self):
        self.assertTrue(inspect.isabstract(UnaryAnalysisMethod))
        with self.assertRaises(TypeError):
            UnaryAnalysisMethod()

    def test_compute_score_uses_layer_data(self):
        self.assertEqual(_Method().compute_score({"v": 3}), 3.0)

    def test_default_pack_returns_none(self):
        self.assertIsNone(_Method().pack_stats_for_distributed_merge({"v": 1}))

    def test_default_merge_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError) as ctx:
            _Method().merge_distributed_stats([{"v": 1}])
        self.assertIn("delta_unary", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
