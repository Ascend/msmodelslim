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

Unit tests for the abstract LayerWiseAnalysisMethod contract.
"""

# pylint: disable=abstract-class-instantiated

import inspect
import unittest

from msmodelslim.processor.analysis.binary_operator_layer_wise.metrics.base import (
    LayerWiseAnalysisMethod,
)


class TestLayerWiseAnalysisMethodContract(unittest.TestCase):
    def test_base_class_is_abstract(self):
        self.assertTrue(inspect.isabstract(LayerWiseAnalysisMethod))
        with self.assertRaises(TypeError):
            LayerWiseAnalysisMethod()

    def test_subclass_without_compute_score_stays_abstract(self):
        class _Incomplete(LayerWiseAnalysisMethod):
            @property
            def name(self):
                return "incomplete"

        with self.assertRaises(TypeError):
            _Incomplete()

    def test_concrete_subclass_defaults_hook_to_none(self):
        class _Complete(LayerWiseAnalysisMethod):
            @property
            def name(self):
                return "delta_layer_wise"

            def compute_score(self, ref_outputs, cand_outputs):
                return float(len(ref_outputs) + len(cand_outputs))

        method = _Complete()
        self.assertEqual(method.name, "delta_layer_wise")
        self.assertIsNone(method.get_hook())
        self.assertEqual(method.compute_score([1, 2], [3]), 3.0)


if __name__ == "__main__":
    unittest.main()
