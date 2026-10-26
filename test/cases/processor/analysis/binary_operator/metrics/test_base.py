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

Unit tests for the abstract BinaryAnalysisMethod contract.
"""

# pylint: disable=abstract-class-instantiated

import inspect
import unittest

from msmodelslim.processor.analysis.binary_operator.metrics.base import BinaryAnalysisMethod


class TestBinaryAnalysisMethodContract(unittest.TestCase):
    def test_base_class_is_abstract(self):
        self.assertTrue(inspect.isabstract(BinaryAnalysisMethod))
        with self.assertRaises(TypeError):
            BinaryAnalysisMethod()

    def test_subclass_without_compute_score_stays_abstract(self):
        class _Incomplete(BinaryAnalysisMethod):
            @property
            def name(self):
                return "incomplete"

            def get_hook(self):
                return None

        with self.assertRaises(TypeError):
            _Incomplete()

    def test_concrete_subclass_exposes_name_score_and_hook(self):
        class _Complete(BinaryAnalysisMethod):
            @property
            def name(self):
                return "delta"

            def get_hook(self):
                return lambda *args, **kwargs: "hooked"

            def compute_score(self, layer_data_before, layer_data_after):
                return layer_data_after["v"] - layer_data_before["v"]

        method = _Complete()
        self.assertEqual(method.name, "delta")
        self.assertEqual(method.compute_score({"v": 1.0}, {"v": 3.5}), 2.5)
        self.assertEqual(method.get_hook()(None, None, None), "hooked")


if __name__ == "__main__":
    unittest.main()
