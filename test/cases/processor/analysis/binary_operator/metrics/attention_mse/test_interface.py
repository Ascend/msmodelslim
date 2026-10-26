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

Unit tests for the abstract AttentionMSEAnalysisInterface contract.
"""

# pylint: disable=abstract-class-instantiated

import inspect
import unittest

from msmodelslim.processor.analysis.binary_operator.metrics.attention_mse.interface import (
    AttentionMSEAnalysisInterface,
)


class TestAttentionMSEAnalysisInterface(unittest.TestCase):
    def test_interface_is_abstract(self):
        self.assertTrue(inspect.isabstract(AttentionMSEAnalysisInterface))
        with self.assertRaises(TypeError):
            AttentionMSEAnalysisInterface()

    def test_partial_subclass_stays_abstract(self):
        class _OnlyCls(AttentionMSEAnalysisInterface):
            def get_attention_module_cls(self):
                return "TinyAttention"

        with self.assertRaises(TypeError):
            _OnlyCls()

    def test_concrete_subclass_is_usable(self):
        class _Complete(AttentionMSEAnalysisInterface):
            def get_attention_module_cls(self):
                return "TinyAttention"

            def get_attention_output_extractor(self):
                return lambda output: output

        adapter = _Complete()
        self.assertIsInstance(adapter, AttentionMSEAnalysisInterface)
        self.assertEqual(adapter.get_attention_module_cls(), "TinyAttention")
        self.assertEqual(adapter.get_attention_output_extractor()("x"), "x")


if __name__ == "__main__":
    unittest.main()
