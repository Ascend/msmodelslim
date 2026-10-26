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

Unit tests for the abstract RaCompressAnalysisInterface contract.
"""

# pylint: disable=abstract-class-instantiated

import inspect
import unittest

from msmodelslim.processor.analysis.unary_operator.metrics.ra_compress.interface import (
    RaCompressAnalysisInterface,
)


class TestRaCompressAnalysisInterface(unittest.TestCase):
    def test_interface_is_abstract(self):
        self.assertTrue(inspect.isabstract(RaCompressAnalysisInterface))
        with self.assertRaises(TypeError):
            RaCompressAnalysisInterface()

    def test_partial_subclass_stays_abstract(self):
        class _OnlyProj(RaCompressAnalysisInterface):
            def get_proj_names(self):
                return {"q": "q_proj"}

        with self.assertRaises(TypeError):
            _OnlyProj()

    def test_concrete_subclass_exposes_proj_names_and_tokenizer(self):
        class _Adapter(RaCompressAnalysisInterface):
            def get_proj_names(self):
                return {"q": "q_proj", "k": "k_proj"}

            def get_tokenizer(self):
                return "tokenizer"

        adapter = _Adapter()
        self.assertIsInstance(adapter, RaCompressAnalysisInterface)
        self.assertEqual(adapter.get_proj_names(), {"q": "q_proj", "k": "k_proj"})
        self.assertEqual(adapter.get_tokenizer(), "tokenizer")


if __name__ == "__main__":
    unittest.main()
