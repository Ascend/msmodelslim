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

Unit tests for BinaryAnalysisMethodFactory registration and creation.
"""

import unittest

from msmodelslim.processor.analysis.binary_operator.metrics.attention_mse.impl import (
    AttentionMSEAnalysisMethod,
)
from msmodelslim.processor.analysis.binary_operator.metrics.attention_mse.interface import (
    AttentionMSEAnalysisInterface,
)
from msmodelslim.processor.analysis.binary_operator.metrics.factory import BinaryAnalysisMethodFactory
from msmodelslim.utils.exception import UnsupportedError


class _Adapter(AttentionMSEAnalysisInterface):
    def get_attention_module_cls(self):
        return "TinyAttention"

    def get_attention_output_extractor(self):
        return lambda output: output


class TestBinaryAnalysisMethodFactory(unittest.TestCase):
    def test_mse_method_is_registered(self):
        self.assertIn("mse", BinaryAnalysisMethodFactory.get_supported_methods())

    def test_create_method_returns_attention_mse_impl(self):
        method = BinaryAnalysisMethodFactory.create_method("mse", adapter=_Adapter())
        self.assertIsInstance(method, AttentionMSEAnalysisMethod)
        self.assertEqual(method.name, "mse")

    def test_create_method_rejects_unknown_name(self):
        with self.assertRaises(UnsupportedError) as ctx:
            BinaryAnalysisMethodFactory.create_method("no_such_metric")
        self.assertIn("no_such_metric", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
