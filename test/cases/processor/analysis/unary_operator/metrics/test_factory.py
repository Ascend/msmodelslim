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

Unit tests for UnaryAnalysisMethodFactory registration and creation.
"""

import unittest

from msmodelslim.processor.analysis.unary_operator.metrics.factory import (
    UnaryAnalysisMethodFactory,
)
from msmodelslim.processor.analysis.unary_operator.metrics.kurtosis import KurtosisAnalysisMethod
from msmodelslim.processor.analysis.unary_operator.metrics.quantile import QuantileAnalysisMethod
from msmodelslim.processor.analysis.unary_operator.metrics.ra_compress.impl import (
    RaCompressAnalysisMethod,
)
from msmodelslim.processor.analysis.unary_operator.metrics.std import StdAnalysisMethod
from msmodelslim.utils.exception import UnsupportedError


class TestUnaryAnalysisMethodFactory(unittest.TestCase):
    def test_supported_methods_include_all_builtin_metrics(self):
        self.assertEqual(
            set(UnaryAnalysisMethodFactory.get_supported_methods()),
            {"quantile", "std", "kurtosis", "ra_compress"},
        )

    def test_create_method_maps_each_builtin_metric(self):
        self.assertIsInstance(UnaryAnalysisMethodFactory.create_method("quantile"), QuantileAnalysisMethod)
        self.assertIsInstance(UnaryAnalysisMethodFactory.create_method("std"), StdAnalysisMethod)
        self.assertIsInstance(UnaryAnalysisMethodFactory.create_method("kurtosis"), KurtosisAnalysisMethod)
        self.assertIsInstance(UnaryAnalysisMethodFactory.create_method("ra_compress"), RaCompressAnalysisMethod)

    def test_create_method_filters_unrelated_adapter_kwarg(self):
        method = UnaryAnalysisMethodFactory.create_method("std", adapter=object())
        self.assertIsInstance(method, StdAnalysisMethod)

    def test_create_method_rejects_unknown_name(self):
        with self.assertRaises(UnsupportedError) as ctx:
            UnaryAnalysisMethodFactory.create_method("no_such_metric")
        self.assertIn("no_such_metric", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
