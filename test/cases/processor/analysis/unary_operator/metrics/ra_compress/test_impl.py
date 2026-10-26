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

Unit tests for ra_compress metric_params validation and method metadata.
"""

import unittest

from msmodelslim.processor.analysis.unary_operator.metrics.ra_compress.impl import (
    RA_COMPRESS_METRIC_PARAMS,
    RaCompressAnalysisMethod,
    validate_metric_params,
)
from msmodelslim.utils.exception import SchemaValidateError


class TestValidateMetricParams(unittest.TestCase):
    def test_validate_metric_params_returns_none_when_params_empty(self):
        self.assertIsNone(validate_metric_params({}))

    def test_validate_metric_params_returns_none_when_ratios_valid(self):
        self.assertIsNone(validate_metric_params({"induction_head_ratio": 0.2, "echo_head_ratio": 0}))

    def test_unknown_key_is_rejected(self):
        with self.assertRaises(SchemaValidateError) as ctx:
            validate_metric_params({"unknown_ratio": 0.5})
        self.assertIn("unknown_ratio", str(ctx.exception))

    def test_out_of_range_is_rejected(self):
        with self.assertRaises(SchemaValidateError):
            validate_metric_params({"induction_head_ratio": 1.5})

    def test_bool_and_non_number_are_rejected(self):
        with self.assertRaises(SchemaValidateError):
            validate_metric_params({"echo_head_ratio": True})
        with self.assertRaises(SchemaValidateError):
            validate_metric_params({"echo_head_ratio": "0.1"})


class TestRaCompressAnalysisMethodMeta(unittest.TestCase):
    def test_supported_param_names(self):
        self.assertEqual(set(RA_COMPRESS_METRIC_PARAMS), {"induction_head_ratio", "echo_head_ratio"})

    def test_name_is_ra_compress_when_method_constructed(self):
        method = RaCompressAnalysisMethod(induction_head_ratio=0.25, echo_head_ratio=0.05)
        self.assertEqual(method.name, "ra_compress")

    def test_supports_distributed_is_true_when_method_constructed(self):
        method = RaCompressAnalysisMethod(induction_head_ratio=0.25, echo_head_ratio=0.05)
        self.assertTrue(method.supports_distributed)


if __name__ == "__main__":
    unittest.main()
