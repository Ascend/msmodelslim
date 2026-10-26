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

Unit tests for LayerWiseMethodFactory registration and creation.
"""

import unittest

from msmodelslim.processor.analysis.binary_operator_layer_wise.metrics.factory import LayerWiseMethodFactory
from msmodelslim.processor.analysis.binary_operator_layer_wise.metrics.mse import (
    MSELayerWiseAnalysisMethod,
)
from msmodelslim.utils.exception import UnsupportedError


class TestLayerWiseMethodFactory(unittest.TestCase):
    def test_mse_layer_wise_is_registered(self):
        self.assertIn("mse_layer_wise", LayerWiseMethodFactory.get_supported_methods())

    def test_create_method_returns_mse_layer_wise_impl(self):
        method = LayerWiseMethodFactory.create_method("mse_layer_wise")
        self.assertIsInstance(method, MSELayerWiseAnalysisMethod)
        self.assertEqual(method.name, "mse_layer_wise")

    def test_create_method_rejects_unknown_name(self):
        with self.assertRaises(UnsupportedError):
            LayerWiseMethodFactory.create_method("no_such_metric")


if __name__ == "__main__":
    unittest.main()
