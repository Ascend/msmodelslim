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

Unit tests for the abstract MSEModelWiseAnalysisInterface contract.
"""

# pylint: disable=abstract-class-instantiated

import inspect
import unittest

import torch

from msmodelslim.processor.analysis.binary_operator_model_wise.metrics.mse_model_wise.interface import (
    MSEModelWiseAnalysisInterface,
)


class TestMSEModelWiseAnalysisInterface(unittest.TestCase):
    def test_interface_is_abstract(self):
        self.assertTrue(inspect.isabstract(MSEModelWiseAnalysisInterface))
        with self.assertRaises(TypeError):
            MSEModelWiseAnalysisInterface()

    def test_concrete_subclass_extracts_hidden_states(self):
        class _Adapter(MSEModelWiseAnalysisInterface):
            def extract_hidden_states(self, value):
                return value["h"]

        tensor = torch.randn(2, 3)
        adapter = _Adapter()
        self.assertIsInstance(adapter, MSEModelWiseAnalysisInterface)
        self.assertTrue(torch.equal(adapter.extract_hidden_states({"h": tensor}), tensor))


if __name__ == "__main__":
    unittest.main()
