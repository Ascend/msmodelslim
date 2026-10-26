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

Unit tests for DefaultMSEModelWiseBlockData key fallthrough and resolver.
"""

import unittest

import torch

from msmodelslim.processor.analysis.binary_operator_model_wise.metrics.mse_model_wise.block_data import (
    DefaultMSEModelWiseBlockData,
    resolve_mse_model_wise_block_data,
)


class TestDefaultMSEModelWiseBlockDataExtra(unittest.TestCase):
    def setUp(self):
        self.block_data = DefaultMSEModelWiseBlockData()

    def test_extract_hidden_states_from_dict_with_last_hidden_state(self):
        tensor = torch.randn(3, 4)
        out = self.block_data.extract_hidden_states({"last_hidden_state": tensor})
        self.assertTrue(torch.equal(out, tensor))

    def test_extract_hidden_states_skips_non_tensor_value_and_falls_through(self):
        tensor = torch.randn(2, 4)
        out = self.block_data.extract_hidden_states({"hidden_states": "not-a-tensor", "pooler_output": tensor})
        self.assertTrue(torch.equal(out, tensor))

    def test_resolve_returns_fresh_default_instance_when_adapter_absent(self):
        first = resolve_mse_model_wise_block_data(None)
        second = resolve_mse_model_wise_block_data(None)
        self.assertIsInstance(first, DefaultMSEModelWiseBlockData)
        self.assertIsInstance(second, DefaultMSEModelWiseBlockData)
        self.assertIsNot(first, second)


if __name__ == "__main__":
    unittest.main()
