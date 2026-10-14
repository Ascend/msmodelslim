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
"""
# pylint: disable=arguments-differ,attribute-defined-outside-init,consider-using-from-import

import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config import QuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.timestep_utils import (
    _add_timestep_statistic,
    _preprocess_data,
    _run_calib_timestep,
    change_timestep_aware_quantizer_config,
    run_calib_timestep,
    load_quant_weight,
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.quantizer import TimestepQuantMixin
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.manager import TimestepManager  # noqa: F401
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import TimestepQuantConfig  # noqa: F401
from collections import defaultdict  # noqa: F401


# Create mock classes for testing
class MockTimestepQuantMixin(nn.Module, TimestepQuantMixin):
    def __init__(self):
        super().__init__()
        self.updated = False

    def update_timestep_scale_offset(self):
        self.updated = True

    def forward(self, x):
        return x


class MockLinearQuantizerTimestep(nn.Module):
    """Mock class for LinearQuantizerTimestep"""

    def __init__(self):
        super().__init__()
        self.loaded_params = {}

    def load_layer_params(self, params, device):
        self.loaded_params = params.copy()
        self.device = device


class MockTimestepAwareTensorQuantizer(nn.Module):
    """Mock class for TimestepAwareTensorQuantizer"""

    def __init__(self):
        super().__init__()
        self.max_dynamic_step = 0
        self.config_updated = False

    def update_config(self, max_dynamic_step):
        self.max_dynamic_step = max_dynamic_step
        self.config_updated = True


# Setup mocks for the import path
patch(
    'msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.timestep_utils.LinearQuantizerTimestep',
    MockLinearQuantizerTimestep,
).start()
patch(
    'msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.timestep_utils.TimestepAwareTensorQuantizer',
    MockTimestepAwareTensorQuantizer,
).start()


class TestTimestepUtils(unittest.TestCase):
    def setUp(self):
        # Create a mock model with various modules
        self.mock_module = MockTimestepQuantMixin()
        self.mock_linear_quantizer = MockLinearQuantizerTimestep()
        self.mock_tensor_quantizer = MockTimestepAwareTensorQuantizer()

        # Create model with nested modules
        self.model = nn.Sequential(
            nn.Linear(10, 10), self.mock_module, nn.Sequential(self.mock_module, nn.Linear(10, 5))
        )

        # Mock calibration data
        self.calib_data = [
            {'timestep_idx': 3, 'args': (torch.randn(1, 10),)},
            {'timestep_idx': 2, 'args': (torch.randn(1, 10),)},
            {'timestep_idx': 3, 'args': (torch.randn(1, 10),)},
            {'timestep_idx': 1, 'args': (torch.randn(1, 10),)},
        ]

        # Config
        self.quant_config = MagicMock(spec=QuantConfig)

    def test_add_timestep_statistic(self):
        # Test that update_timestep_scale_offset is called for each TimestepQuantMixin module
        _add_timestep_statistic(self.model)

        # Check that the mock modules were updated
        self.assertTrue(self.model[1].updated)
        self.assertTrue(self.model[2][0].updated)

    def test_preprocess_data(self):
        # Test data preprocessing
        result = _preprocess_data(self.calib_data)

        # Should be grouped by timestep_idx in descending order
        self.assertEqual(len(result), 3)  # 3 unique timestep indices
        self.assertEqual(result[0][0], 3)  # First group should have timestep_idx 3
        self.assertEqual(len(result[0][1]), 2)  # Two items with timestep_idx 3
        self.assertEqual(result[1][0], 2)  # Second group should have timestep_idx 2
        self.assertEqual(result[2][0], 1)  # Third group should have timestep_idx 1

    def test_preprocess_data_empty(self):
        # Test with empty data
        with self.assertRaises(ValueError):
            _preprocess_data([])

    def test_preprocess_data_missing_timestep(self):
        # Test with data missing timestep_idx
        invalid_data = [{'args': (torch.randn(1, 10),)}]
        with self.assertRaises(ValueError):
            _preprocess_data(invalid_data)

    @patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.manager.TimestepManager.set_timestep_idx')
    def test_run_calib_timestep(self, mock_set_timestep):
        # Test calibration with timestep data
        model = MagicMock()
        _run_calib_timestep(model, self.calib_data)

        # Check if timestep was set for each group
        self.assertEqual(mock_set_timestep.call_count, 3)  # Once per unique timestep

        # Check if model was called with args for each item
        self.assertEqual(model.call_count, 4)  # Once per data item

    @patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.timestep_utils._run_calib_timestep')
    def test_run_calib_timestep_wrapper(self, mock_run_calib):
        # Test the public wrapper function
        run_calib_timestep(self.model, self.calib_data, self.quant_config)

        # Check if internal function was called with correct args
        mock_run_calib.assert_called_once_with(self.model, self.calib_data)

    def test_run_calib_timestep_invalid_config(self):
        # Test with an invalid config (not a QuantConfig)
        invalid_config = {}
        with self.assertRaises(ValueError):
            run_calib_timestep(self.model, self.calib_data, invalid_config)

    def test_change_timestep_aware_quantizer_config(self):
        """Test changing timestep-aware quantizer configuration."""
        # Create a model with TimestepAwareTensorQuantizer modules
        quantizer1 = MockTimestepAwareTensorQuantizer()
        quantizer2 = MockTimestepAwareTensorQuantizer()

        model = nn.Sequential(nn.Linear(10, 10), quantizer1, nn.Sequential(nn.Linear(10, 10), quantizer2))

        # Call the function with a max_dynamic_step value
        max_dynamic_step = 50
        change_timestep_aware_quantizer_config(model, max_dynamic_step)

        # Verify that update_config was called on each quantizer with the correct value
        self.assertTrue(quantizer1.config_updated)
        self.assertEqual(quantizer1.max_dynamic_step, max_dynamic_step)
        self.assertTrue(quantizer2.config_updated)
        self.assertEqual(quantizer2.max_dynamic_step, max_dynamic_step)

    def test_load_quant_weight(self):
        """Test loading quantized weights into a model."""
        # Create a model with LinearQuantizerTimestep modules
        quantizer1 = MockLinearQuantizerTimestep()
        quantizer2 = MockLinearQuantizerTimestep()

        model = nn.Module()
        model.layer1 = quantizer1
        model.block = nn.Module()
        model.block.layer2 = quantizer2

        # Create parameters to load
        params_to_load = {
            "layer1.weight": torch.randn(10, 10),
            "layer1.bias": torch.randn(10),
            "layer1.weight_scale": torch.randn(10),
            "layer1.weight_offset": torch.randn(10),
            "block.layer2.weight": torch.randn(5, 10),
            "block.layer2.bias": torch.randn(5),
            "block.layer2.input_scale": torch.randn(10),
            "block.layer2.input_offset": torch.randn(10),
            "other.weight": torch.randn(10, 10),  # Should be ignored
        }

        # Call the function
        device = "cpu"
        load_quant_weight(params_to_load, model, device)

        # Verify that load_layer_params was called with the correct parameters
        self.assertEqual(len(quantizer1.loaded_params), 4)
        self.assertTrue(torch.equal(quantizer1.loaded_params["weight"], params_to_load["layer1.weight"]))
        self.assertTrue(torch.equal(quantizer1.loaded_params["bias"], params_to_load["layer1.bias"]))
        self.assertTrue(torch.equal(quantizer1.loaded_params["weight_scale"], params_to_load["layer1.weight_scale"]))
        self.assertTrue(torch.equal(quantizer1.loaded_params["weight_offset"], params_to_load["layer1.weight_offset"]))
        self.assertEqual(quantizer1.device, device)

        self.assertEqual(len(quantizer2.loaded_params), 4)
        self.assertTrue(torch.equal(quantizer2.loaded_params["weight"], params_to_load["block.layer2.weight"]))
        self.assertTrue(torch.equal(quantizer2.loaded_params["bias"], params_to_load["block.layer2.bias"]))
        self.assertTrue(
            torch.equal(quantizer2.loaded_params["input_scale"], params_to_load["block.layer2.input_scale"])
        )
        self.assertTrue(
            torch.equal(quantizer2.loaded_params["input_offset"], params_to_load["block.layer2.input_offset"])
        )
        self.assertEqual(quantizer2.device, device)


if __name__ == '__main__':
    unittest.main()
