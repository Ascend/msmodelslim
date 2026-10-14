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
# pylint: disable=consider-using-from-import

import unittest
import torch
import torch.nn as nn

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.manager import TimestepManager
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.quantizer import (
    TimestepQuantMixin,
    TimestepAwareTensorQuantizer,
    LinearQuantizerTimestep,
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config import QuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import TimestepQuantConfig  # noqa: F401


class MockConfig:
    """Mock config for testing TimestepQuantMixin."""

    def __init__(self):
        self.use_timestep_quant = True
        self.max_dynamic_step = 5


class MockTensorQuantizer(nn.Module):
    """Mock TensorQuantizer for testing TimestepQuantMixin."""

    def __init__(self):
        super().__init__()
        self.cfg = MockConfig()
        self.input_scale = torch.ones(1)
        self.input_offset = torch.zeros(1)
        self.is_dynamic = True

    def get_scale_offset(self):
        return self.input_scale, self.input_offset


class MockQuantWeight:
    """Mock implementation of weight quantizer for testing."""

    def __init__(self):
        self.int_weight_tensor = None
        self.weight_offset = None
        self.weight_scale = None
        self.int_infer = False
        self.int_weight_flag = True
        self.has_init_quant_para = False
        self.is_calib = True

    def __call__(self, w):
        return w  # Just return the input weight


class MockQuantInput:
    """Mock implementation of input quantizer for testing."""

    def __init__(self):
        self.is_calib = True
        self.int_infer = False
        self.updated_dict = None

    def update_timestep_scale_offset_from_dict(self, state_dict):
        self.updated_dict = state_dict

    def __call__(self, x):
        return x  # Just return the input as is


class TestTimestepQuantMixin(unittest.TestCase):
    """Test the TimestepQuantMixin functionality."""

    def setUp(self):
        # Create a class that uses the mixin
        class TestModule(TimestepQuantMixin, MockTensorQuantizer):
            def __init__(self):
                super().__init__()

        self.module = TestModule()
        # Reset timestep before each test
        TimestepManager._timestep_var.set(None)

    def test_update_timestep_scale_offset(self):
        """Test updating timestep scale and offset values."""
        # Set timestep
        TimestepManager.set_timestep_idx(3)

        # Update scale/offset for this timestep
        self.module.update_timestep_scale_offset(device='cpu')

        # Check that values were stored
        self.assertIn(3, self.module._timestep_scales)
        self.assertIn(3, self.module._timestep_offsets)
        self.assertTrue(torch.equal(self.module._timestep_scales[3], torch.ones(1)))
        self.assertTrue(torch.equal(self.module._timestep_offsets[3], torch.zeros(1)))

        # Change scale/offset and update again for a different timestep
        self.module.input_scale.data = torch.ones(1) * 2
        self.module.input_offset.data = torch.ones(1) * 0.5
        TimestepManager.set_timestep_idx(4)
        self.module.update_timestep_scale_offset(device='cpu')

        # Check that both values are stored correctly
        self.assertIn(4, self.module._timestep_scales)
        self.assertIn(4, self.module._timestep_offsets)
        self.assertTrue(torch.equal(self.module._timestep_scales[4], torch.ones(1) * 2))
        self.assertTrue(torch.equal(self.module._timestep_offsets[4], torch.ones(1) * 0.5))

    def test_get_timestep_scale_offset_dict(self):
        """Test getting all timestep scale/offset values."""
        # Set up scales for different timesteps
        self.module._timestep_scales[1] = torch.ones(1) * 1
        self.module._timestep_offsets[1] = torch.ones(1) * 0.1
        self.module._timestep_scales[2] = torch.ones(1) * 2
        self.module._timestep_offsets[2] = torch.ones(1) * 0.2

        # Get values as tensors
        scale_tensor, offset_tensor = self.module.get_timestep_scale_offset_dict()
        self.assertEqual(scale_tensor.shape, (2, 1))
        self.assertEqual(offset_tensor.shape, (2, 1))
        self.assertTrue(torch.equal(scale_tensor, torch.tensor([[1.0], [2.0]])))
        self.assertTrue(torch.equal(offset_tensor, torch.tensor([[0.1], [0.2]])))

    def test_update_timestep_scale_offset_from_dict(self):
        """Test updating timestep scale and offset from dictionary."""
        # Create input data
        input_scale = torch.tensor([[1.0], [2.0], [3.0]])
        input_offset = torch.tensor([[0.1], [0.2], [0.3]])
        state_dict = {'input_scale': input_scale, 'input_offset': input_offset}

        # Update from dict
        self.module.update_timestep_scale_offset_from_dict(state_dict)

        # Check that values were stored correctly
        for i in range(3):
            self.assertIn(i, self.module._timestep_scales)
            self.assertIn(i, self.module._timestep_offsets)
            self.assertTrue(torch.equal(self.module._timestep_scales[i], torch.tensor([i + 1.0])))
            self.assertTrue(torch.equal(self.module._timestep_offsets[i], torch.tensor([0.1 * (i + 1)])))

    def test_get_current_timestep(self):
        """Test getting current timestep."""
        # Set timestep and check
        TimestepManager.set_timestep_idx(3)
        self.assertEqual(self.module.get_current_timestep(), 3)

        # Change timestep and check again
        TimestepManager.set_timestep_idx(7)
        self.assertEqual(self.module.get_current_timestep(), 7)

    def test_apply_timestep_quant_settings(self):
        """Test applying timestep quantization settings."""
        # Setup timestep values
        TimestepManager.set_timestep_idx(3)
        self.module._timestep_scales[5] = torch.ones(1) * 5
        self.module._timestep_offsets[5] = torch.ones(1) * 0.5

        # When timestep < max_dynamic_step, should remain dynamic
        self.module.apply_timestep_quant_settings()
        self.assertTrue(self.module.is_dynamic)

        # When timestep >= max_dynamic_step, should use fixed value and switch to static
        TimestepManager.set_timestep_idx(7)
        self.module.apply_timestep_quant_settings()
        self.assertFalse(self.module.is_dynamic)
        self.assertTrue(torch.equal(self.module.input_scale, torch.ones(1) * 5))
        self.assertTrue(torch.equal(self.module.input_offset, torch.ones(1) * 0.5))

    def test_update_config(self):
        """Test updating config max_dynamic_step."""
        self.assertEqual(self.module.cfg.max_dynamic_step, 5)
        self.module.update_config(10)
        self.assertEqual(self.module.cfg.max_dynamic_step, 10)


class TestTimestepAwareTensorQuantizer(unittest.TestCase):
    """Test the TimestepAwareTensorQuantizer class."""

    def setUp(self):
        # Create a config for timestep quantization
        base_cfg = QuantConfig(w_bit=8, a_bit=8, w_sym=True)
        self.cfg = base_cfg.timestep_quant(max_dynamic_step=5)

    def test_initialization(self):
        """Test initialization of TimestepAwareTensorQuantizer."""
        # Create a quantizer with required parameters
        quantizer = TimestepAwareTensorQuantizer(
            bit=8, is_signed=True, is_enable=True, is_input=True, cfg=self.cfg, is_dynamic=True
        )

        # Check that it was initialized correctly
        self.assertEqual(quantizer.cfg, self.cfg)
        self.assertEqual(quantizer.bit, 8)
        self.assertTrue(quantizer.is_signed)
        self.assertTrue(quantizer.is_enable)
        self.assertTrue(quantizer.is_input)
        self.assertTrue(quantizer.is_dynamic)
        self.assertTrue(hasattr(quantizer, '_timestep_scales'))
        self.assertTrue(hasattr(quantizer, '_timestep_offsets'))


class TestLinearQuantizerTimestep(unittest.TestCase):
    """Test the LinearQuantizerTimestep class."""

    def setUp(self):
        # Create a config for timestep quantization
        base_cfg = QuantConfig(w_bit=8, a_bit=8, w_sym=True)
        base_cfg.is_dynamic = True
        self.cfg = base_cfg.timestep_quant(max_dynamic_step=5)

    def test_initialization(self):
        """Test initialization of LinearQuantizerTimestep."""
        # Create a quantizer
        quantizer = LinearQuantizerTimestep(cfg=self.cfg)

        # Check that it was initialized correctly
        self.assertEqual(quantizer.cfg, self.cfg)
        self.assertTrue(hasattr(quantizer, 'quant_input'))
        self.assertTrue(isinstance(quantizer.quant_input, TimestepAwareTensorQuantizer))
        self.assertEqual(quantizer.quant_input.bit, 8)
        self.assertTrue(quantizer.quant_input.is_signed)

    def test_forward_reshape(self):
        """Test the reshape_x_to_blc method."""

        # Create a simple function for testing
        def test_func(x):
            return x * 2

        # Apply the reshape decorator
        wrapped_func = LinearQuantizerTimestep.reshape_x_to_blc(test_func)

        # Test with 3D input (already in BLC format)
        x_3d = torch.ones(2, 3, 4)  # [B, L, C]
        result_3d = wrapped_func(x_3d)
        self.assertTrue(torch.equal(result_3d, x_3d * 2))

        # Test with 2D input (needs reshaping)
        x_2d = torch.ones(6, 4)  # [B*L, C]
        result_2d = wrapped_func(x_2d)
        self.assertTrue(torch.equal(result_2d, x_2d * 2))

        # Test with 4D input (needs reshaping)
        x_4d = torch.ones(2, 3, 1, 4)  # [B, L, X, C]
        result_4d = wrapped_func(x_4d)
        self.assertEqual(result_4d.shape, (2, 3, 1, 4))
        self.assertTrue(torch.equal(result_4d, x_4d * 2))


if __name__ == "__main__":
    unittest.main()
