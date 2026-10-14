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

Unit tests for non_fusion_smooth_quant_ir (NonFusionSmoothQuantWrapper, NonFusionSmoothQuantHookIR).
Covers code added in commit f2075e1 (non-fusion SmoothQuant IR).
"""

import unittest
from unittest.mock import MagicMock

import torch
from torch import nn

from msmodelslim.ir.non_fusion_smooth_quant_ir import (
    NonFusionSmoothQuantWrapper,
    NonFusionSmoothQuantHookIR,
)


class TestNonFusionSmoothQuantWrapper(unittest.TestCase):
    """Tests for NonFusionSmoothQuantWrapper."""

    def test_init_stores_inverse_scales(self):
        """Wrapper stores 1/scales as parameter."""
        scales = torch.tensor([2.0, 4.0])
        linear = nn.Linear(2, 3)
        wrapper = NonFusionSmoothQuantWrapper(scales, linear)
        self.assertIs(wrapper.wrapped_module, linear)
        expected = torch.tensor([0.5, 0.25])
        torch.testing.assert_close(wrapper.scales, expected)
        self.assertFalse(wrapper.scales.requires_grad)

    def test_is_atomic_returns_true(self):
        """is_atomic() returns True for save/export as single unit."""
        self.assertTrue(NonFusionSmoothQuantWrapper.is_atomic())

    def test_forward_tensor_input(self):
        """Forward with tensor input: output = linear(x * (1/scales))."""
        scales = torch.tensor([2.0, 4.0])
        linear = nn.Linear(2, 3)
        torch.nn.init.ones_(linear.weight)
        torch.nn.init.zeros_(linear.bias)
        wrapper = NonFusionSmoothQuantWrapper(scales, linear)
        x = torch.tensor([[1.0, 2.0]])  # (1, 2)
        out = wrapper(x)
        # x * (1/scales) = [0.5, 0.5]; linear gives (1, 3)
        self.assertEqual(out.shape, (1, 3))
        expected_in = x * wrapper.scales  # [0.5, 0.5]
        expected_out = linear(expected_in)
        torch.testing.assert_close(out, expected_out)

    def test_forward_tuple_input_takes_first_element(self):
        """Forward with tuple input uses first element."""
        scales = torch.tensor([1.0, 1.0])
        linear = nn.Linear(2, 2)
        wrapper = NonFusionSmoothQuantWrapper(scales, linear)
        x = torch.randn(2, 2)
        out = wrapper((x,))
        self.assertEqual(out.shape, (2, 2))
        torch.testing.assert_close(out, linear(x))

    def test_forward_device_dtype_align(self):
        """Scales are moved to input device/dtype in forward."""
        scales = torch.tensor([2.0, 2.0])
        linear = nn.Linear(2, 2)
        wrapper = NonFusionSmoothQuantWrapper(scales, linear)
        x = torch.randn(1, 2, dtype=torch.float32)
        out = wrapper(x)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.device, x.device)


class TestNonFusionSmoothQuantHookIR(unittest.TestCase):
    """Tests for NonFusionSmoothQuantHookIR."""

    def test_init_stores_scales(self):
        """HookIR stores scales tensor."""
        scales = torch.tensor([1.0, 2.0])
        hook_ir = NonFusionSmoothQuantHookIR(scales)
        self.assertIs(hook_ir.scales, scales)

    def test_call_returns_first_arg_no_op(self):
        """__call__ is no-op: returns first argument (for pre_hook)."""
        scales = torch.tensor([1.0])
        hook_ir = NonFusionSmoothQuantHookIR(scales)
        module = nn.Linear(1, 1)
        x = torch.tensor([[1.0]])
        result = hook_ir(module, (x,))
        self.assertEqual(result, x)

    def test_wrapper_module_returns_wrapper_and_removes_hook(self):
        """wrapper_module(linear) returns NonFusionSmoothQuantWrapper and removes hook."""
        scales = torch.tensor([2.0])
        hook_ir = NonFusionSmoothQuantHookIR(scales)
        hook_ir.hook_handle = MagicMock()
        linear = nn.Linear(1, 1)
        wrapper = hook_ir.wrapper_module(linear)
        self.assertIsInstance(wrapper, NonFusionSmoothQuantWrapper)
        self.assertIs(wrapper.wrapped_module, linear)
        torch.testing.assert_close(wrapper.scales, torch.tensor([0.5]))
        hook_ir.hook_handle.remove.assert_called_once()

    def test_set_hook_handle(self):
        """set_hook_handle stores handle."""
        hook_ir = NonFusionSmoothQuantHookIR(torch.tensor([1.0]))
        handle = MagicMock()
        hook_ir.set_hook_handle(handle)
        self.assertIs(hook_ir.hook_handle, handle)


if __name__ == "__main__":
    unittest.main()
