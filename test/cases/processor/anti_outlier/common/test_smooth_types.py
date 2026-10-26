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

Unit tests for anti_outlier.common.smooth_types dataclasses.
"""

import unittest

import torch

from msmodelslim.processor.anti_outlier.common.smooth_types import (
    FlexAWQSSZConfig,
    FlexAWQSSZContext,
    FlexSmoothQuantConfig,
    FlexSmoothQuantContext,
    IterSmoothConfig,
    IterSmoothContext,
    OASQConfig,
    OASQContext,
    SmoothQuantConfig,
    SmoothQuantContext,
)


class TestConfigDefaults(unittest.TestCase):
    """Config dataclasses expose documented default values."""

    def test_smooth_quant_config_defaults(self):
        config = SmoothQuantConfig()
        self.assertEqual(config.version, 1)
        self.assertEqual(config.alpha, 0.5)
        self.assertFalse(config.shift)

    def test_iter_smooth_config_defaults(self):
        config = IterSmoothConfig()
        self.assertEqual(config.version, 1)
        self.assertEqual(config.alpha, 0.9)
        self.assertFalse(config.shift)
        self.assertEqual(config.scale_min, 1e-5)

    def test_oasq_config_defaults(self):
        config = OASQConfig()
        self.assertIsNone(config.max_iters)
        self.assertFalse(config.shift)
        self.assertEqual(config.version, 1)

    def test_flex_smooth_quant_config_defaults(self):
        config = FlexSmoothQuantConfig()
        self.assertEqual(config.version, 1)
        self.assertIsNone(config.alpha)
        self.assertIsNone(config.beta)
        self.assertIsNone(config.extra_config)

    def test_flex_awq_ssz_config_defaults(self):
        config = FlexAWQSSZConfig()
        self.assertEqual(config.version, 1)
        self.assertIsNone(config.alpha)
        self.assertIsNone(config.beta)
        self.assertIsNone(config.qconfig)


class TestContextFields(unittest.TestCase):
    """Context dataclasses carry the runtime tensors passed by callers."""

    def test_smooth_quant_context_holds_tensors(self):
        scale = torch.tensor([1.0, 2.0])
        shift = torch.tensor([0.1, 0.2])
        context = SmoothQuantContext(version=1, a_smooth_scale=scale, shift=shift)
        self.assertIs(context.a_smooth_scale, scale)
        self.assertIs(context.shift, shift)

    def test_iter_smooth_context_holds_tensors(self):
        scale = torch.ones(4)
        shift = torch.zeros(4)
        context = IterSmoothContext(version=2, a_smooth_scale=scale, shift=shift)
        self.assertEqual(context.version, 2)
        self.assertTrue(torch.equal(context.a_smooth_scale, scale))

    def test_oasq_context_shift_defaults_to_none(self):
        context = OASQContext(version=1, a_smooth_scale=torch.ones(3))
        self.assertIsNone(context.shift)

    def test_flex_smooth_quant_context_tensors_default_none(self):
        context = FlexSmoothQuantContext(version=1, a_smooth_scale=torch.ones(3))
        self.assertIsNone(context.tensors)

    def test_flex_smooth_quant_context_keeps_tensor_list(self):
        tensors = [torch.ones(2, 3), torch.zeros(2, 3)]
        context = FlexSmoothQuantContext(version=1, a_smooth_scale=torch.ones(3), tensors=tensors)
        self.assertEqual(len(context.tensors), 2)

    def test_flex_awq_ssz_context_requires_tensors(self):
        tensors = [torch.ones(2, 3)]
        context = FlexAWQSSZContext(version=1, tensors=tensors)
        self.assertIs(context.tensors, tensors)


if __name__ == '__main__':
    unittest.main()
