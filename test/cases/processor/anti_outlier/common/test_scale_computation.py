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

Unit tests for anti_outlier.common.scale_computation public helpers.
"""

# pylint: disable=abstract-class-instantiated

import unittest

import torch

from msmodelslim.processor.anti_outlier.common.scale_computation import (
    BaseScaleCalculator,
    FlexAWQSSZScaleCalculator,
    FlexSmoothScaleCalculator,
    compute_multi_weight_scale,
    compute_weight_scale,
    prepare_mqga_parameters,
    validate_and_process_tensors,
)
from msmodelslim.processor.anti_outlier.common.smooth_types import FlexAWQSSZContext


class TestComputeWeightScale(unittest.TestCase):
    """compute_weight_scale reduces a weight to per-input-channel abs max."""

    def test_compute_weight_scale_takes_abs_max_per_input_channel(self):
        weight = torch.tensor([[1.0, -3.0], [2.0, 5.0]])
        scale = compute_weight_scale(weight, torch.float32)
        torch.testing.assert_close(scale, torch.tensor([2.0, 5.0]))

    def test_compute_weight_scale_clamps_zero_to_min(self):
        weight = torch.zeros(3, 4)
        scale = compute_weight_scale(weight, torch.float32)
        self.assertEqual(scale.shape, (4,))
        torch.testing.assert_close(scale, torch.full((4,), 1e-5))

    def test_compute_weight_scale_respects_output_dtype(self):
        weight = torch.tensor([[1.0, 2.0]])
        scale = compute_weight_scale(weight, torch.float16)
        self.assertEqual(scale.dtype, torch.float16)


class TestComputeMultiWeightScale(unittest.TestCase):
    """compute_multi_weight_scale combines per-weight stats with an elementwise max."""

    def test_compute_multi_weight_scale_elementwise_max(self):
        w1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        w2 = torch.tensor([[5.0, 1.0], [0.0, 6.0]])
        scale = compute_multi_weight_scale([w1, w2], torch.float32)
        torch.testing.assert_close(scale, torch.tensor([5.0, 6.0]))

    def test_compute_multi_weight_scale_clamps_min(self):
        scale = compute_multi_weight_scale([torch.zeros(2, 2)], torch.float32)
        torch.testing.assert_close(scale, torch.full((2,), 1e-5))


class TestValidateAndProcessTensors(unittest.TestCase):
    """validate_and_process_tensors concatenates and flattens context tensors."""

    def test_cat_and_flatten(self):
        context = FlexAWQSSZContext(version=1, tensors=[torch.ones(2, 3), torch.zeros(1, 3)])
        result = validate_and_process_tensors(context, torch.device("cpu"), torch.float32)
        self.assertEqual(tuple(result.shape), (3, 3))
        self.assertTrue(torch.all(result[:2] == 1.0))
        self.assertTrue(torch.all(result[2] == 0.0))

    def test_skips_none_and_non_tensor_entries(self):
        context = FlexAWQSSZContext(version=1, tensors=[torch.ones(2, 3), None, "ignored", torch.zeros(1, 3)])
        result = validate_and_process_tensors(context, torch.device("cpu"), torch.float32)
        self.assertEqual(tuple(result.shape), (3, 3))


class TestScaleCalculators(unittest.TestCase):
    """Flex scale calculators apply a^alpha / w^beta with a lower clamp."""

    def test_flex_smooth_scale_calculator_matches_formula(self):
        calculator = FlexSmoothScaleCalculator(alpha=0.5, beta=0.5)
        scales = calculator.compute_smooth_scale(torch.tensor([4.0]), torch.tensor([1.0]))
        torch.testing.assert_close(scales, torch.tensor([2.0]))

    def test_flex_awq_ssz_scale_calculator_beta_zero(self):
        calculator = FlexAWQSSZScaleCalculator(alpha=1.0, beta=0.0)
        scales = calculator.compute_smooth_scale(torch.tensor([4.0]), torch.tensor([9.0]))
        torch.testing.assert_close(scales, torch.tensor([4.0]))

    def test_flex_smooth_scale_calculator_clamps_min(self):
        calculator = FlexSmoothScaleCalculator(alpha=1.0, beta=0.0)
        scales = calculator.compute_smooth_scale(torch.tensor([0.0]), torch.tensor([1.0]))
        torch.testing.assert_close(scales, torch.tensor([1e-5]))

    def test_base_scale_calculator_is_abstract(self):
        with self.assertRaises(TypeError):
            BaseScaleCalculator()


class TestPrepareMqgaParameters(unittest.TestCase):
    """prepare_mqga_parameters returns the q/kv head ratio and zero pad size."""

    def test_ratio_and_pad(self):
        self.assertEqual(prepare_mqga_parameters(8, 4), (2, 0))

    def test_mha_ratio_is_one(self):
        self.assertEqual(prepare_mqga_parameters(8, 8), (1, 0))


if __name__ == "__main__":
    unittest.main()
