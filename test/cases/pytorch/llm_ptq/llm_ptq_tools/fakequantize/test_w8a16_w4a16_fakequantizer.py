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

from unittest.mock import patch
import pytest
import torch
import torch.nn as nn

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.calibrator.calibrator_classes.fake_quantize.w8a16_w4a16_fakequantizer import (
    FakeLinearQuantizerOfW8A16OrW4A16,
)


class TestFakeLinearQuantizerOfW8A16OrW4A16:
    @patch('os.environ', {'sourceBranch': 'test_branch'})
    def __init__(self):
        self.quantizer = FakeLinearQuantizerOfW8A16OrW4A16()

    def test_set_param_with_valid_input(self):
        linear = nn.Linear(10, 5)
        quant_weight = torch.randint(0, 255, (5, 10))
        weight_offset = torch.randint(0, 255, (5, 1))
        weight_scale = torch.randint(0, 255, (5, 1))
        self.quantizer.set_param(
            linear, quant_weight=quant_weight, weight_offset=weight_offset, weight_scale=weight_scale
        )
        assert self.quantizer.deq_weight is not None
        assert self.quantizer.quant_bias is not None

    def test_set_param_with_invalid_input(self):
        linear = nn.Linear(10, 5)
        quant_weight = torch.randint(0, 255, (5, 10))
        weight_offset = torch.randint(0, 255, (5, 2))  # Invalid shape
        weight_scale = torch.randint(0, 255, (5, 1))
        with pytest.raises(ValueError):
            self.quantizer.set_param(
                linear, quant_weight=quant_weight, weight_offset=weight_offset, weight_scale=weight_scale
            )

    def test_forward_with_valid_input(self):
        linear = nn.Linear(10, 5)
        quant_weight = torch.randint(0, 255, (5, 10))
        weight_offset = torch.randint(0, 255, (5, 1))
        weight_scale = torch.randint(0, 255, (5, 1))
        self.quantizer.set_param(
            linear, quant_weight=quant_weight, weight_offset=weight_offset, weight_scale=weight_scale
        )
        x = torch.randn(3, 10)
        output = self.quantizer(x)
        assert output.shape == (3, 5)

    def test_forward_with_invalid_input(self):
        linear = nn.Linear(10, 5)
        quant_weight = torch.randint(0, 255, (5, 10))
        weight_offset = torch.randint(0, 255, (5, 1))
        weight_scale = torch.randint(0, 255, (5, 1))
        self.quantizer.set_param(
            linear, quant_weight=quant_weight, weight_offset=weight_offset, weight_scale=weight_scale
        )
        x = torch.randn(3, 15)  # Invalid input shape
        with pytest.raises(RuntimeError):
            output = self.quantizer(x)  # noqa: F841
