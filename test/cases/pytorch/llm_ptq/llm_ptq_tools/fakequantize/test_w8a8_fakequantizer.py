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
# pylint: disable=consider-using-from-import,ungrouped-imports

from unittest.mock import patch
import pytest
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.calibrator.calibrator_classes.fake_quantize.w8a8_fakequantizer import (
    FakeLinearQuantizerOfW8A8,
)


class TestFakeLinearQuantizerOfW8A8:
    @patch('os.environ', {'sourceBranch': 'test_branch'})
    def __init__(self):
        # 这相当于 unittest 的 setUp 方法
        self.quant_params = {
            'deq_scale': torch.tensor([1], dtype=torch.int64),
            'input_offset': torch.tensor([0], dtype=torch.float32),
            'input_scale': torch.tensor([1], dtype=torch.float32),
            'quant_weight': torch.randn(10, 5),
        }
        self.linear = nn.Linear(5, 10)
        self.quantizer = FakeLinearQuantizerOfW8A8()
        self.quantizer.set_param(linear=self.linear, **self.quant_params)

    def test_int64tofp32(self):
        int64_tensor = torch.tensor([2**32 - 1], dtype=torch.int64)
        expected_fp32_tensor = np.frombuffer(int64_tensor.numpy().astype(np.int32).tobytes(), dtype=np.float32)
        actual_fp32_tensor = self.quantizer.int64tofp32()
        np.testing.assert_array_equal(expected_fp32_tensor, actual_fp32_tensor.numpy())

    def test_deqscale_process(self):
        expected_scale = self.quantizer.deq_scale.to(torch.float64) / self.quantizer.input_scale.to(torch.float64)
        actual_scale = self.quantizer.deqscale_process()
        assert torch.allclose(expected_scale, actual_scale)

    def test_set_param(self):
        # 测试参数是否正确设置
        assert self.quantizer.device == self.linear.weight.device
        assert self.quantizer.dtype == self.linear.weight.dtype
        assert torch.allclose(self.quantizer.deq_scale, self.quant_params['deq_scale'].to(torch.float32))
        assert torch.allclose(self.quantizer.input_offset, self.quant_params['input_offset'])
        assert torch.allclose(self.quantizer.input_scale, self.quant_params['input_scale'])
        assert torch.allclose(self.quantizer.weight, self.quant_params['quant_weight'] * self.quantizer.scale)

    def test_forward(self):
        x = torch.randn(3, 5)
        expected_output = F.linear(
            x / self.quantizer.input_scale + self.quantizer.input_offset,
            self.quantizer.weight,
            self.quantizer.quant_bias,
        )
        actual_output = self.quantizer.forward(x)
        assert torch.allclose(expected_output, actual_output)

    def test_forward_input_scale_zero(self):
        with pytest.raises(ValueError):
            self.quantizer.input_scale = torch.tensor([0], dtype=torch.float32)
            self.quantizer.forward(torch.randn(3, 5))
