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

import unittest

import torch

from msmodelslim.ir.api import calculate_qparam, fake_quantize, quantize, dequantize
from msmodelslim.ir.const import mxfp8_per_channel_sym
from msmodelslim.ir.qal import QStorage, QDType, QScope


class TestMxfp8PerChannelCalculateQparam(unittest.TestCase):
    """覆盖 mx_quantization.calculate_qparam 的 MXFP8 PER_CHANNEL 注册。"""

    def test_calculate_qparam_returns_scale_shape_when_per_channel_mxfp8(self):
        max_val = torch.tensor([1.0, 10.0, 0.5], dtype=torch.float32)
        q_param = calculate_qparam(
            min_val=-max_val,
            max_val=max_val,
            q_dtype=QDType.MXFP8,
            q_scope=QScope.PER_CHANNEL,
            symmetric=True,
        )
        self.assertEqual(q_param.scheme, mxfp8_per_channel_sym)
        self.assertEqual(tuple(q_param.ext["scale"].shape), (3,))

    def test_fake_quantize_returns_same_shape_when_2d_per_channel_input(self):
        x = torch.randn(64, 16, dtype=torch.float32)
        amax = x.abs().amax(dim=0)
        q_param = calculate_qparam(
            min_val=-amax,
            max_val=amax,
            q_dtype=QDType.MXFP8,
            q_scope=QScope.PER_CHANNEL,
            symmetric=True,
        )
        out = fake_quantize(QStorage(QDType.FLOAT, x), q_param).value
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_quantize_dequantize_roundtrip_finite_when_per_channel_mxfp8(self):
        x = torch.randn(32, 8, dtype=torch.float32)
        amax = x.abs().amax(dim=0)
        q_param = calculate_qparam(
            min_val=-amax,
            max_val=amax,
            q_dtype=QDType.MXFP8,
            q_scope=QScope.PER_CHANNEL,
            symmetric=True,
        )
        q = quantize(QStorage(QDType.FLOAT, x), q_param)
        dq = dequantize(q, q_param).value
        self.assertEqual(dq.shape, x.shape)
        self.assertTrue(torch.isfinite(dq).all())


if __name__ == "__main__":
    unittest.main()
