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
from torch import nn

from msmodelslim.ir.api import calculate_qparam
from msmodelslim.ir.auto import AutoFakeQuantActivation
from msmodelslim.ir.const import mxfp8_per_channel_sym
from msmodelslim.ir.mxfp8_activation_static import MXFP8FakeQuantActivationPerChannel
from msmodelslim.ir.qal import QParam, QDType, QScope


class TestMXFP8FakeQuantActivationPerChannel(unittest.TestCase):
    """对应 msmodelslim/ir/mxfp8_activation_static.py"""

    def _make_qparam(self, channels: int = 16) -> QParam:
        amax = torch.ones(channels, dtype=torch.float32)
        return calculate_qparam(
            min_val=-amax,
            max_val=amax,
            q_dtype=QDType.MXFP8,
            q_scope=QScope.PER_CHANNEL,
            symmetric=True,
        )

    def test_init_stores_scheme_and_scale_when_valid_qparam(self):
        q_param = self._make_qparam(8)
        module = MXFP8FakeQuantActivationPerChannel(q_param)
        self.assertEqual(module.x_q_scheme, mxfp8_per_channel_sym)
        self.assertIsInstance(module.input_scale, nn.Parameter)
        self.assertFalse(module.input_scale.requires_grad)
        self.assertEqual(tuple(module.input_scale.shape), (8,))

    def test_init_raises_ValueError_when_scale_missing(self):
        bad = QParam(scheme=mxfp8_per_channel_sym, ext={})
        with self.assertRaises(ValueError):
            MXFP8FakeQuantActivationPerChannel(bad)

    def test_forward_returns_same_shape_when_4d_input(self):
        module = MXFP8FakeQuantActivationPerChannel(self._make_qparam(8))
        # B=2, H=4, S=8, D=2 → 通道 = H*D = 8
        x = torch.randn(2, 4, 8, 2, dtype=torch.float32)
        y = module(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())

    def test_AutoFakeQuantActivation_create_returns_instance_when_mxfp8_per_channel_scheme(self):
        module = AutoFakeQuantActivation.create(self._make_qparam(16))
        self.assertIsInstance(module, MXFP8FakeQuantActivationPerChannel)


if __name__ == "__main__":
    unittest.main()
