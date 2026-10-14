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

TLQ MXFP shared_exp：mx_scale = base | c7。
"""

import unittest

import torch

from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.processor.trainable_linear_quant.core.kernels.mxfp import (
    _compute_shared_exp,
    _mx_format,
    _parse_mx_scale,
    build_mx_context,
    create_mxfp_kernel,
)


class TestTLQMxScale(unittest.TestCase):
    def test_default_is_base(self):
        for dtype in (QDType.MXFP4, QDType.MXFP8):
            cfg = QConfig(dtype=dtype, scope=QScope.PER_BLOCK, symmetric=True, method="minmax")
            self.assertEqual(_parse_mx_scale(cfg), "base")

    def test_parse_c7(self):
        cfg = QConfig(
            dtype=QDType.MXFP4,
            scope=QScope.PER_BLOCK,
            symmetric=True,
            method="minmax",
            ext={"mx_scale": "c7"},
        )
        self.assertEqual(_parse_mx_scale(cfg), "c7")

    def test_c7_rejected_for_mxfp8(self):
        cfg = QConfig(
            dtype=QDType.MXFP8,
            scope=QScope.PER_BLOCK,
            symmetric=True,
            method="minmax",
            ext={"mx_scale": "c7"},
        )
        with self.assertRaises(ValueError):
            _parse_mx_scale(cfg)

    def test_c7_shared_exp(self):
        cfg = QConfig(
            dtype=QDType.MXFP4,
            scope=QScope.PER_BLOCK,
            symmetric=True,
            method="minmax",
            ext={"mx_scale": "c7"},
        )
        mx_fmt = _mx_format(cfg)
        # M=8 → ceil(log2(8/7.25)) = 1
        ctx = build_mx_context(cfg, torch.zeros(1, 32), {})
        ctx.block_max = torch.tensor([[[8.0]]])
        _compute_shared_exp(ctx, cfg, mx_fmt, mx_scale="c7")
        self.assertTrue(torch.equal(ctx.q_param.ext["scale"], torch.tensor([[[1.0]]])))

    def test_base_mxfp4(self):
        cfg = QConfig(dtype=QDType.MXFP4, scope=QScope.PER_BLOCK, symmetric=True, method="minmax")
        mx_fmt = _mx_format(cfg)
        # M=8 → floor(log2(8/0.875)) - 2 = 1
        ctx = build_mx_context(cfg, torch.zeros(1, 32), {})
        ctx.block_max = torch.tensor([[[8.0]]])
        _compute_shared_exp(ctx, cfg, mx_fmt, mx_scale="base")
        self.assertTrue(torch.equal(ctx.q_param.ext["scale"], torch.tensor([[[1.0]]])))

    def test_base_mxfp8(self):
        cfg = QConfig(dtype=QDType.MXFP8, scope=QScope.PER_BLOCK, symmetric=True, method="minmax")
        mx_fmt = _mx_format(cfg)
        # M=8 → floor(log2(8)) - emax = 3 - 8 = -5
        ctx = build_mx_context(cfg, torch.zeros(1, 32), {})
        ctx.block_max = torch.tensor([[[8.0]]])
        _compute_shared_exp(ctx, cfg, mx_fmt, mx_scale="base")
        expected = 3.0 - float(mx_fmt.emax)
        self.assertTrue(torch.equal(ctx.q_param.ext["scale"], torch.tensor([[[expected]]])))

    def test_create_mxfp4_kernel_smoke(self):
        cfg = QConfig(dtype=QDType.MXFP4, scope=QScope.PER_BLOCK, symmetric=True, method="minmax")
        kernel = create_mxfp_kernel(cfg)
        x = torch.randn(4, 64)
        result = kernel.fake_quantize(x)
        out = getattr(result, "tensor")
        self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
