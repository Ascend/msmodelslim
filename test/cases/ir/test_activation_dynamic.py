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

from msmodelslim.ir.activation_dynamic import FakeQuantActivationPerToken, FakeQuantActivationPerBlock
from msmodelslim.ir.qal import QParam, QScope, QDType
from msmodelslim.ir.const import fp8_e4m3_per_token_sym, mxfp4_per_block_sym, mxfp8_per_block_sym


class TestFakeQuantActivationPerToken(unittest.TestCase):
    """测试 FakeQuantActivationPerToken 类"""

    def setUp(self):
        """设置测试环境"""
        # 创建QParam，使用fp8_e4m3_per_token_sym scheme
        self.q_param = QParam(scheme=fp8_e4m3_per_token_sym)

    def test_init(self):
        """测试初始化"""
        ir_module = FakeQuantActivationPerToken(self.q_param)
        self.assertEqual(ir_module.x_q_scheme, fp8_e4m3_per_token_sym)

    def test_forward_4d_shape(self):
        """测试4D输入形状 (B, H, S, D)"""
        ir_module = FakeQuantActivationPerToken(self.q_param)
        x = torch.randn(2, 4, 10, 16)  # (B, H, S, D)

        with torch.no_grad():
            output = ir_module(x)

        # 验证输出形状与输入相同
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(output.dtype, x.dtype)

    def test_forward_preserves_dtype(self):
        """测试保持数据类型"""
        ir_module = FakeQuantActivationPerToken(self.q_param)

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x = torch.randn(2, 4, 10, 16, dtype=dtype)
            with torch.no_grad():
                output = ir_module(x)
            self.assertEqual(output.dtype, dtype)

    def test_forward_with_negative_values(self):
        """测试负值处理"""
        ir_module = FakeQuantActivationPerToken(self.q_param)
        x = torch.randn(2, 4, 10, 16) * 2 - 1  # 范围 [-1, 1]

        with torch.no_grad():
            output = ir_module(x)

        self.assertEqual(output.shape, x.shape)

    def test_forward_quantization_effect(self):
        """测试量化效果（输出应该与输入不同）"""
        ir_module = FakeQuantActivationPerToken(self.q_param)
        x = torch.randn(2, 4, 10, 16)

        with torch.no_grad():
            output = ir_module(x)

        # 由于是量化，输出应该与输入不完全相同（除非输入恰好是量化值）
        # 但形状应该相同
        self.assertEqual(output.shape, x.shape)
        # 量化后的值应该在合理范围内
        self.assertTrue(torch.isfinite(output).all())

    def test_forward_gradient_flow(self):
        """测试梯度流（虽然通常不需要梯度，但确保不会出错）"""
        ir_module = FakeQuantActivationPerToken(self.q_param)
        x = torch.randn(2, 4, 10, 16, requires_grad=True)

        # 即使输入requires_grad，量化操作通常在no_grad下进行
        with torch.no_grad():
            output = ir_module(x)

        self.assertEqual(output.shape, x.shape)
        self.assertFalse(output.requires_grad)  # 量化后通常不需要梯度

    def test_forward_edge_case_single_token(self):
        """测试边界情况：单个token"""
        ir_module = FakeQuantActivationPerToken(self.q_param)
        x = torch.randn(1, 1, 1, 16)  # 最小形状

        with torch.no_grad():
            output = ir_module(x)

        self.assertEqual(output.shape, x.shape)

    def test_forward_edge_case_large_tensor(self):
        """测试边界情况：大张量"""
        ir_module = FakeQuantActivationPerToken(self.q_param)
        x = torch.randn(4, 32, 512, 128)  # 较大的张量

        with torch.no_grad():
            output = ir_module(x)

        self.assertEqual(output.shape, x.shape)

    def test_scheme_property(self):
        """测试scheme属性"""
        ir_module = FakeQuantActivationPerToken(self.q_param)
        self.assertEqual(ir_module.x_q_scheme, fp8_e4m3_per_token_sym)
        self.assertEqual(ir_module.x_q_scheme.scope, QScope.PER_TOKEN)
        self.assertEqual(ir_module.x_q_scheme.dtype, QDType.FP8_E4M3)
        self.assertTrue(ir_module.x_q_scheme.symmetric)


class TestFakeQuantActivationPerBlock(unittest.TestCase):
    """测试 FakeQuantActivationPerBlock（MXFP4 per-block 动态量化）"""

    def setUp(self):
        self.q_param = QParam(scheme=mxfp4_per_block_sym)

    def test_init(self):
        ir_module = FakeQuantActivationPerBlock(self.q_param)
        self.assertEqual(ir_module.x_q_scheme, mxfp4_per_block_sym)

    def test_forward_4d_shape(self):
        ir_module = FakeQuantActivationPerBlock(self.q_param)
        x = torch.randn(2, 4, 10, 128)
        with torch.no_grad():
            output = ir_module(x)
        self.assertEqual(output.shape, x.shape)

    def test_forward_preserves_dtype(self):
        ir_module = FakeQuantActivationPerBlock(self.q_param)
        x = torch.randn(2, 4, 10, 128, dtype=torch.float32)
        with torch.no_grad():
            output = ir_module(x)
        self.assertEqual(output.dtype, torch.float32)

    def test_forward_invalid_block_size(self):
        """非 32 倍数的 head_dim 应正常处理（pad/trim）。"""
        ir_module = FakeQuantActivationPerBlock(self.q_param)
        x = torch.randn(2, 4, 10, 50)
        with torch.no_grad():
            output = ir_module(x)
        self.assertEqual(output.shape, x.shape)

    def test_forward_produces_finite_output(self):
        ir_module = FakeQuantActivationPerBlock(self.q_param)
        x = torch.randn(2, 4, 10, 128)
        with torch.no_grad():
            output = ir_module(x)
        self.assertTrue(torch.isfinite(output).all())

    def test_scheme_property(self):
        ir_module = FakeQuantActivationPerBlock(self.q_param)
        self.assertEqual(ir_module.x_q_scheme.scope, QScope.PER_BLOCK)
        self.assertEqual(ir_module.x_q_scheme.dtype, QDType.MXFP4)
        self.assertTrue(ir_module.x_q_scheme.symmetric)


class TestFakeQuantActivationPerBlockMXFP8(unittest.TestCase):
    """测试 FakeQuantActivationPerBlock（MXFP8 per-block 动态量化）"""

    def setUp(self):
        self.q_param = QParam(scheme=mxfp8_per_block_sym)

    def test_init_returns_correct_scheme_when_mxfp8(self):
        ir_module = FakeQuantActivationPerBlock(self.q_param)
        self.assertEqual(ir_module.x_q_scheme, mxfp8_per_block_sym)

    def test_forward_returns_same_shape_when_4d_input(self):
        ir_module = FakeQuantActivationPerBlock(self.q_param)
        x = torch.randn(2, 4, 10, 128)
        with torch.no_grad():
            output = ir_module(x)
        self.assertEqual(output.shape, x.shape)

    def test_forward_preserves_dtype_when_float32_input(self):
        ir_module = FakeQuantActivationPerBlock(self.q_param)
        x = torch.randn(2, 4, 10, 128, dtype=torch.float32)
        with torch.no_grad():
            output = ir_module(x)
        self.assertEqual(output.dtype, torch.float32)

    def test_forward_returns_same_shape_when_non_block_aligned_dim(self):
        ir_module = FakeQuantActivationPerBlock(self.q_param)
        x = torch.randn(2, 4, 10, 50)
        with torch.no_grad():
            output = ir_module(x)
        self.assertEqual(output.shape, x.shape)

    def test_forward_produces_finite_output_when_normal_input(self):
        ir_module = FakeQuantActivationPerBlock(self.q_param)
        x = torch.randn(2, 4, 10, 128)
        with torch.no_grad():
            output = ir_module(x)
        self.assertTrue(torch.isfinite(output).all())

    def test_scheme_property_returns_mxfp8_when_mxfp8_scheme(self):
        ir_module = FakeQuantActivationPerBlock(self.q_param)
        self.assertEqual(ir_module.x_q_scheme.scope, QScope.PER_BLOCK)
        self.assertEqual(ir_module.x_q_scheme.dtype, QDType.MXFP8)
        self.assertTrue(ir_module.x_q_scheme.symmetric)


if __name__ == '__main__':
    unittest.main()
