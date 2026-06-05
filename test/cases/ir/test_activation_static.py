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

from unittest.mock import patch, MagicMock

import torch
from torch import nn

from msmodelslim.ir.qal import QParam, QScope


class BaseActivationTestMixin:
    """测试 FakeQuantActivationPerHead 系列类的公共 Mixin 类（不继承 TestCase）"""

    # 子类必须覆盖以下属性
    module_class = None  # 待测试的类，如 INT8FakeQuantActivationPerHead
    scheme_const = None  # 量化方案常量，如 int8_per_head_sym
    fake_quantize_path = None  # fake_quantize 函数的 mock 路径
    expected_dtype = None  # 期望的 QDType，如 QDType.INT8

    def setUp(self):
        """构造一个包含合法 scale 的 QParam"""
        self.num_heads = 4
        self.scale = torch.rand(self.num_heads, dtype=torch.float32)
        ext = {"scale": self.scale}
        self.q_param = QParam(scheme=self.scheme_const, ext=ext)

    # ---------- 辅助方法 ----------
    def _make_input(self, batch=2, num_heads=4, seq_len=10, head_dim=16, dtype=torch.float32, requires_grad=False):
        return torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, requires_grad=requires_grad)

    def _get_expected_x_2d(self, x):
        """返回 forward 内部 fake_quantize 接收的 2D 张量。
        形状: (batch * seq_len * head_dim, num_heads)
        """
        return x.movedim(1, -1).reshape(-1, x.shape[1])

    def _rand_like(self, x):
        return torch.rand_like(x)

    # ---------- 初始化测试 ----------
    def test_init_stores_scheme_and_scale_correctly(self):
        """初始化后，x_q_scheme 和 input_scale 被正确保存"""
        module = self.module_class(self.q_param)  # pylint: disable=not-callable
        self.assertEqual(module.x_q_scheme, self.scheme_const)
        self.assertTrue(torch.equal(module.input_scale.data, self.scale))
        self.assertIsInstance(module.input_scale, nn.Parameter)
        self.assertFalse(module.input_scale.requires_grad)

    def test_init_missing_scale_raises_attribute_error_when_ext_is_none(self):
        """ext 为 None 时，ext.get('scale') 抛出 AttributeError"""
        bad_param = QParam(scheme=self.scheme_const)  # ext 默认为 None
        with self.assertRaises(AttributeError):
            self.module_class(bad_param)  # pylint: disable=not-callable

    # ---------- forward 测试 ----------
    def test_forward_calls_fake_quantize_with_per_head_scheme(self):
        """forward 调用 fake_quantize，内部转换为 per-channel 量化最后一维（num_heads）"""
        with patch(self.fake_quantize_path) as mock_fq:
            x = self._make_input(batch=2, num_heads=self.num_heads, seq_len=10, head_dim=16)
            expected_x_2d = self._get_expected_x_2d(x)
            fake_result = MagicMock()
            fake_result.value = torch.rand_like(expected_x_2d)
            mock_fq.return_value = fake_result

            module = self.module_class(self.q_param)  # pylint: disable=not-callable
            _ = module(x)  # 触发调用

            mock_fq.assert_called_once()
            call_args = mock_fq.call_args[0]
            q_storage = call_args[0]
            called_q_param = call_args[1]

            self.assertEqual(q_storage.value.shape, expected_x_2d.shape, "QStorage.value shape mismatch")
            self.assertTrue(
                torch.allclose(q_storage.value, expected_x_2d), "QStorage.value should be the reshaped input"
            )

            self.assertIsInstance(called_q_param, QParam)
            scheme = called_q_param.scheme
            self.assertEqual(scheme.scope, QScope.PER_CHANNEL)
            self.assertEqual(scheme.dtype, self.expected_dtype)
            self.assertTrue(scheme.symmetric)
            self.assertTrue(
                torch.equal(called_q_param.ext["scale"], self.scale), "ext['scale'] should be the original input_scale"
            )

    def test_forward_output_shape_matches_input(self):
        """forward 输出的形状与输入完全相同"""
        with patch(self.fake_quantize_path) as mock_fq:
            module = self.module_class(self.q_param)  # pylint: disable=not-callable
            for shape in [(2, 4, 10, 16), (1, 1, 1, 8), (4, 8, 32, 64)]:
                x = torch.randn(*shape)
                x_2d = self._get_expected_x_2d(x)

                # 修复 cell-var-from-loop: 将 x_2d 作为默认参数绑定
                def side_effect(_, __, x_2d=x_2d):
                    mock_value = MagicMock()
                    mock_value.value = self._rand_like(x_2d)
                    return mock_value

                mock_fq.side_effect = side_effect
                out = module(x)
                self.assertEqual(out.shape, x.shape, f"shape mismatch for input {shape}")

    def test_forward_preserves_dtype(self):
        """forward 输出的 dtype 应与输入相同"""
        with patch(self.fake_quantize_path) as mock_fq:
            module = self.module_class(self.q_param)  # pylint: disable=not-callable
            for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                x = self._make_input(dtype=dtype)
                x_2d = self._get_expected_x_2d(x)

                def side_effect(_, __, x_2d=x_2d):
                    mock_value = MagicMock()
                    mock_value.value = self._rand_like(x_2d)
                    return mock_value

                mock_fq.side_effect = side_effect
                out = module(x)
                self.assertEqual(out.dtype, dtype)

    def test_forward_output_does_not_require_grad(self):
        """量化后输出不应带梯度"""
        with patch(self.fake_quantize_path) as mock_fq:
            module = self.module_class(self.q_param)  # pylint: disable=not-callable
            x = self._make_input(requires_grad=True)
            x_2d = self._get_expected_x_2d(x)

            def side_effect(_, __, x_2d=x_2d):
                mock_value = MagicMock()
                mock_value.value = self._rand_like(x_2d)
                return mock_value

            mock_fq.side_effect = side_effect
            out = module(x)
            self.assertFalse(out.requires_grad)

    def test_forward_handles_negative_values(self):
        """输入包含负值时不应报错"""
        with patch(self.fake_quantize_path) as mock_fq:
            module = self.module_class(self.q_param)  # pylint: disable=not-callable
            x = -torch.abs(self._make_input())
            x_2d = self._get_expected_x_2d(x)

            def side_effect(_, __, x_2d=x_2d):
                mock_value = MagicMock()
                mock_value.value = self._rand_like(x_2d)
                return mock_value

            mock_fq.side_effect = side_effect
            out = module(x)
            self.assertEqual(out.shape, x.shape)

    def test_forward_quantization_alters_values(self):
        """简单验证量化前后数值发生变化（mock 返回明显不同的值）"""
        with patch(self.fake_quantize_path) as mock_fq:
            module = self.module_class(self.q_param)  # pylint: disable=not-callable
            x = self._make_input()
            x_2d_orig = self._get_expected_x_2d(x)

            def side_effect(_, __, x_2d_orig=x_2d_orig):
                mock_value = MagicMock()
                mock_value.value = x_2d_orig + 100.0
                return mock_value

            mock_fq.side_effect = side_effect
            out = module(x)
            self.assertFalse(torch.allclose(out, x), "Quantized output should differ from input")

    # ---------- 边界与属性测试 ----------
    def test_forward_single_head(self):
        """单 head 输入应正常工作"""
        scale_single = torch.rand(1)
        ext = {"scale": scale_single}
        q_param = QParam(scheme=self.scheme_const, ext=ext)
        with patch(self.fake_quantize_path) as mock_fq:
            module = self.module_class(q_param)  # pylint: disable=not-callable
            x = torch.randn(2, 1, 10, 16)
            x_2d = self._get_expected_x_2d(x)

            def side_effect(_, __, x_2d=x_2d):
                mock_value = MagicMock()
                mock_value.value = torch.rand_like(x_2d)
                return mock_value

            mock_fq.side_effect = side_effect
            out = module(x)
            self.assertEqual(out.shape, x.shape)

    def test_scheme_property_consistency(self):
        """验证 x_q_scheme 属性的各字段"""
        module = self.module_class(self.q_param)  # pylint: disable=not-callable
        scheme = module.x_q_scheme
        self.assertEqual(scheme, self.scheme_const)
        self.assertEqual(scheme.scope, QScope.PER_HEAD)
        self.assertEqual(scheme.dtype, self.expected_dtype)
        self.assertTrue(scheme.symmetric)

    def test_scale_is_parameter_without_grad(self):
        """input_scale 应为 nn.Parameter 且 requires_grad=False"""
        module = self.module_class(self.q_param)  # pylint: disable=not-callable
        self.assertIsInstance(module.input_scale, nn.Parameter)
        self.assertFalse(module.input_scale.requires_grad)
