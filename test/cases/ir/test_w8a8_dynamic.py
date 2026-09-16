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
from torch.nn import functional as F

from msmodelslim.ir.api import calculate_qparam, fake_quantize
from msmodelslim.ir.auto import AutoFakeQuantLinear
from msmodelslim.ir.const import int8_per_channel_sym, int8_per_token_sym
from msmodelslim.ir.qal import QDType, QParam, QScope, QStorage
from msmodelslim.ir.w8a8_dynamic import W8A8DynamicPerChannelFakeQuantLinear

_IN_FEATURES = 4
_OUT_FEATURES = 4


def _make_weight_scale(out_features: int = _OUT_FEATURES) -> torch.Tensor:
    """构造 per-channel 权重量化 scale。"""
    return torch.full((out_features,), 0.1, dtype=torch.float32)


def _make_module(weight: torch.Tensor, weight_scale: torch.Tensor, bias=None) -> W8A8DynamicPerChannelFakeQuantLinear:
    """按（激活 per-token，权重 per-channel）方案构造 IR。"""
    x_q_param = QParam(scheme=int8_per_token_sym, ext={"scale": torch.ones(1, dtype=torch.float32)})
    w_q_param = QParam(scheme=int8_per_channel_sym, ext={"scale": weight_scale})
    w_q = QStorage(dtype=QDType.INT8, value=weight)
    return W8A8DynamicPerChannelFakeQuantLinear(x_q_param, w_q_param, w_q, bias)


class TestW8A8DynamicPerChannelFakeQuantLinear(unittest.TestCase):
    """对应 msmodelslim/ir/w8a8_dynamic.py 中的 W8A8DynamicPerChannelFakeQuantLinear。"""

    @staticmethod
    def _make_identity_module(bias=None) -> W8A8DynamicPerChannelFakeQuantLinear:
        """构造权重为单位阵且权重量化 scale 为 1 的 IR，此时 forward 输出即激活伪量化结果。"""
        weight = torch.eye(_OUT_FEATURES, dtype=torch.float32).to(torch.int8)
        weight_scale = torch.ones(_OUT_FEATURES, dtype=torch.float32)
        return _make_module(weight, weight_scale, bias)

    @staticmethod
    def _per_token_reference_output(module, x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor):
        """按激活 per-token、权重 per-channel 的语义手工计算参考输出。"""
        x_reshape = x.reshape(-1, x.shape[-1])
        x_q_param = calculate_qparam(
            torch.amin(x_reshape, dim=1, keepdim=True),
            torch.amax(x_reshape, dim=1, keepdim=True),
            QDType.INT8,
            QScope.PER_TOKEN,
            True,
        )
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x_reshape), x_q_param).value
        weight_q_dq = weight.to(torch.float32) * weight_scale.unsqueeze(1)
        return F.linear(x_q_dq.reshape(x.shape), weight_q_dq, module.bias)

    def test_init_stores_int8_weight_and_scale_when_int8_tensors_given(self):
        """场景：传入 int8 权重与 per-channel 权重 scale。预期：二者被原样保存且不参与梯度。"""
        weight = torch.randint(-127, 127, (_OUT_FEATURES, _IN_FEATURES), dtype=torch.int8)
        weight_scale = _make_weight_scale()

        module = _make_module(weight, weight_scale)

        self.assertEqual(module.weight.dtype, torch.int8)
        self.assertTrue(torch.equal(module.weight, weight))
        self.assertTrue(torch.equal(module.weight_scale, weight_scale))
        self.assertFalse(module.weight.requires_grad)
        self.assertFalse(module.weight_scale.requires_grad)

    def test_init_converts_weight_to_int8_when_float32_weight_given(self):
        """场景：传入整数值的 float32 权重（如量化后以 float 形式落盘）。预期：权重被转换为 int8 存储。"""
        weight = torch.randint(-127, 127, (_OUT_FEATURES, _IN_FEATURES)).to(torch.float32)

        module = _make_module(weight, _make_weight_scale())

        self.assertEqual(module.weight.dtype, torch.int8)
        self.assertTrue(torch.equal(module.weight, weight.to(torch.int8)))

    def test_init_sets_bias_none_when_bias_not_given(self):
        """场景：不传入 bias（Linear 无偏置）。预期：bias 为 None。"""
        module = _make_module(torch.zeros(_OUT_FEATURES, _IN_FEATURES, dtype=torch.int8), _make_weight_scale())

        self.assertIsNone(module.bias)

    def test_init_keeps_bias_when_bias_given(self):
        """场景：传入 bias。预期：bias 被原样保存且不参与梯度。"""
        bias = torch.arange(_OUT_FEATURES, dtype=torch.float32)

        module = _make_module(torch.zeros(_OUT_FEATURES, _IN_FEATURES, dtype=torch.int8), _make_weight_scale(), bias)

        self.assertIsNotNone(module.bias)
        self.assertTrue(torch.equal(module.bias, bias))
        self.assertFalse(module.bias.requires_grad)

    def test_init_raises_key_error_when_weight_scale_missing(self):
        """场景：权重量化参数缺少 scale（量化方案与 IR 不匹配）。预期：抛 KeyError，而非静默使用错误量化参数。"""
        w_q_param = QParam(scheme=int8_per_channel_sym, ext={})
        w_q = QStorage(dtype=QDType.INT8, value=torch.zeros(_OUT_FEATURES, _IN_FEATURES, dtype=torch.int8))

        with self.assertRaises(KeyError):
            W8A8DynamicPerChannelFakeQuantLinear(QParam(scheme=int8_per_token_sym), w_q_param, w_q, None)

    def test_create_returns_per_channel_ir_when_per_token_act_and_per_channel_weight(self):
        """场景：以（激活 per-token，权重 per-channel）方案通过注册表创建。预期：分发到本 IR。"""
        x_q_param = QParam(scheme=int8_per_token_sym)
        w_q_param = QParam(scheme=int8_per_channel_sym, ext={"scale": _make_weight_scale()})
        w_q = QStorage(dtype=QDType.INT8, value=torch.zeros(_OUT_FEATURES, _IN_FEATURES, dtype=torch.int8))

        module = AutoFakeQuantLinear.create(x_q_param, w_q_param, w_q, None)

        self.assertIsInstance(module, W8A8DynamicPerChannelFakeQuantLinear)

    def test_is_atomic_returns_true_when_called(self):
        """场景：查询 IR 原子性。预期：本 IR 为原子 IR，返回 True。"""
        self.assertTrue(W8A8DynamicPerChannelFakeQuantLinear.is_atomic())

    def test_named_modules_yields_self_only_when_atomic(self):
        """场景：遍历原子 IR 的子模块。预期：只产出自身，不下钻展开内部参数。"""
        module = self._make_identity_module()

        self.assertEqual([name for name, _ in module.named_modules()], [""])

    def test_forward_returns_same_shape_when_2d_input(self):
        """场景：2D 激活输入 (tokens, in_features)。预期：输出 shape 与输入一致且结果有限。"""
        weight = torch.randint(-127, 127, (_OUT_FEATURES, _IN_FEATURES), dtype=torch.int8)
        module = _make_module(weight, _make_weight_scale())
        x = torch.randn(3, _IN_FEATURES, dtype=torch.float32)

        output = module(x)

        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_forward_returns_same_shape_when_3d_input(self):
        """场景：3D 激活输入 (batch, seq, in_features)。预期：输出 shape 与输入一致，token 维度被正确展平处理。"""
        weight = torch.randint(-127, 127, (_OUT_FEATURES, _IN_FEATURES), dtype=torch.int8)
        module = _make_module(weight, _make_weight_scale())
        x = torch.randn(2, 3, _IN_FEATURES, dtype=torch.float32)

        output = module(x)

        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_forward_adds_bias_when_bias_given(self):
        """场景：权重全 0（反量化后为 0）且提供 bias。预期：输出恒等于 bias。"""
        bias = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float32)
        module = _make_module(torch.zeros(_OUT_FEATURES, _IN_FEATURES, dtype=torch.int8), _make_weight_scale(), bias)
        x = torch.randn(2, _IN_FEATURES, dtype=torch.float32)

        output = module(x)

        self.assertTrue(torch.allclose(output, bias.expand_as(output), atol=1e-6))

    def test_forward_preserves_each_token_when_tokens_have_different_magnitudes(self):
        """场景：同一 batch 内两个 token 的数值量级相差 200 倍，权重为单位阵。
        预期：每个 token 用自身 min/max 计算量化参数，两个 token 都能被精确表示（per-channel 量化则小 token 会被压掉）。
        """
        module = self._make_identity_module()
        x = torch.tensor([[0.5, -0.5, 0.5, -0.5], [100.0, -100.0, 100.0, -100.0]], dtype=torch.float32)

        output = module(x)

        self.assertTrue(torch.allclose(output, x, atol=1e-4))

    def test_forward_preserves_value_when_single_token_with_constant_input(self):
        """场景：单 token 且各通道取值相同（tokens=1，min=max）的边界情形。预期：shape 保持且数值被还原。"""
        module = self._make_identity_module()
        x = torch.ones(1, _IN_FEATURES, dtype=torch.float32)

        output = module(x)

        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.allclose(output, x, atol=1e-5))

    def test_forward_returns_zeros_when_token_is_all_zero(self):
        """场景：token 全 0，量化 scale 退化到浮点 eps。预期：输出全 0 且无 NaN/Inf。"""
        module = self._make_identity_module()
        x = torch.zeros(2, _IN_FEATURES, dtype=torch.float32)

        output = module(x)

        self.assertTrue(torch.equal(output, torch.zeros_like(x)))
        self.assertTrue(torch.isfinite(output).all())

    def test_forward_equals_row_wise_forward_when_batch_has_multiple_tokens(self):
        """场景：一次性输入多行激活，与逐行分别前向的结果对比（per-token 量化结果不应依赖同 batch 的其他 token）。
        预期：两种方式逐行一致；若退化为 per-channel（dim=0）量化则该性质不成立。
        """
        torch.manual_seed(0)
        weight = torch.randint(-127, 127, (_OUT_FEATURES, _IN_FEATURES), dtype=torch.int8)
        module = _make_module(weight, _make_weight_scale())
        x = torch.cat([torch.randn(4, _IN_FEATURES) * 0.01, torch.randn(4, _IN_FEATURES) * 10.0], dim=0)

        batched = module(x)

        for index in range(x.shape[0]):
            row_wise = module(x[index : index + 1])
            self.assertTrue(torch.allclose(batched[index : index + 1], row_wise, atol=1e-6))

    def test_forward_equals_per_token_reference_when_multiple_tokens(self):
        """场景：多行随机激活，按激活 per-token（沿最后一维求 min/max，keepdim）+ 权重 per-channel 计算参考输出。
        预期：forward 结果与 per-token 参考一致；若 min/max 误按 dim=0 或丢失 keepdim 求取，则与参考不符。
        """
        torch.manual_seed(1)
        weight = torch.randint(-127, 127, (_OUT_FEATURES, _IN_FEATURES), dtype=torch.int8)
        weight_scale = _make_weight_scale()
        module = _make_module(weight, weight_scale)
        x = torch.randn(5, _IN_FEATURES, dtype=torch.float32) * 3

        expected = self._per_token_reference_output(module, x, weight, weight_scale)

        self.assertTrue(torch.allclose(module(x), expected, atol=1e-6))

    def test_forward_raises_runtime_error_when_input_last_dim_mismatches_weight(self):
        """场景：激活最后一维与权重 in_features 不一致（错误的模型接线）。预期：抛 RuntimeError，不返回错误结果。"""
        module = self._make_identity_module()
        x = torch.randn(2, _IN_FEATURES + 1, dtype=torch.float32)

        with self.assertRaises(RuntimeError):
            module(x)


if __name__ == '__main__':
    unittest.main()
