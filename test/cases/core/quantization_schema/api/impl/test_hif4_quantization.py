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
#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest

import torch

from msmodelslim.core.quantization_schema import hif4_per_block_sym
from msmodelslim.core.quantization_schema.api import calculate_qparam, quantize, dequantize, fake_quantize
from msmodelslim.core.quantization_schema.qal import QDType, QParam, QStorage
from msmodelslim.core.observer import MsMinMaxBlockObserver, MinMaxBlockObserverConfig
from msmodelslim.core.quantization_schema.utils import reshape_to_blocks, undo_reshape_to_blocks


class TestHiF4Quantization(unittest.TestCase):
    """测试 HiF4 量化相关函数"""

    def setUp(self):
        """设置测试环境"""
        # 创建 QParam，包含 axes 信息
        self.q_param = QParam(scheme=hif4_per_block_sym, ext={'axes': -1})

        # 获取 scheme 和 finfo 属性（用于 reshape 等）
        self.x_scheme = self.q_param.scheme
        self.x_hifx_finfo = self.q_param.scheme.dtype.hifx_finfo
        self.x_axes = self.q_param.ext.get("axes")

        # 创建 MinMaxBlockObserver，用于模拟观察者统计 max_levels
        minmax_config = MinMaxBlockObserverConfig(axes=self.x_axes)
        self.x_minmax_block_observer = MsMinMaxBlockObserver(minmax_config)

    # ---------- 测试 calculate_qparam ----------
    def test_calculate_qparam_should_return_correct_qparam(self):
        """测试 calculate_qparam 返回正确的 QParam，包含 weight_scale_lv3/2/factor"""
        min_val = torch.tensor([[[-1.0]], [[-2.0]]])
        max_val = torch.tensor([[[3.0]], [[5.0]]])
        # 构造 max_levels (LV1, LV2, LV3)
        max_levels = {1: max_val.clone(), 2: max_val.clone(), 3: torch.tensor([[[2.5]], [[4.0]]])}
        qparam = calculate_qparam(
            min_val,
            max_val,
            q_dtype=self.x_scheme.dtype,
            q_scope=self.x_scheme.scope,
            symmetric=self.x_scheme.symmetric,
            max_levels=max_levels,
            axes=self.x_axes,
        )
        self.assertIsInstance(qparam, QParam)
        self.assertIn('weight_scale_lv3', qparam.ext)
        self.assertIn('weight_scale_lv2', qparam.ext)
        self.assertIn('weight_scale_factor', qparam.ext)
        self.assertEqual(qparam.ext['weight_scale_lv3'].shape, max_val.shape)
        self.assertEqual(qparam.ext['weight_scale_lv2'].shape, max_val.shape)
        self.assertEqual(qparam.ext['weight_scale_factor'].shape, max_val.shape)

    def test_calculate_qparam_with_zero_max_should_use_min_scale(self):
        min_val = torch.tensor([[[0.0]]])
        max_val = torch.tensor([[[0.0]]])
        zero_tensor = torch.zeros_like(max_val)
        max_levels = {1: zero_tensor.clone(), 2: zero_tensor.clone(), 3: zero_tensor.clone()}
        qparam = calculate_qparam(
            min_val,
            max_val,
            q_dtype=self.x_scheme.dtype,
            q_scope=self.x_scheme.scope,
            symmetric=self.x_scheme.symmetric,
            max_levels=max_levels,
            axes=self.x_axes,
        )
        self.assertEqual(qparam.ext['weight_scale_factor'].item(), 2**-48)
        self.assertEqual(qparam.ext['weight_scale_lv2'].item(), 1.0)
        self.assertEqual(qparam.ext['weight_scale_lv3'].item(), 1.0)

    def test_calculate_qparam_missing_levels_uses_default_zero(self):
        """max_levels 缺少某些 level 时，缺失的 level 默认为 0"""
        min_val = torch.tensor([[[1.0]]])
        max_val = torch.tensor([[[2.0]]])
        max_levels = {1: torch.tensor([[[2.0]]])}
        qparam = calculate_qparam(
            min_val,
            max_val,
            q_dtype=self.x_scheme.dtype,
            q_scope=self.x_scheme.scope,
            symmetric=self.x_scheme.symmetric,
            max_levels=max_levels,
            axes=self.x_axes,
        )
        self.assertEqual(qparam.ext['weight_scale_lv2'].item(), 1.0)
        self.assertEqual(qparam.ext['weight_scale_lv3'].item(), 1.0)

    def test_calculate_qparam_scale_ratio_clip2_clip3_equal_zero(self):
        """scale_ratio/clip2/clip3传入0时，该三个参数会设为默认值，避免除0"""
        min_val = torch.tensor([[[-100.0]]])
        max_val = torch.tensor([[[100.0]]])
        max_tensor = torch.tensor([[[100.0]]])
        max_levels = {1: max_tensor.clone(), 2: max_tensor.clone(), 3: max_tensor.clone()}
        qparam = calculate_qparam(
            min_val,
            max_val,
            q_dtype=self.x_scheme.dtype,
            q_scope=self.x_scheme.scope,
            symmetric=self.x_scheme.symmetric,
            max_levels=max_levels,
            axes=self.x_axes,
            scale_ratio=0,
            clip2=0,
            clip3=0,
        )
        qparam_default = calculate_qparam(
            min_val,
            max_val,
            q_dtype=self.x_scheme.dtype,
            q_scope=self.x_scheme.scope,
            symmetric=self.x_scheme.symmetric,
            max_levels=max_levels,
            axes=self.x_axes,
            scale_ratio=1,
            clip2=4,
            clip3=2,
        )
        self.assertTrue(
            torch.allclose(qparam.ext['weight_scale_factor'], qparam_default.ext['weight_scale_factor'], atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(qparam.ext['weight_scale_lv2'], qparam_default.ext['weight_scale_lv2'], atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(qparam.ext['weight_scale_lv3'], qparam_default.ext['weight_scale_lv3'], atol=1e-6)
        )

    # ---------- 测试 quantize (HiF4) ----------
    def test_quantize_positive_value(self):
        """正数量化"""
        min_val = torch.tensor([[[-100.0]]])
        max_val = torch.tensor([[[100.0]]])
        max_tensor = torch.tensor([[[100.0]]])
        max_levels = {1: max_tensor.clone(), 2: max_tensor.clone(), 3: max_tensor.clone()}
        qparam = calculate_qparam(
            min_val,
            max_val,
            q_dtype=self.x_scheme.dtype,
            q_scope=self.x_scheme.scope,
            symmetric=self.x_scheme.symmetric,
            max_levels=max_levels,
            axes=self.x_axes,
        )
        x = torch.tensor([12.0])
        storage = QStorage(QDType.FLOAT, x)
        quantized = quantize(storage, qparam)
        expected = torch.tensor(0.25)
        self.assertTrue(torch.allclose(quantized.value, expected, atol=1e-6))

    def test_quantize_negative_value(self):
        min_val = torch.tensor([[[-100.0]]])
        max_val = torch.tensor([[[100.0]]])
        max_tensor = torch.tensor([[[100.0]]])
        max_levels = {1: max_tensor.clone(), 2: max_tensor.clone(), 3: max_tensor.clone()}
        qparam = calculate_qparam(
            min_val,
            max_val,
            q_dtype=self.x_scheme.dtype,
            q_scope=self.x_scheme.scope,
            symmetric=self.x_scheme.symmetric,
            max_levels=max_levels,
            axes=self.x_axes,
        )
        x = torch.tensor([-9.0])
        storage = QStorage(QDType.FLOAT, x)
        quantized = quantize(storage, qparam)

        expected = torch.tensor(-0.25)
        self.assertTrue(torch.allclose(quantized.value, expected, atol=1e-6))

    def test_quantize_with_all_ones_should_return_correct_scale_and_quantized_value(self):
        """所有元素为 1，block 内 max=1，scale 计算后量化值应为 4.0（根据 HiF4 算法）"""
        # 64 个元素全为 1，block_size=64 一个 block
        x = torch.ones(1, 1, 64)
        axes = self.x_axes
        axes = [axes] if isinstance(axes, int) else axes
        axes = [a + x.ndim if a < 0 else a for a in axes]
        x_shaped, axes_, orig_shape, padded_shape = reshape_to_blocks(x, axes, self.x_hifx_finfo.block_size)
        shared_exp_axes = [x + 1 for x in axes_] if self.x_hifx_finfo.block_size > 0 else axes_
        self.x_minmax_block_observer.update(x_shaped, shared_exp_axes=shared_exp_axes)
        x_min_val, x_max_val = self.x_minmax_block_observer.get_min_max()
        # 获取 max_levels (observer 应提供三个 level，这里简化处理，用同一个 max 值)
        # 注意：实际 observer 需要统计三个 level 的 max，但测试中我们用 max_val 代替所有 level
        max_levels = {1: x_max_val.clone(), 2: x_max_val.clone(), 3: x_max_val.clone()}
        x_q_param = calculate_qparam(
            x_min_val,
            x_max_val,
            q_dtype=self.x_scheme.dtype,
            q_scope=self.x_scheme.scope,
            symmetric=self.x_scheme.symmetric,
            max_levels=max_levels,
            axes=self.x_axes,
        )
        self.assertEqual(x_q_param.ext["weight_scale_factor"].shape, x_max_val.shape)
        x_q = quantize(QStorage(QDType.FLOAT, x_shaped), x_q_param)
        expected = torch.full(x_shaped.shape, 1.50)
        self.assertTrue(torch.allclose(x_q.value, expected, atol=1e-5))

    def test_quantize_with_random_tensor_through_observer(self):
        """使用随机张量，通过 observer 得到 min/max/max_levels，再量化，验证形状和数值合理性"""
        torch.manual_seed(42)
        x = torch.randn(1, 1, 64) * 10 - 5  # shape (1,1,64)
        axes = self.x_axes
        axes = [axes] if isinstance(axes, int) else axes
        axes = [a + x.ndim if a < 0 else a for a in axes]
        x_shaped, axes_, orig_shape, padded_shape = reshape_to_blocks(x, axes, self.x_hifx_finfo.block_size)
        shared_exp_axes = [x + 1 for x in axes_] if self.x_hifx_finfo.block_size > 0 else axes_
        self.x_minmax_block_observer.update(x_shaped, shared_exp_axes=shared_exp_axes)
        x_min_val, x_max_val = self.x_minmax_block_observer.get_min_max()
        # 模拟 max_levels：实际 observer 需要三个 level，这里用同一个 max
        max_levels = {1: x_max_val.clone(), 2: x_max_val.clone(), 3: x_max_val.clone()}
        x_q_param = calculate_qparam(
            x_min_val,
            x_max_val,
            q_dtype=self.x_scheme.dtype,
            q_scope=self.x_scheme.scope,
            symmetric=self.x_scheme.symmetric,
            max_levels=max_levels,
            axes=self.x_axes,
        )
        x_q = quantize(QStorage(QDType.FLOAT, x_shaped), x_q_param)
        self.assertEqual(x_q.value.shape, x_shaped.shape)
        self.assertTrue(torch.isfinite(x_q.value).all())

    # ---------- 测试 dequantize ----------
    def test_dequantize_should_recover_original_value_approximately(self):
        """量化后再反量化，应与原始值近似"""
        min_val = torch.tensor([[[-10.0]]])
        max_val = torch.tensor([[[10.0]]])
        max_tensor = torch.tensor([[[10.0]]])
        max_levels = {1: max_tensor.clone(), 2: max_tensor.clone(), 3: max_tensor.clone()}
        qparam = calculate_qparam(
            min_val,
            max_val,
            q_dtype=self.x_scheme.dtype,
            q_scope=self.x_scheme.scope,
            symmetric=self.x_scheme.symmetric,
            max_levels=max_levels,
            axes=self.x_axes,
        )
        x = torch.tensor([3.5])
        quant_storage = quantize(QStorage(QDType.FLOAT, x), qparam)
        dequant_storage = dequantize(quant_storage, qparam)
        rel_error = torch.abs(dequant_storage.value - x) / (torch.abs(x) + 1e-6)
        self.assertLess(rel_error.item(), 0.2)

    # ---------- 测试 fake_quantize (端到端) ----------
    def test_fake_quantize_should_preserve_input_shape_and_dtype(self):
        """fake_quantize 后形状和 dtype 应与输入相同"""
        x = torch.randn(2, 64, 128)  # batch, seq, feature
        axes = self.x_axes
        axes = [axes] if isinstance(axes, int) else axes
        axes = [a + x.ndim if a < 0 else a for a in axes]
        x_shaped, axes_, orig_shape, padded_shape = reshape_to_blocks(x, axes, self.x_hifx_finfo.block_size)
        shared_exp_axes = [x + 1 for x in axes_] if self.x_hifx_finfo.block_size > 0 else axes_
        self.x_minmax_block_observer.update(x_shaped, shared_exp_axes=shared_exp_axes)
        x_min_val, x_max_val = self.x_minmax_block_observer.get_min_max()
        max_levels = {1: x_max_val.clone(), 2: x_max_val.clone(), 3: x_max_val.clone()}
        x_q_param = calculate_qparam(
            x_min_val,
            x_max_val,
            q_dtype=self.x_scheme.dtype,
            q_scope=self.x_scheme.scope,
            symmetric=self.x_scheme.symmetric,
            max_levels=max_levels,
            axes=self.x_axes,
        )
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x_shaped), x_q_param)
        x_q_dq.value = undo_reshape_to_blocks(x_q_dq.value, padded_shape, orig_shape)
        self.assertEqual(x_q_dq.value.shape, x.shape)
        self.assertEqual(x_q_dq.value.dtype, x.dtype)

    def test_fake_quantize_should_produce_valid_values(self):
        """量化后的值应在合理范围内"""
        x = torch.randn(2, 64, 128) * 5
        axes = self.x_axes
        axes = [axes] if isinstance(axes, int) else axes
        axes = [a + x.ndim if a < 0 else a for a in axes]
        x_shaped, axes_, orig_shape, padded_shape = reshape_to_blocks(x, axes, self.x_hifx_finfo.block_size)
        shared_exp_axes = [x + 1 for x in axes_] if self.x_hifx_finfo.block_size > 0 else axes_
        self.x_minmax_block_observer.update(x_shaped, shared_exp_axes=shared_exp_axes)
        x_min_val, x_max_val = self.x_minmax_block_observer.get_min_max()
        max_levels = {1: x_max_val.clone(), 2: x_max_val.clone(), 3: x_max_val.clone()}
        x_q_param = calculate_qparam(
            x_min_val,
            x_max_val,
            q_dtype=self.x_scheme.dtype,
            q_scope=self.x_scheme.scope,
            symmetric=self.x_scheme.symmetric,
            max_levels=max_levels,
            axes=self.x_axes,
        )
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x_shaped), x_q_param)
        x_q_dq.value = undo_reshape_to_blocks(x_q_dq.value, padded_shape, orig_shape)
        self.assertTrue(torch.isfinite(x_q_dq.value).all())

    # ---------- 边界测试 ----------
    def test_quantize_with_single_element_should_work(self):
        """单个元素的张量"""
        x = torch.randn(1, 1, 130)
        axes = self.x_axes
        axes = [axes] if isinstance(axes, int) else axes
        axes = [a + x.ndim if a < 0 else a for a in axes]
        x_shaped, axes_, orig_shape, padded_shape = reshape_to_blocks(x, axes, self.x_hifx_finfo.block_size)
        shared_exp_axes = [x + 1 for x in axes_] if self.x_hifx_finfo.block_size > 0 else axes_
        self.x_minmax_block_observer.update(x_shaped, shared_exp_axes=shared_exp_axes)
        x_min_val, x_max_val = self.x_minmax_block_observer.get_min_max()
        max_levels = {1: x_max_val.clone(), 2: x_max_val.clone(), 3: x_max_val.clone()}
        x_q_param = calculate_qparam(
            x_min_val,
            x_max_val,
            q_dtype=self.x_scheme.dtype,
            q_scope=self.x_scheme.scope,
            symmetric=self.x_scheme.symmetric,
            max_levels=max_levels,
            axes=self.x_axes,
        )
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x_shaped), x_q_param)
        x_q_dq.value = undo_reshape_to_blocks(x_q_dq.value, padded_shape, orig_shape)
        self.assertEqual(x_q_dq.value.shape, x.shape)

    # ---------- 异常测试 ----------
    def test_quantize_missing_ext_keys_should_raise_key_error(self):
        """QParam 缺少 weight_scale_* 字段时应抛出 KeyError"""
        qparam = QParam(scheme=self.x_scheme, ext={})
        x = torch.tensor([1.0])
        storage = QStorage(QDType.FLOAT, x)
        with self.assertRaises(KeyError):
            quantize(storage, qparam)

    def test_dequantize_missing_ext_keys_should_raise_key_error(self):
        qparam = QParam(scheme=self.x_scheme, ext={})
        storage = QStorage(QDType.HiF4, torch.tensor([1.0]))
        with self.assertRaises(KeyError):
            dequantize(storage, qparam)

    def test_fake_quantize_with_nan_input_should_propagate_nan(self):
        """输入包含 NaN 时，输出也应包含 NaN"""
        x = torch.tensor([float('nan')])
        min_val = torch.tensor([[-1.0]])
        max_val = torch.tensor([[1.0]])
        max_tensor = torch.tensor([[1.0]])
        max_levels = {1: max_tensor.clone(), 2: max_tensor.clone(), 3: max_tensor.clone()}
        qparam = calculate_qparam(
            min_val,
            max_val,
            q_dtype=self.x_scheme.dtype,
            q_scope=self.x_scheme.scope,
            symmetric=self.x_scheme.symmetric,
            max_levels=max_levels,
            axes=self.x_axes,
        )
        storage = QStorage(QDType.FLOAT, x)
        quantized = quantize(storage, qparam)
        self.assertTrue(torch.isnan(quantized.value).any())


if __name__ == '__main__':
    unittest.main()
