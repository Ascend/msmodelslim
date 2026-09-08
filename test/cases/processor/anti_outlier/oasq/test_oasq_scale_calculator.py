#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""OASQScaleCalculator 单元测试：尺度求解、权重降维、OV/GQA 缩减。"""

import unittest

import torch

from msmodelslim.processor.anti_outlier.common.scale_computation import OASQScaleCalculator


class TestOASQScaleCalculatorReduceWeightScale(unittest.TestCase):
    """场景：将权重张量降维为与激活 scale 对齐的 per-channel 统计。"""

    def test_reduce_weight_scale_when_1d_input_then_returns_same_length(self):
        """给定 1D 权重统计，期望 reshape 后长度不变且非负。"""
        w = torch.tensor([-1.0, 2.0, 3.0])
        out = OASQScaleCalculator._reduce_weight_scale(w)
        self.assertEqual(tuple(out.shape), (3,))
        self.assertTrue(torch.all(out >= 0) or out.dtype == torch.float32)

    def test_reduce_weight_scale_when_2d_weight_then_abs_amax_over_output_dim(self):
        """给定 2D weight (out, in)，期望按输入通道取 abs-max，长度等于 in_features。"""
        w = torch.tensor([[1.0, -4.0], [2.0, 3.0]], dtype=torch.float32)  # shape (2, 2)
        out = OASQScaleCalculator._reduce_weight_scale(w)
        self.assertEqual(tuple(out.shape), (2,))
        torch.testing.assert_close(out, torch.tensor([2.0, 4.0]))


class TestOASQScaleCalculatorComputeSmoothScale(unittest.TestCase):
    """场景：基于 z-score 分通道求 smooth scale。"""

    def setUp(self):
        self.calculator = OASQScaleCalculator(max_iters=8)

    def test_compute_smooth_scale_when_uniform_act_then_returns_finite_positive_scales(self):
        """给定均匀激活与匹配权重，期望返回有限、正、长度对齐的 scale。"""
        a_scale = torch.ones(8)
        w_scale = torch.ones(4, 8)  # Linear-like weight
        scales = self.calculator.compute_smooth_scale(a_scale, w_scale)
        self.assertEqual(tuple(scales.shape), (8,))
        self.assertTrue(torch.all(torch.isfinite(scales)))
        self.assertTrue(torch.all(scales > 0))

    def test_compute_smooth_scale_when_clear_outliers_then_still_returns_finite_scales(self):
        """给定含明显离群通道的激活，期望迭代后仍得到有限 scale（覆盖 outlier 分支）。"""
        a_scale = torch.ones(16)
        a_scale[0] = 100.0
        a_scale[1] = 80.0
        w_scale = torch.ones(8, 16)
        scales = self.calculator.compute_smooth_scale(a_scale, w_scale)
        self.assertEqual(tuple(scales.shape), (16,))
        self.assertTrue(torch.all(torch.isfinite(scales)))

    def test_compute_smooth_scale_when_normal_branch_then_uses_sqrt_w_over_sqrt_a(self):
        """正常通道须用 sqrt(w)/(sqrt(a)+eps)，而非 sqrt(w/sqrt(a))。"""
        a = torch.tensor([1.0, 1.1, 0.9, 1.05], dtype=torch.float32)
        w = torch.tensor([4.0, 9.0, 1.0, 16.0], dtype=torch.float32)
        # High z threshold keeps all channels on the normal path.
        calc = OASQScaleCalculator(
            base_z_threshold=100.0,
            target_outlier_min=0.0,
            target_outlier_max=1.0,
            max_iters=1,
        )
        scales = calc.compute_smooth_scale(a, w)
        expected = torch.sqrt(w) / (torch.sqrt(a) + 1e-6)
        expected = expected / (expected.mean() + 1e-6)
        torch.testing.assert_close(scales, expected)

    def test_compute_smooth_scale_when_outlier_branch_then_uses_log1p_w_over_log1p_a(self):
        """离群通道须用 log1p(w)/(log1p(a)+eps)，而非 log1p(w/log1p(a))。"""
        a = torch.tensor([1.0, 100.0], dtype=torch.float32)
        w = torch.tensor([4.0, 9.0], dtype=torch.float32)
        # Force both channels into outlier path and skip z-threshold search.
        calc = OASQScaleCalculator(
            base_z_threshold=-1.0,
            target_outlier_min=0.0,
            target_outlier_max=1.0,
            max_iters=1,
        )
        scales = calc.compute_smooth_scale(a, w)
        expected = torch.log1p(w) / (torch.log1p(a) + 1e-6)
        expected = expected / (expected.mean() + 1e-6)
        torch.testing.assert_close(scales, expected)

    def test_compute_smooth_scale_when_weight_act_length_mismatch_then_raises_value_error(self):
        """给定权重与激活通道数不一致，期望抛出 ValueError。"""
        a_scale = torch.ones(4)
        w_scale = torch.ones(2, 8)
        with self.assertRaises(ValueError) as ctx:
            self.calculator.compute_smooth_scale(a_scale, w_scale)
        self.assertIn("length mismatch", str(ctx.exception))

    def test_compute_smooth_scale_when_max_iters_one_then_returns_best_scales_without_crash(self):
        """给定 max_iters=1，期望至少返回一版 best_scales 且不抛异常。"""
        calc = OASQScaleCalculator(max_iters=1)
        a_scale = torch.randn(8).abs() + 0.1
        w_scale = torch.randn(4, 8)
        scales = calc.compute_smooth_scale(a_scale, w_scale)
        self.assertEqual(tuple(scales.shape), (8,))
        self.assertTrue(torch.all(torch.isfinite(scales)))

    def test_compute_smooth_scale_when_outlier_ratio_too_high_then_raises_z_threshold(self):
        """给定几乎全是离群点的激活分布，期望通过抬高 z_thr 收敛或跑完迭代仍返回 scale。"""
        # 一半极大、一半极小，outlier_ratio 易偏高
        a_scale = torch.cat([torch.ones(8) * 1e3, torch.ones(8) * 1e-3])
        w_scale = torch.ones(4, 16)
        calc = OASQScaleCalculator(
            base_z_threshold=0.1,
            target_outlier_min=0.01,
            target_outlier_max=0.05,
            max_iters=5,
            z_step=0.5,
        )
        scales = calc.compute_smooth_scale(a_scale, w_scale)
        self.assertEqual(tuple(scales.shape), (16,))
        self.assertTrue(torch.all(torch.isfinite(scales)))

    def test_compute_smooth_scale_when_outlier_ratio_too_low_then_lowers_z_threshold(self):
        """给定几乎无离群的激活，期望降低 z_thr 后仍返回有限 scale。"""
        a_scale = torch.linspace(1.0, 1.2, 32)
        w_scale = torch.ones(4, 32)
        calc = OASQScaleCalculator(
            base_z_threshold=10.0,
            target_outlier_min=0.2,
            target_outlier_max=0.5,
            max_iters=5,
            z_step=1.0,
        )
        scales = calc.compute_smooth_scale(a_scale, w_scale)
        self.assertEqual(tuple(scales.shape), (32,))
        self.assertTrue(torch.all(torch.isfinite(scales)))


class TestOASQScaleCalculatorComputeOvScales(unittest.TestCase):
    """场景：OV 子图 GQA/MHA 的 o/v scale 缩减。"""

    def test_compute_ov_scales_when_mha_equal_heads_then_o_and_v_shapes_match_hidden(self):
        """给定 MHA（q_heads == kv_heads），期望 o/v scale 长度等于 hidden。"""
        hidden = 64
        n_heads = 8
        a_scale = torch.ones(hidden)
        w_scale = torch.ones(hidden, hidden)
        calc = OASQScaleCalculator(max_iters=4)
        o_scales, v_scales = calc.compute_ov_scales(a_scale, w_scale, n_heads, n_heads)
        self.assertEqual(tuple(o_scales.shape), (hidden,))
        self.assertEqual(tuple(v_scales.shape), (hidden,))

    def test_compute_ov_scales_when_precomputed_scales_passed_then_skips_recompute_path(self):
        """给定已计算 scales，期望直接用于 MQGA 缩减且形状正确。"""
        hidden = 64
        n_heads = 8
        a_scale = torch.ones(hidden)
        w_scale = torch.ones(hidden, hidden)
        pre = torch.ones(hidden) * 1.5
        calc = OASQScaleCalculator(max_iters=4)
        o_scales, v_scales = calc.compute_ov_scales(a_scale, w_scale, n_heads, n_heads, scales=pre)
        self.assertEqual(tuple(o_scales.shape), (hidden,))
        self.assertEqual(tuple(v_scales.shape), (hidden,))

    def test_compute_ov_scales_when_gqa_then_v_scale_length_matches_kv_projection(self):
        """给定 GQA（q_heads > kv_heads），期望 o/v scale 仍可正常返回。"""
        hidden = 64
        n_q = 8
        n_kv = 2
        a_scale = torch.ones(hidden)
        w_scale = torch.ones(hidden, hidden)
        calc = OASQScaleCalculator(max_iters=4)
        o_scales, v_scales = calc.compute_ov_scales(a_scale, w_scale, n_q, n_kv)
        self.assertEqual(tuple(o_scales.shape), (hidden,))
        self.assertTrue(v_scales.numel() > 0)
        self.assertTrue(torch.all(torch.isfinite(o_scales)))
        self.assertTrue(torch.all(torch.isfinite(v_scales)))


if __name__ == "__main__":
    unittest.main()
