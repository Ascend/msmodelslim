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

from msmodelslim.ir.api import calculate_qparam, fake_quantize, quantize
from msmodelslim.ir.qal import QScheme, QParam, QStorage, QDType

from msmodelslim.ir import mxfp4_per_block_sym
from msmodelslim.core.observer import MsMinMaxBlockObserver, MinMaxBlockObserverConfig
from msmodelslim.ir.utils import reshape_to_blocks, undo_reshape_to_blocks


class TestMXFP4Quantization(unittest.TestCase):
    """测试 calculate_mxfp4_qparam & mxfp4_quantize 函数"""

    def setUp(self):
        """设置测试环境"""
        # 创建QParam，使用mxfp4_per_block_sym scheme
        self.q_param = QParam(scheme=mxfp4_per_block_sym, ext={'axes': -1})

        self.x_scheme = self.q_param.scheme
        self.x_mx_finfo = self.q_param.scheme.dtype.mx_finfo
        self.x_axes = self.q_param.ext.get("axes")

        minmax_config = MinMaxBlockObserverConfig(axes=self.x_axes)
        self.x_minmax_block_observer = MsMinMaxBlockObserver(minmax_config)

    def test_calculate_qparam_should_return_correct_qparam(self):
        """测试 calculate_qparam 函数是否返回正确的 QParam"""
        min_val = torch.Tensor([[[1.0]]])
        max_val = torch.Tensor([[[10.9511]]])
        q_param = calculate_qparam(
            min_val,
            max_val,
            q_dtype=self.x_scheme.dtype,
            q_scope=self.x_scheme.scope,
            symmetric=self.x_scheme.symmetric,
        )
        self.assertIsInstance(q_param, QParam)
        self.assertIn('scale', q_param.ext)
        self.assertEqual(q_param.ext['scale'].shape, max_val.shape)
        self.assertEqual(q_param.ext['scale'], torch.tensor([[[1.0]]]))

    def test_quantize_with_positive_value_should_return_correct_result(self):
        """测试正数的量化"""

        x = torch.Tensor([0.6672])
        shared_exp = torch.Tensor([-2])
        x_q_param = QParam(
            scheme=QScheme(
                dtype=self.x_scheme.dtype,
                scope=self.x_scheme.scope,
                symmetric=self.x_scheme.symmetric,
            ),
            ext={
                "scale": shared_exp,
            },
        )
        x_q = quantize(QStorage(QDType.FLOAT, x), x_q_param)
        self.assertEqual(x_q.value, 3.0)

    def test_quantize_with_negative_value_should_return_correct_result(self):
        """测试负数的量化"""

        x = torch.Tensor([-2.7230])
        shared_exp = torch.Tensor([-1])
        x_q_param = QParam(
            scheme=QScheme(
                dtype=self.x_scheme.dtype,
                scope=self.x_scheme.scope,
                symmetric=self.x_scheme.symmetric,
            ),
            ext={
                "scale": shared_exp,
            },
        )
        x_q = quantize(QStorage(QDType.FLOAT, x), x_q_param)
        self.assertEqual(x_q.value, -6.0)

    def test_quantize_with_all_ones_should_return_correct_scale_and_quantized_value(self):
        """64 个元素全为 1：两 block，每块 max=1 → scale=-2；quantize 后恒为 4.0。"""
        x_min_val = torch.tensor([[[[1.0], [1.0]]]])  # (1, 1, 2, 1)
        x_max_val = torch.tensor([[[[1.0], [1.0]]]])
        x_q_param = calculate_qparam(
            x_min_val,
            x_max_val,
            q_dtype=self.x_scheme.dtype,
            q_scope=self.x_scheme.scope,
            symmetric=self.x_scheme.symmetric,
        )
        self.assertIsInstance(x_q_param, QParam)
        self.assertIn("scale", x_q_param.ext)
        self.assertEqual(x_q_param.ext["scale"].shape, x_max_val.shape)
        expected_scale = torch.tensor([[[[-2.0], [-2.0]]]])
        self.assertTrue(torch.equal(x_q_param.ext["scale"], expected_scale))

        x = torch.ones(1, 1, 2, 32)
        x_q = quantize(QStorage(QDType.FLOAT, x), x_q_param)
        self.assertTrue(torch.all(x_q.value == 4.0))

    def test_quantize_with_random_blocks_should_return_correct_scale_and_quantized_value(self):
        """64 个元素随机：与 fake_quantize 用例相同，用 observer 得到 min/max 再 quantize。"""
        # torch.manual_seed(0)
        # x = torch.randn(1, 1, 64) * 10 - 5
        x = torch.tensor(
            [
                [
                    [
                        -16.2584,
                        -16.5236,
                        -7.5058,
                        -9.3388,
                        3.4871,
                        1.9201,
                        -8.1601,
                        -26.1522,
                        -1.7773,
                        -17.6333,
                        -1.5002,
                        -1.9187,
                        -3.8016,
                        7.3766,
                        6.1678,
                        -7.4728,
                        -18.5265,
                        -21.9593,
                        0.6665,
                        2.9351,
                        0.9884,
                        -20.5510,
                        -8.4136,
                        13.5301,
                        2.5019,
                        -10.8550,
                        -6.7340,
                        -3.1652,
                        8.8937,
                        10.8633,
                        4.4630,
                        -13.4368,
                        -11.1358,
                        -4.6841,
                        -9.9268,
                        -2.5159,
                        -0.6030,
                        -3.8759,
                        1.4079,
                        -0.5884,
                        -6.0231,
                        2.9244,
                        -7.8967,
                        -4.4749,
                        0.2286,
                        18.0221,
                        -19.6889,
                        -20.8669,
                        -11.7309,
                        3.7283,
                        5.5536,
                        -3.2216,
                        -7.3034,
                        -8.9175,
                        0.4329,
                        -8.9516,
                        -9.4622,
                        2.4402,
                        10.2098,
                        29.1050,
                        -20.3118,
                        -17.3414,
                        13.1973,
                        -10.5153,
                    ]
                ]
            ]
        )
        axes = self.x_axes
        axes = [axes] if isinstance(axes, int) else axes
        axes = [a + x.ndim if a < 0 else a for a in axes]
        x_shaped, axes_, orig_shape, padded_shape = reshape_to_blocks(x, axes, self.x_mx_finfo.block_size)
        shared_exp_axes = [x + 1 for x in axes_] if self.x_mx_finfo.block_size > 0 else axes_
        self.x_minmax_block_observer.update(x_shaped, shared_exp_axes=shared_exp_axes)
        x_min_val, x_max_val = self.x_minmax_block_observer.get_min_max()
        x_q_param = calculate_qparam(
            x_min_val,
            x_max_val,
            q_dtype=self.x_scheme.dtype,
            q_scope=self.x_scheme.scope,
            symmetric=self.x_scheme.symmetric,
        )

        self.assertIsInstance(x_q_param, QParam)
        self.assertIn("scale", x_q_param.ext)
        self.assertEqual(x_q_param.ext['scale'].shape[-2], x_shaped.shape[-2])
        expected_scale = torch.tensor([[[[2.0], [3.0]]]])
        self.assertTrue(torch.equal(x_q_param.ext["scale"], expected_scale))

        x_q = quantize(QStorage(QDType.FLOAT, x_shaped), x_q_param)
        expected_x_q = torch.tensor(
            [
                [
                    [
                        [
                            -4.0,
                            -4.0,
                            -2.0,
                            -2.0,
                            1.0,
                            0.5,
                            -2.0,
                            -6.0,
                            -0.5,
                            -4.0,
                            -0.5,
                            -0.5,
                            -1.0,
                            2.0,
                            1.5,
                            -2.0,
                            -4.0,
                            -6.0,
                            0.0,
                            0.5,
                            0.0,
                            -6.0,
                            -2.0,
                            3.0,
                            0.5,
                            -3.0,
                            -1.5,
                            -1.0,
                            2.0,
                            3.0,
                            1.0,
                            -3.0,
                        ],
                        [
                            -1.5,
                            -0.5,
                            -1.0,
                            -0.5,
                            0.0,
                            -0.5,
                            0.0,
                            0.0,
                            -1.0,
                            0.5,
                            -1.0,
                            -0.5,
                            0.0,
                            2.0,
                            -2.0,
                            -3.0,
                            -1.5,
                            0.5,
                            0.5,
                            -0.5,
                            -1.0,
                            -1.0,
                            0.0,
                            -1.0,
                            -1.0,
                            0.5,
                            1.5,
                            4.0,
                            -3.0,
                            -2.0,
                            1.5,
                            -1.5,
                        ],
                    ]
                ]
            ]
        )
        self.assertTrue(torch.equal(x_q.value, expected_x_q))

    def test_fake_quantize_should_preserve_input_shape_and_dtype(self):
        """测试输出形状"""

        x = torch.randn(2, 64, 128)
        axes = self.x_axes
        axes = [axes] if isinstance(axes, int) else axes
        axes = [a + x.ndim if a < 0 else a for a in axes]
        x_shaped, axes_, orig_shape, padded_shape = reshape_to_blocks(x, axes, self.x_mx_finfo.block_size)
        shared_exp_axes = [x + 1 for x in axes_] if self.x_mx_finfo.block_size > 0 else axes_
        self.x_minmax_block_observer.update(x_shaped, shared_exp_axes=shared_exp_axes)
        x_min_val, x_max_val = self.x_minmax_block_observer.get_min_max()
        x_q_param = calculate_qparam(
            x_min_val,
            x_max_val,
            q_dtype=self.x_scheme.dtype,
            q_scope=self.x_scheme.scope,
            symmetric=self.x_scheme.symmetric,
        )
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x_shaped), x_q_param)
        x_q_dq.value = undo_reshape_to_blocks(x_q_dq.value, padded_shape, orig_shape, axes)

        self.assertEqual(x_q_param.ext['scale'].shape[-2], x_shaped.shape[-2])

        # 验证输出形状与输入相同
        self.assertEqual(x_q_dq.value.shape, x.shape)
        self.assertEqual(x_q_dq.value.dtype, x.dtype)

    def test_fake_quantize_should_produce_valid_values(self):
        """测试量化效果（量化后的值应该在合理范围内）"""
        x = torch.randn(2, 64, 128)
        axes = self.x_axes
        axes = [axes] if isinstance(axes, int) else axes
        axes = [a + x.ndim if a < 0 else a for a in axes]
        x_shaped, axes_, orig_shape, padded_shape = reshape_to_blocks(x, axes, self.x_mx_finfo.block_size)
        shared_exp_axes = [x + 1 for x in axes_] if self.x_mx_finfo.block_size > 0 else axes_
        self.x_minmax_block_observer.update(x_shaped, shared_exp_axes=shared_exp_axes)
        x_min_val, x_max_val = self.x_minmax_block_observer.get_min_max()
        x_q_param = calculate_qparam(
            x_min_val,
            x_max_val,
            q_dtype=self.x_scheme.dtype,
            q_scope=self.x_scheme.scope,
            symmetric=self.x_scheme.symmetric,
        )
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x_shaped), x_q_param)
        x_q_dq.value = undo_reshape_to_blocks(x_q_dq.value, padded_shape, orig_shape, axes)

        # 由于是量化，输出应该与输入不完全相同（除非输入恰好是量化值）
        # 但形状应该相同
        self.assertEqual(x_q_dq.value.shape, x.shape)
        # 量化后的值应该在合理范围内
        self.assertTrue(torch.isfinite(x_q_dq.value).all())

    def test_fake_quantize_with_single_element_tensor_should_preserve_shape_and_dtype(self):
        """测试边界情况：单个token"""
        x = torch.randn(1, 1, 32)  # 最小形状
        axes = self.x_axes
        axes = [axes] if isinstance(axes, int) else axes
        axes = [a + x.ndim if a < 0 else a for a in axes]
        x_shaped, axes_, orig_shape, padded_shape = reshape_to_blocks(x, axes, self.x_mx_finfo.block_size)
        shared_exp_axes = [x + 1 for x in axes_] if self.x_mx_finfo.block_size > 0 else axes_
        self.x_minmax_block_observer.update(x_shaped, shared_exp_axes=shared_exp_axes)
        x_min_val, x_max_val = self.x_minmax_block_observer.get_min_max()
        x_q_param = calculate_qparam(
            x_min_val,
            x_max_val,
            q_dtype=self.x_scheme.dtype,
            q_scope=self.x_scheme.scope,
            symmetric=self.x_scheme.symmetric,
        )
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x_shaped), x_q_param)
        x_q_dq.value = undo_reshape_to_blocks(x_q_dq.value, padded_shape, orig_shape, axes)

        # 验证输出形状与输入相同
        self.assertEqual(x_q_dq.value.shape, x.shape)
        self.assertEqual(x_q_dq.value.dtype, x.dtype)

    def test_fake_quantize_with_large_tensor_should_preserve_shape_and_dtype(self):
        """测试边界情况：大张量"""
        x = torch.randn(32, 512, 128)  # 较大的张量
        axes = self.x_axes
        axes = [axes] if isinstance(axes, int) else axes
        axes = [a + x.ndim if a < 0 else a for a in axes]
        x_shaped, axes_, orig_shape, padded_shape = reshape_to_blocks(x, axes, self.x_mx_finfo.block_size)
        shared_exp_axes = [x + 1 for x in axes_] if self.x_mx_finfo.block_size > 0 else axes_
        self.x_minmax_block_observer.update(x_shaped, shared_exp_axes=shared_exp_axes)
        x_min_val, x_max_val = self.x_minmax_block_observer.get_min_max()
        x_q_param = calculate_qparam(
            x_min_val,
            x_max_val,
            q_dtype=self.x_scheme.dtype,
            q_scope=self.x_scheme.scope,
            symmetric=self.x_scheme.symmetric,
        )
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x_shaped), x_q_param)
        x_q_dq.value = undo_reshape_to_blocks(x_q_dq.value, padded_shape, orig_shape, axes)

        # 验证输出形状与输入相同
        self.assertEqual(x_q_dq.value.shape, x.shape)
        self.assertEqual(x_q_dq.value.dtype, x.dtype)


if __name__ == '__main__':
    unittest.main()
