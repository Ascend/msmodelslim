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
# pylint: disable=consider-using-from-import

import unittest
import torch
import torch.nn as nn

from msmodelslim.ir.flatquant import (
    FlatQuantOnlineWrapper,
    FlatQuantOnlineHookIR,
)


class TestFlatQuantOnlineWrapper(unittest.TestCase):
    """Unit tests for FlatQuantOnlineWrapper."""

    def setUp(self):
        self.wrapped_module = nn.Linear(32, 16)

    def test_FlatQuantOnlineWrapper_wrapped_module_equals_input_after_init(self):
        """FlatQuantOnlineWrapper-初始化后-wrapped_module与传入模块一致"""
        wrapper = FlatQuantOnlineWrapper(self.wrapped_module)
        self.assertIs(wrapper.wrapped_module, self.wrapped_module)

    def test_FlatQuantOnlineWrapper_clip_factor_is_none_after_init(self):
        """FlatQuantOnlineWrapper-初始化后-clip_factor为None"""
        wrapper = FlatQuantOnlineWrapper(self.wrapped_module)
        self.assertIsNone(wrapper.clip_factor)

    def test_FlatQuantOnlineWrapper_trans_is_none_after_init(self):
        """FlatQuantOnlineWrapper-初始化后-变换矩阵为None"""
        wrapper = FlatQuantOnlineWrapper(self.wrapped_module)
        self.assertIsNone(wrapper.left_trans)
        self.assertIsNone(wrapper.right_trans)
        self.assertIsNone(wrapper.save_trans)

    def test_FlatQuantOnlineWrapper_returns_true_when_is_atomic(self):
        """FlatQuantOnlineWrapper-调用is_atomic-返回True"""
        self.assertTrue(FlatQuantOnlineWrapper.is_atomic())

    def test_FlatQuantOnlineWrapper_clip_factor_is_parameter_after_add_clip(self):
        """FlatQuantOnlineWrapper-调用_add_clip后-clip_factor为Parameter"""
        wrapper = FlatQuantOnlineWrapper(self.wrapped_module)
        clip_factor = torch.randn(32)
        wrapper._add_clip(clip_factor)
        self.assertIsNotNone(wrapper.clip_factor)
        self.assertIsInstance(wrapper.clip_factor, nn.Parameter)
        self.assertFalse(wrapper.clip_factor.requires_grad)

    def test_FlatQuantOnlineWrapper_left_trans_is_parameter_after_add_flat_with_left_trans(self):
        """FlatQuantOnlineWrapper-调用_add_flat传入left_trans后-left_trans为Parameter"""
        wrapper = FlatQuantOnlineWrapper(self.wrapped_module)
        left_trans = torch.randn(4, 4)
        save_trans = {"left_trans": left_trans}
        wrapper._add_flat(save_trans)
        self.assertIsNotNone(wrapper.left_trans)
        self.assertIsInstance(wrapper.left_trans, nn.Parameter)
        self.assertFalse(wrapper.left_trans.requires_grad)

    def test_FlatQuantOnlineWrapper_right_trans_is_parameter_after_add_flat_with_right_trans(self):
        """FlatQuantOnlineWrapper-调用_add_flat传入right_trans后-right_trans为Parameter"""
        wrapper = FlatQuantOnlineWrapper(self.wrapped_module)
        right_trans = torch.randn(8, 8)
        save_trans = {"right_trans": right_trans}
        wrapper._add_flat(save_trans)
        self.assertIsNotNone(wrapper.right_trans)
        self.assertIsInstance(wrapper.right_trans, nn.Parameter)
        self.assertFalse(wrapper.right_trans.requires_grad)

    def test_FlatQuantOnlineWrapper_both_trans_not_none_after_add_flat_with_both(self):
        """FlatQuantOnlineWrapper-调用_add_flat传入左右变换后-两者均不为None"""
        wrapper = FlatQuantOnlineWrapper(self.wrapped_module)
        left_trans = torch.randn(4, 4)
        right_trans = torch.randn(8, 8)
        save_trans = {"left_trans": left_trans, "right_trans": right_trans}
        wrapper._add_flat(save_trans)
        self.assertIsNotNone(wrapper.left_trans)
        self.assertIsNotNone(wrapper.right_trans)

    def test_FlatQuantOnlineWrapper_trans_remain_none_after_add_flat_with_none(self):
        """FlatQuantOnlineWrapper-调用_add_flat传入None后-变换矩阵仍为None"""
        wrapper = FlatQuantOnlineWrapper(self.wrapped_module)
        wrapper._add_flat(None)
        self.assertIsNone(wrapper.left_trans)
        self.assertIsNone(wrapper.right_trans)

    def test_FlatQuantOnlineWrapper_output_shape_equals_input_when_apply_clip(self):
        """FlatQuantOnlineWrapper-调用_apply_clip后-输出形状与输入一致"""
        wrapper = FlatQuantOnlineWrapper(self.wrapped_module)
        clip_factor = torch.ones(32) * 0.5
        wrapper._add_clip(clip_factor)
        x = torch.randn(2, 32)
        clipped = wrapper._apply_clip(x)
        self.assertEqual(clipped.shape, x.shape)

    def test_FlatQuantOnlineWrapper_output_shape_equals_input_when_apply_flat_with_right_trans(self):
        """FlatQuantOnlineWrapper-调用_apply_flat带right_trans后-输出形状与输入一致"""
        wrapper = FlatQuantOnlineWrapper(self.wrapped_module)
        right_trans = torch.eye(8)
        save_trans = {"right_trans": right_trans}
        wrapper._add_flat(save_trans)
        x = torch.randn(2, 8)
        transformed = wrapper._apply_flat(x)
        self.assertEqual(transformed.shape, x.shape)

    def test_FlatQuantOnlineWrapper_output_shape_equals_input_when_apply_flat_with_left_trans(self):
        """FlatQuantOnlineWrapper-调用_apply_flat带left_trans后-输出形状与输入一致"""
        wrapper = FlatQuantOnlineWrapper(self.wrapped_module)
        left_trans = torch.eye(4)
        save_trans = {"left_trans": left_trans}
        wrapper._add_flat(save_trans)
        x = torch.randn(2, 32)
        transformed = wrapper._apply_flat(x)
        self.assertEqual(transformed.shape, x.shape)

    def test_FlatQuantOnlineWrapper_output_shape_correct_when_forward_without_clip_and_trans(self):
        """FlatQuantOnlineWrapper-调用forward无裁剪和变换-输出形状正确"""
        wrapper = FlatQuantOnlineWrapper(self.wrapped_module)
        x = torch.randn(2, 32)
        output = wrapper(x)
        self.assertEqual(output.shape, (2, 16))

    def test_FlatQuantOnlineWrapper_output_shape_correct_when_forward_with_clip(self):
        """FlatQuantOnlineWrapper-调用forward带裁剪-输出形状正确"""
        wrapper = FlatQuantOnlineWrapper(self.wrapped_module)
        clip_factor = torch.ones(32) * 0.5
        wrapper._add_clip(clip_factor)
        x = torch.randn(2, 32)
        output = wrapper(x)
        self.assertEqual(output.shape, (2, 16))

    def test_FlatQuantOnlineWrapper_output_shape_correct_when_forward_with_trans(self):
        """FlatQuantOnlineWrapper-调用forward带变换-输出形状正确"""
        wrapper = FlatQuantOnlineWrapper(self.wrapped_module)
        right_trans = torch.eye(8)
        save_trans = {"right_trans": right_trans}
        wrapper._add_flat(save_trans)
        x = torch.randn(2, 32)
        output = wrapper(x)
        self.assertEqual(output.shape, (2, 16))

    def test_FlatQuantOnlineWrapper_output_shape_correct_when_forward_with_clip_and_trans(self):
        """FlatQuantOnlineWrapper-调用forward带裁剪和变换-输出形状正确"""
        wrapper = FlatQuantOnlineWrapper(self.wrapped_module)
        clip_factor = torch.ones(32) * 0.5
        wrapper._add_clip(clip_factor)
        right_trans = torch.eye(8)
        save_trans = {"right_trans": right_trans}
        wrapper._add_flat(save_trans)
        x = torch.randn(2, 32)
        output = wrapper(x)
        self.assertEqual(output.shape, (2, 16))

    def test_FlatQuantOnlineWrapper_returns_instance_when_create(self):
        """FlatQuantOnlineWrapper-调用create-返回正确实例"""
        wrapper = FlatQuantOnlineWrapper.create(self.wrapped_module)
        self.assertIsInstance(wrapper, FlatQuantOnlineWrapper)
        self.assertIs(wrapper.wrapped_module, self.wrapped_module)

    def test_FlatQuantOnlineWrapper_extra_repr_contains_kronecker_when_trans_exists(self):
        """FlatQuantOnlineWrapper-调用extra_repr带变换-包含kronecker_rotation"""
        wrapper = FlatQuantOnlineWrapper(self.wrapped_module)
        left_trans = torch.randn(4, 4)
        right_trans = torch.randn(8, 8)
        save_trans = {"left_trans": left_trans, "right_trans": right_trans}
        wrapper._add_flat(save_trans)
        repr_str = wrapper.extra_repr()
        self.assertIn("kronecker_rotation", repr_str)

    def test_FlatQuantOnlineWrapper_extra_repr_contains_no_affine_when_trans_not_exists(self):
        """FlatQuantOnlineWrapper-调用extra_repr无变换-包含No affine transformation"""
        wrapper = FlatQuantOnlineWrapper(self.wrapped_module)
        repr_str = wrapper.extra_repr()
        self.assertIn("No affine transformation", repr_str)


class TestFlatQuantOnlineHookIR(unittest.TestCase):
    """Unit tests for FlatQuantOnlineHookIR."""

    def setUp(self):
        self.clip_factor = torch.ones(1) * 0.5
        self.left_trans = torch.eye(4)
        self.right_trans = torch.eye(8)
        self.save_trans = {"left_trans": self.left_trans, "right_trans": self.right_trans}

    def test_FlatQuantOnlineHookIR_clip_factor_equals_input_after_init(self):
        """FlatQuantOnlineHookIR-初始化后-clip_factor与传入一致"""
        hook_ir = FlatQuantOnlineHookIR(self.clip_factor, self.save_trans)
        self.assertTrue(torch.equal(hook_ir.clip_factor, self.clip_factor))

    def test_FlatQuantOnlineHookIR_save_trans_equals_input_after_init(self):
        """FlatQuantOnlineHookIR-初始化后-save_trans与传入一致"""
        hook_ir = FlatQuantOnlineHookIR(self.clip_factor, self.save_trans)
        self.assertEqual(hook_ir.save_trans, self.save_trans)

    def test_FlatQuantOnlineHookIR_clip_factor_is_none_after_init_with_none_clip(self):
        """FlatQuantOnlineHookIR-初始化传入None clip-clip_factor为None"""
        hook_ir = FlatQuantOnlineHookIR(None, self.save_trans)
        self.assertIsNone(hook_ir.clip_factor)

    def test_FlatQuantOnlineHookIR_save_trans_is_none_after_init_with_none_trans(self):
        """FlatQuantOnlineHookIR-初始化传入None trans-save_trans为None"""
        hook_ir = FlatQuantOnlineHookIR(self.clip_factor, None)
        self.assertIsNone(hook_ir.save_trans)

    def test_FlatQuantOnlineHookIR_returns_first_arg_when_call(self):
        """FlatQuantOnlineHookIR-调用__call__-返回第一个参数"""
        hook_ir = FlatQuantOnlineHookIR(self.clip_factor, self.save_trans)
        module = nn.Linear(32, 16)
        x = torch.randn(2, 32)
        result = hook_ir(module, (x,))
        self.assertTrue(torch.equal(result, x))

    def test_FlatQuantOnlineHookIR_returns_first_arg_when_call_with_multiple_args(self):
        """FlatQuantOnlineHookIR-调用__call__多参数-返回第一个参数"""
        hook_ir = FlatQuantOnlineHookIR(self.clip_factor, self.save_trans)
        module = nn.Linear(32, 16)
        x = torch.randn(2, 32)
        y = torch.randn(2, 32)
        result = hook_ir(module, (x, y))
        self.assertTrue(torch.equal(result, x))

    def test_FlatQuantOnlineHookIR_returns_FlatQuantOnlineWrapper_when_wrapper_module(self):
        """FlatQuantOnlineHookIR-调用wrapper_module-返回FlatQuantOnlineWrapper"""
        hook_ir = FlatQuantOnlineHookIR(self.clip_factor, self.save_trans)
        module = nn.Linear(32, 16)
        wrapper = hook_ir.wrapper_module(module)
        self.assertIsInstance(wrapper, FlatQuantOnlineWrapper)
        self.assertIs(wrapper.wrapped_module, module)

    def test_FlatQuantOnlineHookIR_clip_factor_is_none_when_wrapper_module_without_clip(self):
        """FlatQuantOnlineHookIR-调用wrapper_module无裁剪-clip_factor为None"""
        hook_ir = FlatQuantOnlineHookIR(None, self.save_trans)
        module = nn.Linear(32, 16)
        wrapper = hook_ir.wrapper_module(module)
        self.assertIsNone(wrapper.clip_factor)

    def test_FlatQuantOnlineHookIR_clip_factor_not_none_when_wrapper_module_with_clip(self):
        """FlatQuantOnlineHookIR-调用wrapper_module带裁剪-clip_factor不为None"""
        hook_ir = FlatQuantOnlineHookIR(self.clip_factor, self.save_trans)
        module = nn.Linear(32, 16)
        wrapper = hook_ir.wrapper_module(module)
        self.assertIsNotNone(wrapper.clip_factor)

    def test_FlatQuantOnlineHookIR_trans_not_none_when_wrapper_module(self):
        """FlatQuantOnlineHookIR-调用wrapper_module-变换矩阵不为None"""
        hook_ir = FlatQuantOnlineHookIR(self.clip_factor, self.save_trans)
        module = nn.Linear(32, 16)
        wrapper = hook_ir.wrapper_module(module)
        self.assertIsNotNone(wrapper.left_trans)
        self.assertIsNotNone(wrapper.right_trans)


if __name__ == '__main__':
    unittest.main()
