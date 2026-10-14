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

from msmodelslim.utils.memory import (
    align_input_to_module_device_hook,
    register_device_alignment_hook,
    unregister_device_alignment_hook,
)


class MockModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


class TestDeviceAlignmentHook(unittest.TestCase):
    def setUp(self):
        self.module = MockModule()
        self.device = torch.device('cpu')
        self.module.to(self.device)

    def test_align_input_to_module_device_hook(self):
        input_tensor = torch.randn(5, 10).to(torch.device('cpu'))

        args = (input_tensor,)
        kwargs = {}

        aligned_args, aligned_kwargs = align_input_to_module_device_hook(self.module, args, kwargs)

        self.assertEqual(aligned_args[0].device, self.module.linear.weight.device)

    def test_register_device_alignment_hook(self):
        # 注册 hook
        hook_handle = register_device_alignment_hook(self.module)

        # 验证 hook 已被注册
        self.assertTrue(hasattr(self.module, '_device_alignment_hooks_registered'))
        self.assertTrue(self.module._device_alignment_hooks_registered)
        self.assertTrue(hasattr(self.module, '_device_alignment_pre_hook_handle'))
        self.assertIsNotNone(hook_handle)
        self.assertIsInstance(hook_handle, dict)
        self.assertIn('pre_hook', hook_handle)

    def test_register_device_alignment_hook_with_name(self):
        # 注册 hook 并指定名称
        custom_name = "MockModule"
        hook_handle = register_device_alignment_hook(self.module, name=custom_name)

        # 验证 hook 已被注册
        self.assertTrue(hasattr(self.module, '_device_alignment_hooks_registered'))
        self.assertTrue(self.module._device_alignment_hooks_registered)
        self.assertTrue(hasattr(self.module, '_device_alignment_pre_hook_handle'))
        self.assertIsNotNone(hook_handle)
        self.assertIsInstance(hook_handle, dict)
        self.assertIn('pre_hook', hook_handle)

        # 清理
        unregister_device_alignment_hook(self.module, name=custom_name)

    def test_unregister_device_alignment_hook(self):
        # 先注册 hook
        register_device_alignment_hook(self.module)

        # 验证 hook 已被注册
        self.assertTrue(hasattr(self.module, '_device_alignment_hooks_registered'))

        # 移除 hook
        unregister_device_alignment_hook(self.module)

        # 验证 hook 已被移除
        self.assertFalse(hasattr(self.module, '_device_alignment_hooks_registered'))
        self.assertFalse(hasattr(self.module, '_device_alignment_pre_hook_handle'))

    def test_unregister_device_alignment_hook_with_name(self):
        # 先注册 hook 并指定名称
        custom_name = "MockModule"
        register_device_alignment_hook(self.module, name=custom_name)

        # 验证 hook 已被注册
        self.assertTrue(hasattr(self.module, '_device_alignment_hooks_registered'))

        # 移除 hook 并指定名称
        unregister_device_alignment_hook(self.module, name=custom_name)

        # 验证 hook 已被移除
        self.assertFalse(hasattr(self.module, '_device_alignment_hooks_registered'))
        self.assertFalse(hasattr(self.module, '_device_alignment_pre_hook_handle'))

    def test_hook_prevents_duplicate_registration(self):
        # 第一次注册
        handle1 = register_device_alignment_hook(self.module)

        # 第二次注册（应该返回同一个 handle）
        handle2 = register_device_alignment_hook(self.module)

        # 验证返回的是同一个 handle
        self.assertEqual(handle1, handle2)

        # 清理
        unregister_device_alignment_hook(self.module)

    def test_hook_prevents_duplicate_registration_with_name(self):
        custom_name = "MockModule"

        # 第一次注册
        handle1 = register_device_alignment_hook(self.module, name=custom_name)

        # 第二次注册（应该返回同一个 handle）
        handle2 = register_device_alignment_hook(self.module, name=custom_name)

        # 验证返回的是同一个 handle
        self.assertEqual(handle1, handle2)

        # 清理
        unregister_device_alignment_hook(self.module, name=custom_name)

    def test_hook_with_complex_input(self):
        # 创建复杂的输入结构
        input_tensor1 = torch.randn(5, 10).to(torch.device('cpu'))
        input_tensor2 = torch.randn(5, 10).to(torch.device('cpu'))

        complex_input = {
            'tensor1': input_tensor1,
            'tensor2': input_tensor2,
            'list_data': [input_tensor1, input_tensor2],
            'tuple_data': (input_tensor1, input_tensor2),
        }

        # 模拟 hook 的输入格式
        args = (complex_input,)
        kwargs = {}

        # 调用 hook 函数
        aligned_args, aligned_kwargs = align_input_to_module_device_hook(self.module, args, kwargs)

        # 验证所有张量都被对齐到模块设备
        aligned_complex_input = aligned_args[0]
        self.assertEqual(aligned_complex_input['tensor1'].device, self.module.linear.weight.device)
        self.assertEqual(aligned_complex_input['tensor2'].device, self.module.linear.weight.device)
        self.assertEqual(aligned_complex_input['list_data'][0].device, self.module.linear.weight.device)
        self.assertEqual(aligned_complex_input['list_data'][1].device, self.module.linear.weight.device)
        self.assertEqual(aligned_complex_input['tuple_data'][0].device, self.module.linear.weight.device)
        self.assertEqual(aligned_complex_input['tuple_data'][1].device, self.module.linear.weight.device)

    def test_hook_with_no_parameters_module(self):
        # 创建一个没有参数的模块
        class NoParamModule(nn.Module):
            def forward(self, x):
                return x

        no_param_module = NoParamModule()

        # 创建输入
        input_tensor = torch.randn(5, 10)
        args = (input_tensor,)
        kwargs = {}

        # 调用 hook 函数（应该返回原始输入）
        aligned_args, aligned_kwargs = align_input_to_module_device_hook(no_param_module, args, kwargs)

        # 验证返回的是原始输入
        self.assertEqual(aligned_args, args)
        self.assertEqual(aligned_kwargs, kwargs)

    def test_hook_with_none_input(self):
        # 测试 None 模块
        args = (torch.randn(5, 10),)
        kwargs = {}
        result_args, result_kwargs = align_input_to_module_device_hook(None, args, kwargs)
        self.assertEqual(result_args, args)
        self.assertEqual(result_kwargs, kwargs)

    def test_register_hook_with_none_module(self):
        result = register_device_alignment_hook(None)
        self.assertIsNone(result)

    def test_hook_data_statistics(self):
        # 创建不同设备上的张量
        cpu_tensor1 = torch.randn(100, 100)  # 40000 bytes (float32)
        cpu_tensor2 = torch.randn(50, 50)  # 10000 bytes (float32)
        cpu_tensor3 = torch.randn(10, 10)  # 400 bytes (float32)

        # 确保张量在 CPU 上
        cpu_tensor1 = cpu_tensor1.cpu()
        cpu_tensor2 = cpu_tensor2.cpu()
        cpu_tensor3 = cpu_tensor3.cpu()

        # 创建复杂输入结构
        complex_input = {
            'tensor1': cpu_tensor1,
            'tensor2': cpu_tensor2,
            'list_data': [cpu_tensor3],
            'already_on_device': torch.randn(5, 5).to(self.module.linear.weight.device),
        }

        # 模拟 hook 的输入格式
        args = (complex_input,)
        kwargs = {}

        # 调用 hook 函数
        aligned_args, aligned_kwargs = align_input_to_module_device_hook(self.module, args, kwargs)

        # 验证所有张量都被对齐到模块设备
        aligned_complex_input = aligned_args[0]
        self.assertEqual(aligned_complex_input['tensor1'].device, self.module.linear.weight.device)
        self.assertEqual(aligned_complex_input['tensor2'].device, self.module.linear.weight.device)
        self.assertEqual(aligned_complex_input['list_data'][0].device, self.module.linear.weight.device)
        self.assertEqual(aligned_complex_input['already_on_device'].device, self.module.linear.weight.device)

    def test_hook_no_movement_statistics(self):
        # 创建已经在正确设备上的张量
        correct_device_tensor = torch.randn(10, 10).to(self.module.linear.weight.device)

        args = (correct_device_tensor,)
        kwargs = {}

        # 调用 hook 函数
        aligned_args, aligned_kwargs = align_input_to_module_device_hook(self.module, args, kwargs)

        # 验证张量没有被移动
        self.assertEqual(aligned_args[0].device, self.module.linear.weight.device)

    def test_hook_large_tensor_statistics(self):
        # 创建一个大张量（超过 1MB）
        large_tensor = torch.randn(1000, 1000)  # 4MB (float32)
        large_tensor = large_tensor.cpu()

        args = (large_tensor,)
        kwargs = {}

        # 调用 hook 函数
        aligned_args, aligned_kwargs = align_input_to_module_device_hook(self.module, args, kwargs)

        # 验证张量被正确移动
        self.assertEqual(aligned_args[0].device, self.module.linear.weight.device)

    def test_hook_with_kwargs(self):
        # 注册带 kwargs 的 hook
        hook_handle = register_device_alignment_hook(self.module, with_kwargs=True)

        # 验证 hook 已被注册
        self.assertTrue(hasattr(self.module, '_device_alignment_hooks_registered'))
        self.assertIsNotNone(hook_handle)
        self.assertIsInstance(hook_handle, dict)
        self.assertIn('pre_hook', hook_handle)

        # 清理
        unregister_device_alignment_hook(self.module)

    def test_hook_with_mixed_input_types(self):
        # 创建混合类型的输入
        tensor_input = torch.randn(5, 10).cpu()
        string_input = "test_string"
        int_input = 42
        list_input = [tensor_input, string_input, int_input]

        args = (tensor_input, string_input, int_input, list_input)
        kwargs = {'tensor': tensor_input, 'string': string_input}

        # 调用 hook 函数
        aligned_args, aligned_kwargs = align_input_to_module_device_hook(self.module, args, kwargs)

        # 验证张量被对齐，其他类型保持不变
        self.assertEqual(aligned_args[0].device, self.module.linear.weight.device)
        self.assertEqual(aligned_args[1], string_input)
        self.assertEqual(aligned_args[2], int_input)
        self.assertEqual(aligned_args[3][0].device, self.module.linear.weight.device)
        self.assertEqual(aligned_args[3][1], string_input)
        self.assertEqual(aligned_args[3][2], int_input)
        self.assertEqual(aligned_kwargs['tensor'].device, self.module.linear.weight.device)
        self.assertEqual(aligned_kwargs['string'], string_input)


if __name__ == '__main__':
    unittest.main()
