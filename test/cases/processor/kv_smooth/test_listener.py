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
# pylint: disable=no-name-in-module

import unittest
from unittest.mock import Mock

import pytest
import torch
from transformers import Cache

from msmodelslim.processor.kv_smooth.listener import KVCacheListener, KVCacheListenerManager
from msmodelslim.utils.exception import SpecError


class MockCache(Cache):
    """模拟的 Cache 类用于测试"""

    def __init__(self):
        super().__init__()
        self.update_called = False
        self.update_args = None
        self.custom_method_called = False
        self.custom_attribute = "test_value"
        self.max_cache_shape = 1024

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self.update_called = True
        self.update_args = (key_states, value_states, layer_idx, cache_kwargs)
        return key_states, value_states

    def get_max_cache_shape(self):
        """返回最大缓存形状"""
        return self.max_cache_shape

    def get_seq_length(self, layer_idx=None):
        """返回序列长度"""
        return 10

    def reorder_cache(self, beam_idx):
        """重新排序缓存"""
        # 模拟实现，实际不做任何操作
        pass

    def custom_method(self):
        self.custom_method_called = True
        return "custom_result"


class TestKVCacheListener(unittest.TestCase):
    """KVCacheListener 单元测试"""

    def setUp(self):
        """测试前的准备工作"""
        self.listen_helper_called = False
        self.listen_helper_args = None

        def mock_listen_helper(layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor):
            self.listen_helper_called = True
            self.listen_helper_args = (layer_idx, key_states, value_states)

        self.listen_helper = mock_listen_helper
        self.mock_cache = MockCache()

        # 创建测试用的张量
        self.key_states = torch.randn(2, 8, 10, 16)
        self.value_states = torch.randn(2, 8, 10, 16)
        self.layer_idx = 0

    def tearDown(self):
        """测试后的清理工作"""
        self.listen_helper_called = False
        self.listen_helper_args = None

    def test_init_when_cache_provided_then_initialize_successfully(self):
        """当提供cache时，应成功初始化KVCacheListener"""
        listener = KVCacheListener(self.listen_helper, cache=self.mock_cache)

        self.assertEqual(listener.cache, self.mock_cache)
        self.assertEqual(listener.listen_helper, self.listen_helper)

    def test_init_when_no_cache_provided_then_raise_spec_error(self):
        """当未提供cache时，应抛出SpecError异常"""
        with self.assertRaises(SpecError) as context:
            KVCacheListener(self.listen_helper, cache=None)

        error_message = str(context.exception)
        self.assertIn("Cache cannot be None", error_message)
        self.assertIn("Please provide a valid Cache instance", error_message)

    @unittest.skip("calibration do NOT use KVCache, update makes NO change")
    def test_update_when_cache_exists_then_call_helper_and_cache_update(self):
        """当存在cache时，应调用listen_helper和cache.update方法"""
        listener = KVCacheListener(self.listen_helper, cache=self.mock_cache)

        result_key, result_value = listener.update(self.key_states, self.value_states, self.layer_idx)

        # 验证 listen_helper 被调用
        self.assertTrue(self.listen_helper_called)
        self.assertEqual(self.listen_helper_args[0], self.layer_idx)
        self.assertTrue(torch.equal(self.listen_helper_args[1], self.key_states))
        self.assertTrue(torch.equal(self.listen_helper_args[2], self.value_states))

        # 验证 cache.update 被调用
        self.assertTrue(self.mock_cache.update_called)
        self.assertTrue(torch.equal(self.mock_cache.update_args[0], self.key_states))
        self.assertTrue(torch.equal(self.mock_cache.update_args[1], self.value_states))
        self.assertEqual(self.mock_cache.update_args[2], self.layer_idx)

        # 验证返回值
        self.assertTrue(torch.equal(result_key, self.key_states))
        self.assertTrue(torch.equal(result_value, self.value_states))

    @unittest.skip("calibration do NOT use KVCache, update makes NO change")
    def test_update_when_cache_kwargs_provided_then_pass_to_cache(self):
        """当提供cache_kwargs时，应正确传递给cache"""
        cache_kwargs = {"test_param": "test_value"}
        listener = KVCacheListener(self.listen_helper, cache=self.mock_cache)

        listener.update(self.key_states, self.value_states, self.layer_idx, cache_kwargs)

        # 验证 cache_kwargs 被正确传递
        self.assertEqual(self.mock_cache.update_args[3], cache_kwargs)

    def test_getattr_when_cache_exists_then_delegate_to_cache(self):
        """当存在cache时，应将属性访问委托给cache"""
        listener = KVCacheListener(self.listen_helper, cache=self.mock_cache)

        # 测试访问属性
        self.assertEqual(listener.custom_attribute, "test_value")

        # 测试调用方法
        result = listener.custom_method()
        self.assertEqual(result, "custom_result")
        self.assertTrue(self.mock_cache.custom_method_called)

    def test_getattr_when_calling_cache_method_then_execute_successfully(self):
        """当调用cache方法时，应成功执行并返回结果"""
        listener = KVCacheListener(self.listen_helper, cache=self.mock_cache)

        # 调用 cache 的方法
        result = listener.custom_method()

        self.assertEqual(result, "custom_result")
        self.assertTrue(self.mock_cache.custom_method_called)

    @pytest.mark.skip
    def test_inheritance_when_creating_listener_then_inherit_from_cache(self):
        """当创建listener时，应正确继承自Cache类"""
        listener = KVCacheListener(self.listen_helper, cache=self.mock_cache)

        # 验证是 Cache 的实例
        self.assertIsInstance(listener, Cache)

        # 验证可以访问 Cache 的属性和方法
        self.assertTrue(hasattr(listener, 'get_seq_length'))


class TestKVCacheListenerManager(unittest.TestCase):
    """KVCacheListenerManager 单元测试"""

    def setUp(self):
        """测试前的准备工作"""
        self.manager = KVCacheListenerManager()
        self.listen_helper_called = False

        def mock_listen_helper(layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor):
            self.listen_helper_called = True

        self.listen_helper = mock_listen_helper
        self.mock_module = Mock()

    def tearDown(self):
        """测试后的清理工作"""
        self.manager.remove_listeners()
        self.listen_helper_called = False

    def test_init_when_creating_manager_then_initialize_empty_handlers(self):
        """当创建manager时，应初始化空的remove_handlers列表"""
        manager = KVCacheListenerManager()
        self.assertEqual(len(manager.remove_handlers), 0)

    def test_attach_listener_when_past_key_values_exists_then_replace_with_listener(self):
        """当存在past_key_values时，应替换为KVCacheListener"""
        # 模拟模块的前向传播参数
        mock_cache = MockCache()
        args = ()
        kwargs = {"past_key_values": mock_cache}

        # 注册 hook
        self.manager.attach_listener_to_module(self.mock_module, self.listen_helper)

        # 验证 hook 被注册
        self.assertEqual(len(self.manager.remove_handlers), 1)

        # 模拟调用 hook
        hook = self.mock_module.register_forward_pre_hook.call_args[0][0]
        new_args, new_kwargs = hook(self.mock_module, args, kwargs)

        # 验证 past_key_values 被替换为 KVCacheListener
        self.assertIsInstance(new_kwargs["past_key_values"], KVCacheListener)
        self.assertEqual(new_kwargs["past_key_values"].cache, mock_cache)

    def test_attach_listener_when_past_key_value_exists_then_replace_with_listener(self):
        """当存在past_key_value时，应替换为KVCacheListener"""
        # 模拟模块的前向传播参数
        mock_cache = MockCache()
        args = ()
        kwargs = {"past_key_value": mock_cache}

        # 注册 hook
        self.manager.attach_listener_to_module(self.mock_module, self.listen_helper)

        # 模拟调用 hook
        hook = self.mock_module.register_forward_pre_hook.call_args[0][0]
        new_args, new_kwargs = hook(self.mock_module, args, kwargs)

        # 验证 past_key_value 被替换为 KVCacheListener
        self.assertIsInstance(new_kwargs["past_key_value"], KVCacheListener)
        self.assertEqual(new_kwargs["past_key_value"].cache, mock_cache)

    def test_attach_listener_when_no_cache_then_raise_spec_error(self):
        """当不存在cache时，应抛出SpecError异常"""
        # 模拟模块的前向传播参数（没有 cache）
        args = ()
        kwargs = {}

        # 注册 hook
        self.manager.attach_listener_to_module(self.mock_module, self.listen_helper)

        # 模拟调用 hook
        hook = self.mock_module.register_forward_pre_hook.call_args[0][0]
        with self.assertRaises(SpecError) as context:
            _ = hook(self.mock_module, args, kwargs)
        error_message = str(context.exception)
        self.assertIn("both are None or missing", error_message)
        self.assertIn("requires a valid Cache", error_message)

    def test_remove_listeners_when_called_then_clear_all_handlers(self):
        """当调用remove_listeners时，应清除所有handler"""
        # 注册多个 hook
        self.manager.attach_listener_to_module(self.mock_module, self.listen_helper)
        self.manager.attach_listener_to_module(self.mock_module, self.listen_helper)

        # 验证有多个 handler（由于是同一个 mock_module，可能只注册了一个）
        self.assertGreaterEqual(len(self.manager.remove_handlers), 1)

        # 移除所有监听器
        self.manager.remove_listeners()

        # 验证所有 handler 都被移除
        self.assertEqual(len(self.manager.remove_handlers), 0)


if __name__ == '__main__':
    unittest.main()
