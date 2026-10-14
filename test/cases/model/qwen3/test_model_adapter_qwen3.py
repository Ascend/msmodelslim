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
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch.nn as nn

from msmodelslim.core.const import DeviceType
from msmodelslim.model.qwen3.model_adapter import Qwen3ModelAdapter
from msmodelslim.utils.exception import InvalidModelError


class DummyConfig:
    """模拟配置对象"""

    def __init__(self):
        self.head_dim = 64
        self.hidden_size = 128
        self.num_attention_heads = 8
        self.num_key_value_heads = 4


class TestQwen3ModelAdapterLoadModel(unittest.TestCase):
    """测试Qwen3ModelAdapter的load_model方法"""

    def setUp(self):
        """测试前的准备工作"""
        self.model_type = 'Qwen3-8B'
        self.model_path = Path('.')

    def test_load_model_with_npu_device_when_called_then_delegate_to_load_model(self):
        """测试load_model方法：使用NPU设备时应委托给_load_model方法"""
        with patch('msmodelslim.model.qwen3.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen3ModelAdapter(model_type=self.model_type, model_path=self.model_path)

            mock_model = nn.Linear(10, 10)
            adapter._load_model = MagicMock(return_value=mock_model)

            result = adapter.load_model(device=DeviceType.NPU)

            # 验证返回的是模型
            self.assertIs(result, mock_model)
            # 验证_load_model被正确调用
            adapter._load_model.assert_called_once_with(DeviceType.NPU)


class TestQwen3ModelAdapterGetHeadDim(unittest.TestCase):
    """测试Qwen3ModelAdapter的get_head_dim方法"""

    def setUp(self):
        """测试前的准备工作"""
        self.model_type = 'Qwen3-8B'
        self.model_path = Path('.')

    def test_get_head_dim_with_head_dim_in_config_when_called_then_return_head_dim(self):
        """测试get_head_dim方法：config中有head_dim时应直接返回"""
        with patch('msmodelslim.model.qwen3.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen3ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = DummyConfig()
            adapter.config.head_dim = 64

            result = adapter.get_head_dim()

            # 验证返回config中的head_dim
            self.assertEqual(result, 64)

    def test_get_head_dim_without_head_dim_when_called_then_calculate_from_hidden_size(self):
        """测试get_head_dim方法：config中无head_dim时应通过hidden_size计算"""
        with patch('msmodelslim.model.qwen3.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen3ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            # 创建没有head_dim的config
            adapter.config = type('Config', (), {'hidden_size': 128, 'num_attention_heads': 8})()

            with patch('msmodelslim.model.qwen3.model_adapter.get_logger') as mock_logger:
                result = adapter.get_head_dim()

                # 验证计算结果：128 // 8 = 16
                self.assertEqual(result, 16)

                # 验证警告被记录
                mock_logger().warning.assert_called_once()
                warning_msg = mock_logger().warning.call_args[0][0]
                self.assertIn('head_dim is not found', warning_msg)

    def test_get_head_dim_missing_hidden_size_when_called_then_raise_invalid_model_error(self):
        """测试get_head_dim方法：缺少hidden_size时应抛出InvalidModelError"""
        with patch('msmodelslim.model.qwen3.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen3ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            # 创建没有head_dim和hidden_size的config
            adapter.config = type('Config', (), {})()

            with self.assertRaises(InvalidModelError) as context:
                adapter.get_head_dim()

            # 验证异常消息
            self.assertIn("hidden_size is not found", str(context.exception))

    def test_get_head_dim_missing_num_attention_heads_when_called_then_raise_invalid_model_error(self):
        """测试get_head_dim方法：缺少num_attention_heads时应抛出InvalidModelError"""
        with patch('msmodelslim.model.qwen3.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen3ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            # 创建有hidden_size但没有num_attention_heads的config
            adapter.config = type('Config', (), {'hidden_size': 128})()

            with self.assertRaises(InvalidModelError) as context:
                adapter.get_head_dim()

            # 验证异常消息
            self.assertIn("num_attention_heads is not found", str(context.exception))

    def test_get_head_dim_with_zero_num_attention_heads_when_called_then_raise_invalid_model_error(self):
        """测试get_head_dim方法：num_attention_heads为0时应抛出InvalidModelError"""
        with patch('msmodelslim.model.qwen3.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen3ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = type('Config', (), {'hidden_size': 128, 'num_attention_heads': 0})()

            with self.assertRaises(InvalidModelError) as context:
                adapter.get_head_dim()

            # 验证异常消息
            self.assertIn("num_attention_heads is 0", str(context.exception))


class TestQwen3ModelAdapterGetNumKeyValueGroups(unittest.TestCase):
    """测试Qwen3ModelAdapter的get_num_key_value_groups方法"""

    def setUp(self):
        """测试前的准备工作"""
        self.model_type = 'Qwen3-8B'
        self.model_path = Path('.')

    def test_get_num_key_value_groups_with_valid_config_when_called_then_return_groups(self):
        """测试get_num_key_value_groups方法：有效配置时应返回正确的组数"""
        with patch('msmodelslim.model.qwen3.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen3ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = DummyConfig()
            adapter.model_path = self.model_path

            result = adapter.get_num_key_value_groups()

            # 验证计算结果：num_attention_heads=8, num_key_value_heads=4, groups=2
            expected = adapter.config.num_attention_heads // adapter.config.num_key_value_heads
            self.assertEqual(result, expected)
            self.assertEqual(result, 2)

    def test_get_num_key_value_groups_missing_num_attention_heads_when_called_then_raise_error(self):
        """测试get_num_key_value_groups方法：缺少num_attention_heads时应抛出InvalidModelError"""
        with patch('msmodelslim.model.qwen3.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen3ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = type('Config', (), {})()
            adapter.model_path = self.model_path

            with self.assertRaises(InvalidModelError) as context:
                adapter.get_num_key_value_groups()

            # 验证异常消息
            self.assertIn("num_attention_heads is not found", str(context.exception))

    def test_get_num_key_value_groups_missing_num_key_value_heads_when_called_then_raise_error(self):
        """测试get_num_key_value_groups方法：缺少num_key_value_heads时应抛出InvalidModelError"""
        with patch('msmodelslim.model.qwen3.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen3ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = type('Config', (), {'num_attention_heads': 8})()
            adapter.model_path = self.model_path

            with self.assertRaises(InvalidModelError) as context:
                adapter.get_num_key_value_groups()

            # 验证异常消息
            self.assertIn("num_key_value_heads is not found", str(context.exception))

    def test_get_num_key_value_groups_with_zero_num_key_value_heads_when_called_then_raise_error(self):
        """测试get_num_key_value_groups方法：num_key_value_heads为0时应抛出InvalidModelError"""
        with patch('msmodelslim.model.qwen3.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen3ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = type('Config', (), {'num_attention_heads': 8, 'num_key_value_heads': 0})()
            adapter.model_path = self.model_path

            with self.assertRaises(InvalidModelError) as context:
                adapter.get_num_key_value_groups()

            # 验证异常消息
            self.assertIn("num_key_value_heads is 0", str(context.exception))

    def test_get_num_key_value_groups_with_different_ratios_when_called_then_return_correct_groups(self):
        """测试get_num_key_value_groups方法：不同的头数比例应返回正确的组数"""
        with patch('msmodelslim.model.qwen3.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen3ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.model_path = self.model_path

            # 测试场景1: 16 / 2 = 8组
            adapter.config = type('Config', (), {'num_attention_heads': 16, 'num_key_value_heads': 2})()
            result = adapter.get_num_key_value_groups()
            self.assertEqual(result, 8)

            # 测试场景2: 32 / 8 = 4组
            adapter.config = type('Config', (), {'num_attention_heads': 32, 'num_key_value_heads': 8})()
            result = adapter.get_num_key_value_groups()
            self.assertEqual(result, 4)

            # 测试场景3: 12 / 12 = 1组（MHA情况）
            adapter.config = type('Config', (), {'num_attention_heads': 12, 'num_key_value_heads': 12})()
            result = adapter.get_num_key_value_groups()
            self.assertEqual(result, 1)
