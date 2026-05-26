#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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
from unittest.mock import Mock

import torch
import torch.nn as nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.ir import FakeQuantActivationPerHead
from msmodelslim.processor.quant.fa3 import (
    FA3QuantProcessor,
    FA3QuantProcessorConfig,
    FA3QuantAdapterInterface,
    FA3QuantPlaceHolder,
)
from msmodelslim.processor.quant.fa3.processor import _FA3PerHeadObserver
from msmodelslim.utils.exception import UnsupportedError


def create_processor_config(include: list = None, exclude: list = None) -> FA3QuantProcessorConfig:
    """创建处理器配置"""
    if include is None:
        include = ["*"]
    if exclude is None:
        exclude = []
    return FA3QuantProcessorConfig(
        include=include,
        exclude=exclude,
    )


def create_mock_adapter() -> FA3QuantAdapterInterface:
    """创建模拟适配器"""
    adapter = Mock(spec=FA3QuantAdapterInterface)
    adapter.inject_fa3_placeholders = Mock()
    return adapter


def create_simple_model():
    """创建简单的测试模型"""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(64, 64)
            
        def forward(self, x):
            return self.linear(x)
    
    return SimpleModel()


class TestFA3QuantProcessor(unittest.TestCase):
    """测试FA3QuantProcessor的核心功能"""

    def __init__(self, *args, **kwargs):
        """初始化测试类"""
        super().__init__(*args, **kwargs)
        self.model = None
        self.adapter = None

    def setUp(self):
        """测试前的准备工作"""
        self.model = create_simple_model()
        self.adapter = create_mock_adapter()

    def test_config_creation(self):
        """测试配置创建"""
        config = create_processor_config()
        self.assertIsInstance(config, FA3QuantProcessorConfig)
        self.assertEqual(config.type, "fa3_quant")
        self.assertEqual(config.include, ["*"])
        self.assertEqual(config.exclude, [])

    def test_config_with_include_exclude(self):
        """测试包含排除配置"""
        config = create_processor_config(include=["layer1"], exclude=["layer2"])
        self.assertEqual(config.include, ["layer1"])
        self.assertEqual(config.exclude, ["layer2"])

    def test_processor_initialization(self):
        """测试处理器初始化"""
        config = create_processor_config()
        processor = FA3QuantProcessor(self.model, config, self.adapter)
        
        self.assertEqual(processor.config, config)
        self.assertEqual(processor.adapter, self.adapter)
        self.assertIsNotNone(processor.include)
        self.assertIsNotNone(processor.exclude)

    def test_processor_without_adapter(self):
        """测试在没有适配器时抛出异常"""
        config = create_processor_config()
        with self.assertRaises(UnsupportedError):
            FA3QuantProcessor(self.model, config, adapter=None)

    def test_processor_with_invalid_adapter(self):
        """测试使用无效适配器时抛出异常"""
        config = create_processor_config()
        invalid_adapter = Mock()
        with self.assertRaises(UnsupportedError):
            FA3QuantProcessor(self.model, config, adapter=invalid_adapter)

    def test_processor_properties(self):
        """测试处理器基本属性"""
        config = create_processor_config()
        processor = FA3QuantProcessor(self.model, config, self.adapter)
        
        self.assertFalse(processor.is_data_free())
        self.assertTrue(processor.support_distributed())

    def test_preprocess_calls_adapter_and_replaces_placeholder(self):
        """测试preprocess调用适配器并替换占位符"""
        config = create_processor_config()
        processor = FA3QuantProcessor(self.model, config, self.adapter)
        
        test_module = nn.Module()
        test_module.placeholder = FA3QuantPlaceHolder(ratio=0.8)
        self.model.test_module = test_module
        
        request = BatchProcessRequest(
            name="test_module",
            module=test_module,
            datas=None,
            outputs=None
        )
        
        processor.preprocess(request)
        
        # 验证适配器被调用
        self.adapter.inject_fa3_placeholders.assert_called_once()
        # 验证占位符被替换为监听器
        self.assertIsInstance(test_module.placeholder, _FA3PerHeadObserver)

    def test_postprocess_replaces_observer_with_ir(self):
        """测试postprocess将监听器替换为IR"""
        config = create_processor_config()
        processor = FA3QuantProcessor(self.model, config, self.adapter)
        
        test_module = nn.Module()
        observer = _FA3PerHeadObserver(ratio=0.8)
        test_module.observer = observer
        self.model.test_module = test_module
        
        # 模拟监听器收集了数据
        with torch.no_grad():
            test_input = torch.randn(2, 4, 10, 16)  # (B, H, S, D)
            observer(test_input)
        
        request = BatchProcessRequest(
            name="test_module",
            module=test_module,
            datas=None,
            outputs=None
        )
        
        processor.postprocess(request)
        
        # 验证监听器被替换为IR
        self.assertIsInstance(test_module.observer, FakeQuantActivationPerHead)

        # 测试 FakeQuantActivationPerHead.forward()
        ir_module = test_module.observer
        test_input = torch.randn(2, 4, 10, 16)  # (B, H, S, D)
        with torch.no_grad():
            output = ir_module(test_input)
        # 验证输出形状正确
        self.assertEqual(output.shape, test_input.shape)


if __name__ == '__main__':
    unittest.main()
