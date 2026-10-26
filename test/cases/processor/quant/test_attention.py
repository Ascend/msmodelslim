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
from unittest.mock import patch

from torch import nn

from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.processor.quant.attention import DynamicCacheProcessorConfig, DynamicCacheQuantProcessor
from msmodelslim.utils.exception import VersionError


def create_basic_qconfig() -> QConfig:
    """创建基本的量化配置"""
    return QConfig(dtype=QDType.INT8, scope=QScope.PER_CHANNEL, symmetric=True, method="minmax")


def create_processor_config(include: list = None, exclude: list = None) -> DynamicCacheProcessorConfig:
    """创建处理器配置"""
    if include is None:
        include = ["*"]
    if exclude is None:
        exclude = []
    qconfig = create_basic_qconfig()
    return DynamicCacheProcessorConfig(
        qconfig=qconfig,
        include=include,
        exclude=exclude,
    )


def create_simple_model():
    """创建简单的测试模型"""

    class SimpleAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(64, 64)
            self.key = nn.Linear(64, 64)
            self.value = nn.Linear(64, 64)

        def forward(self, x):
            return self.query(x) + self.key(x) + self.value(x)

    class SimpleDecoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = SimpleAttention()  # 包含'attn'的属性名
            self.mlp = nn.Linear(64, 64)

        def forward(self, x):
            x = self.self_attn(x)
            x = self.mlp(x)
            return x

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 创建有层索引的结构，这样_detect_attention_layers能找到
            self.layers = nn.ModuleList([SimpleDecoderLayer() for _ in range(2)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    return SimpleModel()


class TestDynamicCacheQuantProcessor(unittest.TestCase):
    """测试DynamicCacheQuantProcessor的核心功能"""

    def __init__(self, *args, **kwargs):
        """初始化测试类"""
        super().__init__(*args, **kwargs)
        self.model = None

    def setUp(self):
        """测试前的准备工作"""
        self.model = create_simple_model()

    def test_config_creation(self):
        """测试配置创建"""
        config = create_processor_config()
        self.assertIsInstance(config, DynamicCacheProcessorConfig)
        self.assertEqual(config.qconfig.dtype, QDType.INT8)  # pylint: disable=no-member
        self.assertEqual(config.qconfig.scope, QScope.PER_CHANNEL)  # pylint: disable=no-member
        self.assertTrue(config.qconfig.symmetric)  # pylint: disable=no-member

    def test_config_with_include_exclude(self):
        """测试包含排除配置"""
        config = create_processor_config(include=["layer1"], exclude=["layer2"])
        self.assertEqual(config.include, ["layer1"])
        self.assertEqual(config.exclude, ["layer2"])

    @patch('msmodelslim.processor.quant.attention.DYNAMIC_AVAILABLE', True)
    def test_processor_initialization(self):
        """测试处理器初始化"""
        config = create_processor_config()
        processor = DynamicCacheQuantProcessor(self.model, config)

        self.assertEqual(processor.config, config)
        self.assertIsNotNone(processor.include)
        self.assertIsNotNone(processor.exclude)

    @patch('msmodelslim.processor.quant.attention.DYNAMIC_AVAILABLE', False)
    def test_processor_without_dynamic_cache(self):
        """测试在DynamicCache不可用时抛出异常"""
        config = create_processor_config()
        with self.assertRaises(VersionError):
            DynamicCacheQuantProcessor(self.model, config)

    def test_invalid_qconfig_scope(self):
        """测试无效的量化配置scope"""
        qconfig = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_TENSOR,  # 应该是PER_CHANNEL
            symmetric=True,
            method="minmax",
        )
        config = DynamicCacheProcessorConfig(qconfig=qconfig)
        with self.assertRaises(ValueError):
            DynamicCacheQuantProcessor(self.model, config)

    @patch('msmodelslim.processor.quant.attention.DYNAMIC_AVAILABLE', True)
    def test_processor_properties(self):
        """测试处理器基本属性"""
        config = create_processor_config()
        processor = DynamicCacheQuantProcessor(self.model, config)

        self.assertFalse(processor.is_data_free())
        self.assertTrue(processor.need_kv_cache())
        self.assertFalse(processor.support_distributed())

    @patch('msmodelslim.processor.quant.attention.DYNAMIC_AVAILABLE', True)
    @patch('msmodelslim.processor.quant.attention._detect_attention_layers')
    def test_pre_run_creates_quantizers(self, mock_detect):
        """测试pre_run创建量化器"""
        # 模拟_detect_attention_layers返回正确的层结构
        mock_detect.return_value = {0: "layers.0.self_attn", 1: "layers.1.self_attn"}

        config = create_processor_config()
        processor = DynamicCacheQuantProcessor(self.model, config)

        # Mock _create_quantizer方法
        with patch.object(processor, "_create_quantizer") as mock_create_quantizer:
            processor.pre_run()

        # 验证_create_quantizer被调用了正确的次数（每个检测到的层一次）
        self.assertEqual(mock_create_quantizer.call_count, 2)

    @patch('msmodelslim.processor.quant.attention.DYNAMIC_AVAILABLE', True)
    def test_attention_layer_detection(self):
        """测试attention层检测功能"""
        from msmodelslim.processor.quant.attention import _detect_attention_layers

        # 检测我们的简单模型中的attention层
        attention_layers = _detect_attention_layers(self.model)

        # 验证检测到了正确数量的层
        self.assertEqual(len(attention_layers), 2)
        # 验证层名称格式正确
        self.assertIn(0, attention_layers)
        self.assertIn(1, attention_layers)
        self.assertIn("self_attn", attention_layers[0])
        self.assertIn("self_attn", attention_layers[1])

    @patch('msmodelslim.processor.quant.attention.DYNAMIC_AVAILABLE', True)
    def test_config_validation(self):
        """测试配置验证"""
        # 测试有效配置
        valid_config = create_processor_config()
        self.assertIsInstance(valid_config.qconfig, QConfig)

        # 测试include/exclude为空列表
        empty_config = create_processor_config(include=[], exclude=[])
        self.assertEqual(empty_config.include, [])
        self.assertEqual(empty_config.exclude, [])


if __name__ == '__main__':
    unittest.main()
