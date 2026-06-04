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

import torch

from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.attention import DynamicCacheQuantizer
from msmodelslim.ir import AutoFakeQuantDynamicCache
from msmodelslim.ir.qal import QDType, QScope


def create_test_config():
    """创建测试配置的辅助函数"""
    return QConfig(dtype=QDType.INT8, scope=QScope.PER_CHANNEL, method="minmax", symmetric=True)


class TestDynamicCacheQuantizer:
    """测试DynamicCache量化器"""

    def test_initialization(self):
        """测试初始化"""
        config = create_test_config()
        quantizer = DynamicCacheQuantizer(config)
        assert quantizer.config == config
        assert hasattr(quantizer, 'input_quantizer')

    def test_forward_pass(self):
        """测试前向传播"""
        config = create_test_config()
        quantizer = DynamicCacheQuantizer(config)

        # 创建测试输入 - 模拟KV缓存的形状 [batch, num_heads, seq_len, head_dim]
        x = torch.randn(2, 8, 16, 64, dtype=torch.float32)

        # 测试量化输出
        with torch.no_grad():
            result = quantizer(x)

        assert result.shape == x.shape
        assert result.dtype == x.dtype

    def test_forward_with_different_shapes(self):
        """测试不同输入形状的前向传播"""
        config = create_test_config()
        quantizer = DynamicCacheQuantizer(config)

        # 测试不同的batch size和sequence length
        test_shapes = [
            (1, 4, 8, 32),  # 小batch
            (4, 12, 32, 64),  # 中等大小
            (2, 16, 64, 128),  # 较大尺寸
        ]

        for shape in test_shapes:
            x = torch.randn(*shape, dtype=torch.float32)

            # 测试
            with torch.no_grad():
                result = quantizer(x)

            assert result.shape == x.shape

    def test_setup_method(self):
        """测试setup方法"""
        config = create_test_config()
        quantizer = DynamicCacheQuantizer(config)
        quantizer.setup()  # setup方法目前是空的，但确保不会出错

    def test_deploy_method(self):
        """测试deploy方法"""
        config = create_test_config()
        quantizer = DynamicCacheQuantizer(config)

        # 先进行一次前向传播来初始化量化器
        x = torch.randn(2, 8, 16, 64, dtype=torch.float32)
        with torch.no_grad():
            _ = quantizer(x)

        # 测试deploy
        deployed = quantizer.deploy()
        assert isinstance(deployed, AutoFakeQuantDynamicCache)

    def test_quantizer_properties(self):
        """测试量化器属性"""
        config = create_test_config()
        quantizer = DynamicCacheQuantizer(config)

        # 检查基本属性
        assert hasattr(quantizer, 'config')
        assert hasattr(quantizer, 'input_quantizer')

        # 检查配置是否正确传递
        assert quantizer.config.dtype == QDType.INT8
        assert quantizer.config.scope == QScope.PER_CHANNEL
        assert quantizer.config.symmetric is True

    def test_tensor_dtype_consistency(self):
        """测试张量数据类型一致性"""
        config = create_test_config()
        quantizer = DynamicCacheQuantizer(config)

        # 测试float16输入
        x_fp16 = torch.randn(2, 8, 16, 64, dtype=torch.float16)
        with torch.no_grad():
            result_fp16 = quantizer(x_fp16)

        assert result_fp16.dtype == torch.float16

        # 测试float32输入
        x_fp32 = torch.randn(2, 8, 16, 64, dtype=torch.float32)
        with torch.no_grad():
            result_fp32 = quantizer(x_fp32)

        assert result_fp32.dtype == torch.float32

    def test_multiple_forward_passes(self):
        """测试多次前向传播"""
        config = create_test_config()
        quantizer = DynamicCacheQuantizer(config)
        x = torch.randn(2, 8, 16, 64, dtype=torch.float32)

        # 进行多次前向传播
        with torch.no_grad():
            result1 = quantizer(x)
            result2 = quantizer(x)

        assert result1.shape == x.shape
        assert result2.shape == x.shape
