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

from typing import List

import pytest
import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.graph.adapter_types import AdapterConfig, MappingConfig
from msmodelslim.ir.norm_bias import RMSNormBias
from msmodelslim.processor.anti_outlier.common.smooth_components import StatKey
from msmodelslim.processor.anti_outlier.iter_smooth import IterSmoothProcessor, IterSmoothProcessorConfig
from msmodelslim.processor.anti_outlier.iter_smooth.interface import IterSmoothInterface


class MockModel(nn.Module):
    """模拟模型用于测试"""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 30)
        self.norm_layer = nn.LayerNorm(10)

    def get_submodule(self, name):
        """模拟get_submodule方法"""
        if name == "layer1":
            return self.layer1
        elif name == "layer2":
            return self.layer2
        elif name == "model.layers.0.input_layernorm":
            return self.norm_layer
        return None

    def set_submodule(self, name, module):
        """模拟set_submodule方法"""
        if name == "model.layers.0.input_layernorm":
            self.norm_layer = module


class MockAdapter(IterSmoothInterface):
    """模拟实现IterSmoothInterface的适配器"""

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        return [
            AdapterConfig(
                subgraph_type="norm-linear",
                mapping=MappingConfig(source="model.layers.0.input_layernorm", targets=["layer1"]),
            )
        ]


class MockAdapterWithInvalidNorm(IterSmoothInterface):
    """模拟包含无效norm_module的适配器"""

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        return [
            AdapterConfig(subgraph_type="norm-linear", mapping=MappingConfig(source="invalid_norm", targets=["layer1"]))
        ]


class MockAdapterWithNormWithoutWeight(IterSmoothInterface):
    """模拟包含无weight属性的norm_module的适配器"""

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        return [
            AdapterConfig(subgraph_type="norm-linear", mapping=MappingConfig(source="norm_layer", targets=["layer1"]))
        ]


class MockNormWithoutWeight(nn.Module):
    """模拟没有weight属性的norm模块（故意不设置weight属性）"""


class TestIterSmoothProcessor:
    """测试IterSmoothProcessor的各种功能"""

    def test_iter_smooth_processor_init(self):
        """验证IterSmoothProcessor类在初始化时是否正确设置所有属性"""
        # 前置操作：创建一个IterSmoothProcessor实例，传入model、config和adapter
        model = MockModel()
        config = IterSmoothProcessorConfig()
        adapter = MockAdapter()

        processor = IterSmoothProcessor(model, config, adapter)

        # 验证初始化后的属性设置
        assert processor.model == model
        assert processor.config == config
        assert processor.adapter == adapter
        assert isinstance(processor.stats_collector.act_stats, dict)
        # 验证config属性
        assert processor.config.alpha == 0.9
        assert processor.config.scale_min == 1e-5
        assert processor.config.symmetric is True
        assert processor.config.enable_subgraph_type == ["norm-linear", "linear-linear", "ov", "up-down"]

    def test_iter_smooth_processor_preprocess_normal(self):
        """验证preprocess方法在正常情况下是否能正确处理norm-linear子图"""
        # 前置操作：创建一个包含norm-linear子图的模型，并调用preprocess方法
        model = MockModel()
        config = IterSmoothProcessorConfig()
        adapter = MockAdapter()

        processor = IterSmoothProcessor(model, config, adapter)

        # 设置adapter_config（模拟pre_run后的状态）
        processor.global_adapter_config = adapter.get_adapter_config_for_subgraph()

        # 创建BatchProcessRequest
        request = BatchProcessRequest(name="model.layers.0", module=model.layer1)

        # 调用preprocess方法
        processor.preprocess(request)

        # 验证norm_layer已被替换为RMSNormBias
        assert isinstance(model.norm_layer, RMSNormBias)
        assert model.norm_layer.weight.shape == torch.Size([10])
        assert model.norm_layer.bias.shape == torch.Size([10])

    def test_iter_smooth_processor_preprocess_invalid(self):
        """验证preprocess方法在遇到无效norm_module时是否能正确处理异常"""
        # 前置操作：创建一个包含无效norm_module的模型，并调用preprocess方法
        model = MockModel()
        config = IterSmoothProcessorConfig()
        adapter = MockAdapterWithInvalidNorm()

        processor = IterSmoothProcessor(model, config, adapter)

        # 设置adapter_config（模拟pre_run后的状态）
        processor.global_adapter_config = adapter.get_adapter_config_for_subgraph()

        # 创建BatchProcessRequest
        request = BatchProcessRequest(name="test_module", module=model.layer1)

        # 调用preprocess方法，应该不会抛出异常，而是记录警告
        # 因为代码中有try-except处理
        processor.preprocess(request)

        # 验证模型没有被修改（因为norm_module为None）
        assert isinstance(model.norm_layer, nn.LayerNorm)

    def test_iter_smooth_processor_preprocess_norm_without_weight(self):
        """验证preprocess方法在遇到没有weight属性的norm_module时的处理"""
        # 前置操作：创建一个包含无weight属性的norm_module的模型
        model = MockModel()
        model.norm_layer = MockNormWithoutWeight()  # 替换为没有weight的norm

        config = IterSmoothProcessorConfig()
        adapter = MockAdapterWithNormWithoutWeight()

        processor = IterSmoothProcessor(model, config, adapter)

        # 设置adapter_config（模拟pre_run后的状态）
        processor.global_adapter_config = adapter.get_adapter_config_for_subgraph()

        # 创建BatchProcessRequest
        request = BatchProcessRequest(name="test_module", module=model.layer1)

        # 调用preprocess方法，应该不会抛出异常，而是记录警告
        processor.preprocess(request)

        # 验证模型没有被修改（因为norm_module没有weight属性）
        assert isinstance(model.norm_layer, MockNormWithoutWeight)

    def test_iter_smooth_processor_get_stats_hook(self):
        """验证_get_stats_hook方法是否能正确收集和更新统计信息"""
        # 前置操作：模拟一个输入张量，调用_get_stats_hook方法
        model = MockModel()
        config = IterSmoothProcessorConfig()
        adapter = MockAdapter()

        processor = IterSmoothProcessor(model, config, adapter)

        # 获取stats_hook函数
        stats_hook = processor.stats_collector.create_hook("test_module")

        # 创建模拟的Linear模块和输入张量
        linear_module = nn.Linear(10, 20)
        input_tensor = torch.randn(5, 10)  # [batch_size, hidden_dim]

        # 调用stats_hook
        stats_hook(linear_module, (input_tensor,), {})

        # 验证统计信息被正确收集
        assert "test_module" in processor.stats_collector.act_stats
        stats = processor.stats_collector.act_stats["test_module"]

        # 验证各种统计信息都存在
        assert StatKey.TENSOR in stats
        assert StatKey.STAT_KEY_MAX in stats
        assert StatKey.STAT_KEY_MIN in stats
        assert StatKey.STAT_KEY_SHIFT in stats
        assert StatKey.STAT_KEY_SMOOTH_SCALE in stats

        # 验证统计信息的形状
        assert stats[StatKey.TENSOR].shape == input_tensor.shape
        assert stats[StatKey.STAT_KEY_MAX].shape == torch.Size([10])  # hidden_dim
        assert stats[StatKey.STAT_KEY_MIN].shape == torch.Size([10])
        assert stats[StatKey.STAT_KEY_SHIFT].shape == torch.Size([10])
        assert stats[StatKey.STAT_KEY_SMOOTH_SCALE].shape == torch.Size([10])

    def test_iter_smooth_processor_get_stats_hook_multiple_calls(self):
        """验证_get_stats_hook方法在多次调用时能正确更新统计信息"""
        model = MockModel()
        config = IterSmoothProcessorConfig()
        adapter = MockAdapter()

        processor = IterSmoothProcessor(model, config, adapter)
        stats_hook = processor.stats_collector.create_hook("test_module")

        linear_module = nn.Linear(10, 20)

        # 第一次调用
        input_tensor1 = torch.randn(5, 10)
        stats_hook(linear_module, (input_tensor1,), {})

        # 第二次调用
        input_tensor2 = torch.randn(3, 10)
        stats_hook(linear_module, (input_tensor2,), {})

        # 验证统计信息被正确更新
        stats = processor.stats_collector.act_stats["test_module"]

        # 验证max值被正确更新（应该取两次的最大值）
        expected_max = torch.max(torch.max(input_tensor1, dim=0)[0], torch.max(input_tensor2, dim=0)[0])
        assert torch.allclose(stats[StatKey.STAT_KEY_MAX], expected_max)

        # 验证min值被正确更新（应该取两次的最小值）
        expected_min = torch.min(torch.min(input_tensor1, dim=0)[0], torch.min(input_tensor2, dim=0)[0])
        assert torch.allclose(stats[StatKey.STAT_KEY_MIN], expected_min)

    def test_iter_smooth_processor_support_distributed(self):
        """验证support_distributed方法返回True"""
        model = MockModel()
        config = IterSmoothProcessorConfig()
        adapter = MockAdapter()

        processor = IterSmoothProcessor(model, config, adapter)

        # 验证支持分布式
        assert processor.support_distributed() is True


if __name__ == "__main__":
    pytest.main([__file__])
