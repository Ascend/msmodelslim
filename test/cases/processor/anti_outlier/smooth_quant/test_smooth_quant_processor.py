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

from typing import List

import pytest
import torch
import torch.nn as nn

from msmodelslim.processor.anti_outlier.common.subgraph_type import NormLinearSubgraph
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.graph.adapter_types import AdapterConfig, MappingConfig
from msmodelslim.ir.norm_bias import RMSNormBias
from msmodelslim.processor.anti_outlier.common import (
    SmoothQuantConfig,
    SmoothQuantContext,
    StatKey,
)
from msmodelslim.processor.anti_outlier.smooth_quant import SmoothQuantProcessor, SmoothQuantProcessorConfig
from msmodelslim.processor.anti_outlier.smooth_quant.api import smooth_quant, smooth_quant_impl_norm_linear
from msmodelslim.processor.anti_outlier.smooth_quant.interface import SmoothQuantInterface
from msmodelslim.utils.exception import UnsupportedError  # noqa: F401


class MockRMSNorm(nn.Module):
    """模拟RMSNorm模块"""

    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = None


class MockModel(nn.Module):
    """模拟模型用于测试"""

    def __init__(self, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(hidden_size, hidden_size * 2)
        self.linear2 = nn.Linear(hidden_size, hidden_size * 2)
        self.norm = MockRMSNorm(hidden_size)
        self._submodules = {
            "model.layers.0.input_layernorm": self.norm,
            "model.layers.0.mlp.gate_proj": self.linear1,
            "model.layers.0.mlp.up_proj": self.linear2,
        }

    def get_submodule(self, name):
        return self._submodules.get(name)

    def set_submodule(self, name, module):
        self._submodules[name] = module
        if name == "model.layers.0.input_layernorm":
            self.norm = module


class MockAdapter(SmoothQuantInterface):
    """模拟SmoothQuantInterface适配器"""

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        return [
            AdapterConfig(
                subgraph_type="norm-linear",
                mapping=MappingConfig(
                    source="model.layers.0.input_layernorm",
                    targets=["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"],
                ),
            ),
            # 添加一个非 norm-linear 类型，用于测试过滤
            AdapterConfig(
                subgraph_type="linear-linear",
                mapping=MappingConfig(source="model.layers.0.mlp.gate_proj", targets=["other"]),
            ),
        ]


class MockInvalidAdapter:
    """不实现SmoothQuantInterface的适配器"""

    pass


class TestSmoothQuantProcessorConfig:
    """测试SmoothQuantProcessorConfig配置类"""

    @staticmethod
    def test_custom_values():
        """验证自定义配置值"""
        config = SmoothQuantProcessorConfig(alpha=0.7, symmetric=False, include=["layer1"], exclude=["layer2"])
        assert config.alpha == 0.7
        assert config.symmetric is False
        assert config.include == ["layer1"]
        assert config.exclude == ["layer2"]


class TestSmoothQuantProcessor:
    """测试SmoothQuantProcessor处理器"""

    @staticmethod
    def test_init():
        """验证处理器初始化"""
        model = MockModel()
        config = SmoothQuantProcessorConfig()
        adapter = MockAdapter()

        processor = SmoothQuantProcessor(model, config, adapter)

        assert processor.model == model
        assert processor.config == config
        assert processor.adapter == adapter
        assert processor.ENABLE_SUBGRAPH_TYPE == ["norm-linear"]
        assert processor.stats_collector is not None
        assert processor.dist_helper is None

    @staticmethod
    def test_init_with_invalid_adapter():
        """验证使用无效适配器时回退到默认适配器逻辑"""
        model = MockModel()
        config = SmoothQuantProcessorConfig()
        adapter = MockInvalidAdapter()

        processor = SmoothQuantProcessor(model, config, adapter)

        # 不再抛出异常，而是回退到默认适配器逻辑
        assert isinstance(processor, SmoothQuantProcessor)
        assert processor.is_defalut_adapter is True

    @staticmethod
    def test_filter_adapter_configs_by_config():
        """验证_filter_adapter_configs_by_config只保留norm-linear类型"""
        model = MockModel()
        config = SmoothQuantProcessorConfig()
        adapter = MockAdapter()

        processor = SmoothQuantProcessor(model, config, adapter)
        adapter_configs = adapter.get_adapter_config_for_subgraph()

        result = processor._filter_adapter_configs_by_config(adapter_configs, config, "model.layers.0")

        # 应该只保留 norm-linear 类型
        assert len(result) == 1
        assert result[0].subgraph_type == "norm-linear"

    @staticmethod
    def test_filter_with_include_exclude():
        """验证include/exclude规则过滤"""
        model = MockModel()
        config = SmoothQuantProcessorConfig(exclude=["model.layers.0.input_layernorm"])
        adapter = MockAdapter()

        processor = SmoothQuantProcessor(model, config, adapter)
        adapter_configs = adapter.get_adapter_config_for_subgraph()

        result = processor._filter_adapter_configs_by_config(adapter_configs, config, "model.layers.0")

        # 应该被exclude规则排除
        assert len(result) == 0

    @staticmethod
    def test_preprocess():
        """验证preprocess方法正确替换norm模块"""
        model = MockModel()
        config = SmoothQuantProcessorConfig(symmetric=False)
        adapter = MockAdapter()

        processor = SmoothQuantProcessor(model, config, adapter)
        processor.global_adapter_config = adapter.get_adapter_config_for_subgraph()

        request = BatchProcessRequest(name="model.layers.0", module=model.linear1)
        processor.preprocess(request)

        # 验证norm模块被替换为RMSNormBias
        replaced_norm = model.get_submodule("model.layers.0.input_layernorm")
        assert isinstance(replaced_norm, RMSNormBias)
        assert replaced_norm.weight.shape == torch.Size([model.hidden_size])
        assert replaced_norm.bias.shape == torch.Size([model.hidden_size])

    @staticmethod
    def test_build_smooth_context_with_stats():
        """验证_build_smooth_context正确构建上下文"""
        model = MockModel()
        config = SmoothQuantProcessorConfig()
        adapter = MockAdapter()

        processor = SmoothQuantProcessor(model, config, adapter)

        # 模拟统计信息
        linear_name = "model.layers.0.mlp.gate_proj"
        processor.stats_collector.act_stats[linear_name] = {
            StatKey.STAT_KEY_SMOOTH_SCALE: torch.ones(model.hidden_size),
            StatKey.STAT_KEY_SHIFT: torch.zeros(model.hidden_size),
        }

        context = processor._build_smooth_context([linear_name])

        assert context is not None
        assert context.version == 1
        assert context.a_smooth_scale.shape == torch.Size([model.hidden_size])
        assert context.shift.shape == torch.Size([model.hidden_size])

    @staticmethod
    def test_build_smooth_context_missing_smooth_scale():
        """验证_build_smooth_context处理统计信息中缺失smooth_scale的情况"""
        model = MockModel()
        config = SmoothQuantProcessorConfig()
        adapter = MockAdapter()

        processor = SmoothQuantProcessor(model, config, adapter)

        # 模拟统计信息存在但不包含STAT_KEY_SMOOTH_SCALE
        linear_name = "model.layers.0.mlp.gate_proj"
        processor.stats_collector.act_stats[linear_name] = {
            StatKey.STAT_KEY_SHIFT: torch.zeros(model.hidden_size),
            # 不设置 STAT_KEY_SMOOTH_SCALE，使 a_smooth_scale 为 None
        }

        context = processor._build_smooth_context([linear_name])

        # 当 a_smooth_scale 为 None 时应返回 None
        assert context is None


class TestSmoothQuantAPI:
    """测试smooth_quant API"""

    @staticmethod
    def test_smooth_quant_impl_norm_linear_symmetric():
        """验证smooth_quant_impl_norm_linear对称模式"""
        hidden_size = 32
        norm = RMSNormBias(hidden_size)
        linear1 = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        linear2 = nn.Linear(hidden_size, hidden_size * 2, bias=False)

        # 初始化权重
        nn.init.normal_(linear1.weight, std=0.1)
        nn.init.normal_(linear2.weight, std=0.1)

        subgraph = NormLinearSubgraph(norm, [linear1, linear2])
        config = SmoothQuantConfig(alpha=0.5, shift=False)
        context = SmoothQuantContext(version=1, a_smooth_scale=torch.rand(hidden_size) + 0.1, shift=None)

        # 记录原始权重
        orig_linear1_weight = linear1.weight.data.clone()
        orig_norm_weight = norm.weight.data.clone()

        smooth_quant_impl_norm_linear(subgraph, config, context)

        # 验证权重已被修改
        assert not torch.allclose(linear1.weight.data, orig_linear1_weight)
        assert not torch.allclose(norm.weight.data, orig_norm_weight)

    @staticmethod
    def test_smooth_quant_impl_norm_linear_asymmetric():
        """验证smooth_quant_impl_norm_linear非对称模式（带shift）"""
        hidden_size = 32
        norm = RMSNormBias(hidden_size)
        linear = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        nn.init.normal_(linear.weight, std=0.1)

        subgraph = NormLinearSubgraph(norm, [linear])
        config = SmoothQuantConfig(alpha=0.5, shift=True)
        context = SmoothQuantContext(
            version=1, a_smooth_scale=torch.rand(hidden_size) + 0.1, shift=torch.randn(hidden_size) * 0.1
        )

        smooth_quant_impl_norm_linear(subgraph, config, context)

        # 验证linear层bias被创建并设置
        assert linear.bias is not None
        # 验证norm层的shift被设置
        assert not torch.allclose(norm.bias.data, torch.zeros(hidden_size))

    @staticmethod
    def test_smooth_quant_dispatch():
        """验证smooth_quant API正确分发到实现"""
        hidden_size = 16
        norm = RMSNormBias(hidden_size)
        linear = nn.Linear(hidden_size, hidden_size, bias=False)

        subgraph = NormLinearSubgraph(norm, [linear])
        config = SmoothQuantConfig(alpha=0.5, shift=False)
        context = SmoothQuantContext(version=1, a_smooth_scale=torch.ones(hidden_size), shift=None)

        # 应该不抛出异常
        smooth_quant(subgraph, config, context)


class TestSmoothQuantProcessorIntegration:
    """集成测试"""

    @staticmethod
    def test_full_preprocess_postprocess_flow():
        """验证完整的预处理和后处理流程"""
        model = MockModel(hidden_size=32)
        config = SmoothQuantProcessorConfig(alpha=0.5, symmetric=True)
        adapter = MockAdapter()

        processor = SmoothQuantProcessor(model, config, adapter)
        processor.global_adapter_config = adapter.get_adapter_config_for_subgraph()

        # 预处理
        request = BatchProcessRequest(name="model.layers.0", module=model.linear1)
        processor.preprocess(request)

        # 模拟钩子收集统计信息
        linear_name = "model.layers.0.mlp.gate_proj"
        hook = processor.stats_collector.create_hook(linear_name, "norm-linear")
        input_tensor = torch.randn(4, 32)
        hook(model.linear1, (input_tensor,), None)

        # 验证统计信息已收集
        assert linear_name in processor.stats_collector.act_stats
        stats = processor.stats_collector.act_stats[linear_name]
        assert StatKey.STAT_KEY_SMOOTH_SCALE in stats
        assert StatKey.STAT_KEY_SHIFT in stats

        # 后处理
        processor.postprocess(request)

        # 验证统计信息已清理
        assert len(processor.stats_collector.act_stats) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
