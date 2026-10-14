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

from unittest.mock import MagicMock

from testing_utils.mock import mock_kia_library
from transformers import PreTrainedModel, PretrainedConfig

mock_kia_library()

from msmodelslim.pytorch.llm_ptq.model.deepseek_v2 import DeepseekV2Adapter  # noqa: E402
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig  # noqa: E402


class TestGetNormLinearSubgraph:
    """测试 get_norm_linear_subgraph 函数的不同场景"""

    @staticmethod
    def create_adapter(num_layers, is_chat=True):
        """创建模拟的预训练模型"""

        class MockDeepSeekConfig(PretrainedConfig):
            num_hidden_layers = num_layers
            model_type = "deepseek_v2"
            q_lora_rank = 8 if is_chat else None

        class MockDeepSeekModel(PreTrainedModel):
            config = MockDeepSeekConfig()

        model = MockDeepSeekModel(MockDeepSeekConfig())
        adapter = DeepseekV2Adapter(model)
        return adapter

    def test_chat_model_with_three_layers(self):
        """测试 chat 模型当层数为3层的情况"""
        adapter = self.create_adapter(3, is_chat=True)

        assert adapter.is_chat

        result = adapter.get_norm_linear_subgraph(AntiOutlierConfig(anti_method='m4'))

        # 检查总共有6个键（3层，每层3组）
        assert len(result) == 9

        # 检查第一层
        input_layernorm = 'model.layers.0.input_layernorm'
        q_a_layernorm = 'model.layers.0.self_attn.q_a_layernorm'
        assert input_layernorm in result
        assert q_a_layernorm in result
        assert len(result[input_layernorm]) == 2  # q_a_proj, kv_a_proj_with_mqa
        assert len(result[q_a_layernorm]) == 1  # q_b_proj

        # 检查 kv_b_proj -> o_proj 的连接
        kv_b_proj = 'model.layers.0.self_attn.kv_b_proj'
        assert kv_b_proj in result
        assert result[kv_b_proj] == ['model.layers.0.self_attn.o_proj']

    def test_lite_model_with_three_layers(self):
        """测试 lite 模型当层数为3层的情况"""
        adapter = self.create_adapter(3, is_chat=False)
        result = adapter.get_norm_linear_subgraph(AntiOutlierConfig(anti_method='m4'))

        # 检查总共有6个键（3层，每层3组）
        assert len(result) == 9

        # 检查第一层
        input_layernorm = 'model.layers.0.input_layernorm'
        kv_a_layernorm = 'model.layers.0.self_attn.kv_a_layernorm'
        assert input_layernorm in result
        assert kv_a_layernorm in result
        assert len(result[input_layernorm]) == 2  # q_proj, kv_a_proj_with_mqa
        assert len(result[kv_a_layernorm]) == 1  # kv_b_proj

        # 检查 kv_b_proj -> o_proj 的连接
        kv_b_proj = 'model.layers.0.self_attn.kv_b_proj'
        assert kv_b_proj in result
        assert result[kv_b_proj] == ['model.layers.0.self_attn.o_proj']

    def test_min_layer(self):
        """测试边界条件：模型最少层数"""
        adapter = self.create_adapter(1)
        result = adapter.get_norm_linear_subgraph(AntiOutlierConfig(anti_method='m4'))
        assert len(result) == 3  # 1层，每层3组


class TestModifySmoothArgs:
    """测试 modify_smooth_args 函数的不同场景"""

    @staticmethod
    def create_adapter(num_layers, is_chat=True):
        """创建模拟的预训练模型"""
        model = MagicMock()
        model.config = MagicMock()
        model.config.num_hidden_layers = num_layers
        model.config.model_type = "deepseek_v2"
        if is_chat:
            model.config.q_lora_rank = 8
        else:
            model.config.q_lora_rank = None
        adapter = DeepseekV2Adapter(model)
        return adapter

    @staticmethod
    def create_anti_config(method='m4'):
        cfg = AntiOutlierConfig(anti_method=method)
        return cfg

    def test_m4_with_norm_in_name(self):
        """测试当 anti_method=m4 且 norm_name 包含 'norm' 的情况"""
        adapter = self.create_adapter(3)
        cfg = self.create_anti_config()
        args, kwargs = adapter.modify_smooth_args(
            cfg=cfg, norm_name="layer.norm", linear_names=["dummy"], args=[1, 2], kwargs={"existing": "value"}
        )

        assert kwargs["is_shift"] is True
        assert kwargs["alpha"] == cfg.alpha
        assert args == [1, 2]  # 原始 args 应保持不变
        assert "existing" in kwargs  # 原有参数应保留

    def test_m4_with_kv_b_in_name(self):
        """测试当 anti_method=m4 且 linear_names 包含 'kv_b' 的情况"""
        adapter = self.create_adapter(3)
        cfg = self.create_anti_config()
        args, kwargs = adapter.modify_smooth_args(
            cfg=cfg, norm_name="layer.norm", linear_names=["kv_b_proj"], args=[], kwargs={}
        )

        assert kwargs["is_shift"] is False
        assert kwargs["alpha"] == cfg.alpha

    def test_non_m4_method(self):
        """测试当 anti_method 不是 m4 时不应修改参数"""
        adapter = self.create_adapter(3)
        cfg = self.create_anti_config(method="m1")
        original_kwargs = {"key": "value"}

        args, kwargs = adapter.modify_smooth_args(
            cfg=cfg, norm_name="layer.norm", linear_names=["dummy"], args=[3.14], kwargs=original_kwargs.copy()
        )

        assert kwargs == original_kwargs  # 参数应无变化
        assert "is_shift" not in kwargs
        assert "alpha" not in kwargs
