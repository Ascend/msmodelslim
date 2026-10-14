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

from unittest.mock import MagicMock

import pytest
from testing_utils.mock import mock_kia_library

mock_kia_library()

from msmodelslim.pytorch.llm_ptq.model.qwen3.qwen3 import Qwen3Adapter  # noqa: E402
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig  # noqa: E402


class TestGetNormLinearSubgraph:
    """测试 get_norm_linear_subgraph 函数的不同场景"""

    @staticmethod
    def create_adapter(num_layers, is_moe=False):
        """创建模拟的预训练模型"""
        model = MagicMock()
        model.config = MagicMock()
        model.config.num_hidden_layers = num_layers
        model.config.model_type = "qwen3_moe" if is_moe else "qwen3"
        model.config.num_attention_heads = 32
        model.config.num_key_value_heads = 8
        adapter = Qwen3Adapter(model)
        return adapter

    def test_normal_model_with_three_layers(self):
        """测试普通模型当层数为3层的情况"""
        adapter = self.create_adapter(3, is_moe=False)
        result = adapter.get_norm_linear_subgraph(AntiOutlierConfig(anti_method='m4'))

        # 对于qwen3稠密模型，应用m4异常值抑制，每一层应该包含4组
        assert len(result) == 12

        # 检查第一层
        input_layernorm = 'model.layers.0.input_layernorm'
        post_layernorm = 'model.layers.0.post_attention_layernorm'
        v_proj = 'model.layers.0.self_attn.v_proj'

        assert input_layernorm in result
        assert post_layernorm in result
        assert v_proj in result

        # 检查连接关系
        assert len(result[input_layernorm]) == 3  # q_proj, k_proj, v_proj
        assert len(result[post_layernorm]) == 2  # gate_proj, up_proj
        assert result[v_proj] == ['model.layers.0.self_attn.o_proj']

    def test_moe_model_with_three_layers(self):
        """测试 MoE 模型当层数为3层的情况"""
        adapter = self.create_adapter(3, is_moe=True)
        result = adapter.get_norm_linear_subgraph(AntiOutlierConfig(anti_method='m4'))

        # 对于qwen3 MOE模型，应用m4异常值抑制，每一层应该包含2组
        assert len(result) == 6

        # 检查第一层
        input_layernorm = 'model.layers.0.input_layernorm'
        v_proj = 'model.layers.0.self_attn.v_proj'

        assert input_layernorm in result
        assert v_proj in result
        assert 'model.layers.0.post_attention_layernorm' not in result

    def test_min_layer(self):
        """测试边界条件：模型最少层数"""
        adapter = self.create_adapter(1)
        result = adapter.get_norm_linear_subgraph(AntiOutlierConfig(anti_method='m4'))
        assert len(result) == 4  # 1层，每层3个norm

    def test_invalid_layers_zero(self):
        """测试边界条件：模型层数小于条件值"""
        adapter = self.create_adapter(0)
        with pytest.raises(ValueError) as excinfo:
            adapter.get_norm_linear_subgraph(AntiOutlierConfig(anti_method='m4'))
        assert "invalid" in str(excinfo.value).lower()

    def test_invalid_layers_exceed(self):
        """测试边界条件：模型层数大于条件值"""
        adapter = self.create_adapter(1000)
        with pytest.raises(ValueError) as excinfo:
            adapter.get_norm_linear_subgraph(AntiOutlierConfig(anti_method='m4'))
        assert "invalid" in str(excinfo.value).lower()


class TestModifySmoothArgs:
    """测试 modify_smooth_args 函数的不同场景"""

    @staticmethod
    def create_adapter(num_layers, is_moe=False):
        """创建模拟的预训练模型"""

        class MockQwenModelConfig:
            num_hidden_layers = num_layers
            model_type = "qwen3_moe" if is_moe else "qwen3"
            num_attention_heads = 32
            num_key_value_heads = 8

        model = MagicMock()
        model.config = MockQwenModelConfig()
        adapter = Qwen3Adapter(model)
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
            cfg=cfg,
            norm_name="layer.norm",
            linear_names="dummy",
            args=[1, 2],
            kwargs={"existing": "value", "num_attention_heads": None},
        )

        assert kwargs["is_shift"] is True
        assert kwargs["alpha"] == cfg.alpha
        assert kwargs["num_attention_heads"] == [32, 8]  # 检查注意力头数设置
        assert args == [1, 2]  # 原始 args 应保持不变
        assert "existing" in kwargs  # 原有参数应保留

    def test_m4_without_norm_in_name(self):
        """测试当 anti_method=m4 但 norm_name 不含 'norm' 的情况"""
        adapter = self.create_adapter(3)
        cfg = self.create_anti_config()
        args, kwargs = adapter.modify_smooth_args(
            cfg=cfg, norm_name="linear_layer", linear_names="dummy", args=[], kwargs={"num_attention_heads": None}
        )

        assert kwargs["is_shift"] is False
        assert kwargs["alpha"] == cfg.alpha
        assert kwargs["num_attention_heads"] == [32, 8]

    def test_non_m4_method(self):
        """测试当 anti_method 不是 m4 时不应修改参数"""
        adapter = self.create_adapter(3)
        cfg = self.create_anti_config(method="m1")
        original_kwargs = {"key": "value", "num_attention_heads": None}

        args, kwargs = adapter.modify_smooth_args(
            cfg=cfg, norm_name="layer.norm", linear_names="dummy", args=[3.14], kwargs=original_kwargs.copy()
        )

        assert kwargs == original_kwargs  # 参数应无变化
        assert "is_shift" not in kwargs
        assert "alpha" not in kwargs
