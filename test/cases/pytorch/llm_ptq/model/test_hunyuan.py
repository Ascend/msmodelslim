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

from msmodelslim.pytorch.llm_ptq.model.hunyuan.hunyuan_large import HunyuanLargeAdapter  # noqa: E402
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig  # noqa: E402


class TestGetNormLinearSubgraph:
    """测试 get_norm_linear_subgraph 函数的不同场景"""

    @staticmethod
    def create_adapter(num_layers):
        """创建模拟的预训练模型"""
        model = MagicMock()
        model.config = MagicMock()
        model.config.num_hidden_layers = num_layers
        adapter = HunyuanLargeAdapter(model)
        return adapter

    def test_normal_case_with_three_layers(self):
        """测试当模型层为3层 覆盖全分支情形"""
        adapter = self.create_adapter(3)
        result = adapter.get_norm_linear_subgraph(AntiOutlierConfig(anti_method='m4'))

        # 检查总共有3层
        assert len(result) == 3

        # 层0（偶数）：包含 q_proj, k_proj, v_proj
        key0 = 'model.layers.0.input_layernorm'
        assert key0 in result
        assert len(result[key0]) == 3
        assert 'model.layers.0.self_attn.q_proj' in result[key0]
        assert 'model.layers.0.self_attn.k_proj' in result[key0]
        assert 'model.layers.0.self_attn.v_proj' in result[key0]

        # 层1（奇数）：仅包含 q_proj
        key1 = 'model.layers.1.input_layernorm'
        assert key1 in result
        assert len(result[key1]) == 1
        assert 'model.layers.1.self_attn.q_proj' in result[key1]

        # 层2（偶数）：包含三个 proj
        key2 = 'model.layers.2.input_layernorm'
        assert key2 in result
        assert len(result[key2]) == 3

    def test_min_layer(self):
        """测试边界条件：模型最少层数"""
        adapter = self.create_adapter(1)
        result = adapter.get_norm_linear_subgraph(AntiOutlierConfig(anti_method='m4'))

        assert len(result) == 1
        key = 'model.layers.0.input_layernorm'
        assert key in result
        assert len(result[key]) == 3
        assert 'model.layers.0.self_attn.q_proj' in result[key]
        assert 'model.layers.0.self_attn.k_proj' in result[key]
        assert 'model.layers.0.self_attn.v_proj' in result[key]

    def test_max_layers(self):
        """测试边界条件：模型最多层数"""
        adapter = self.create_adapter(999)
        result = adapter.get_norm_linear_subgraph(AntiOutlierConfig(anti_method='m4'))

        # 检查最后一层（998，偶数）
        key = 'model.layers.998.input_layernorm'
        assert key in result
        assert len(result[key]) == 3
        assert 'model.layers.998.self_attn.q_proj' in result[key]

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
    def create_adapter(num_layers):
        """创建模拟的预训练模型"""
        model = MagicMock()
        model.config = MagicMock()
        model.config.num_hidden_layers = num_layers
        adapter = HunyuanLargeAdapter(model)
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
            cfg=cfg, norm_name="layer.norm", linear_names="dummy", args=[1, 2], kwargs={"existing": "value"}
        )

        assert kwargs["is_shift"] is True
        assert kwargs["alpha"] == cfg.alpha
        assert args == [1, 2]  # 原始 args 应保持不变
        assert "existing" in kwargs  # 原有参数应保留

    def test_m4_without_norm_in_name(self):
        """测试当 anti_method=m4 但 norm_name 不含 'norm' 的情况"""
        adapter = self.create_adapter(3)
        cfg = self.create_anti_config()
        args, kwargs = adapter.modify_smooth_args(
            cfg=cfg, norm_name="linear_layer", linear_names="dummy", args=[], kwargs={}
        )

        assert kwargs["is_shift"] is False
        assert kwargs["alpha"] == cfg.alpha

    def test_non_m4_method(self):
        """测试当 anti_method 不是 m4 时不应修改参数"""
        adapter = self.create_adapter(3)
        cfg = self.create_anti_config(method="m1")
        original_kwargs = {"key": "value"}

        args, kwargs = adapter.modify_smooth_args(
            cfg=cfg,
            norm_name="layer.norm",  # 即使包含 norm 也不应生效
            linear_names="dummy",
            args=[3.14],
            kwargs=original_kwargs.copy(),
        )

        assert kwargs == original_kwargs  # 参数应无变化
        assert "is_shift" not in kwargs
        assert "alpha" not in kwargs

    def test_edge_case_empty_norm_name(self):
        """测试边界条件：空 norm_name"""
        adapter = self.create_adapter(3)
        cfg = self.create_anti_config()
        args, kwargs = adapter.modify_smooth_args(cfg=cfg, norm_name="", linear_names="dummy", args=[], kwargs={})

        assert kwargs["is_shift"] is False

    def test_override_existing_kwargs(self):
        """测试覆盖已存在的 kwargs 参数"""
        adapter = self.create_adapter(3)
        cfg = self.create_anti_config()
        args, kwargs = adapter.modify_smooth_args(
            cfg=cfg, norm_name="norm.bias", linear_names="dummy", args=[], kwargs={"is_shift": False, "alpha": 1.0}
        )

        assert kwargs["is_shift"] is True  # 应覆盖原有值
        assert kwargs["alpha"] == cfg.alpha  # 应覆盖原有值
