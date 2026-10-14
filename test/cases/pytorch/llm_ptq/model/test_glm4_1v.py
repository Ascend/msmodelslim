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


class TestGetNormLinearSubgraph:
    """测试 get_norm_linear_subgraph 函数的不同场景"""

    @staticmethod
    def create_adapter(num_layers):
        """创建模拟的预训练模型"""
        from msmodelslim.pytorch.llm_ptq.model.glm.glm4_1v import GLM41VAdapter

        model = MagicMock()
        model.config = MagicMock()
        model.config.text_config.num_hidden_layers = num_layers
        adapter = GLM41VAdapter(model)
        return adapter

    def test_normal_case_with_three_layers(self):
        """测试当模型层为3层 覆盖全分支情形"""
        adapter = self.create_adapter(3)
        from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig

        result = adapter.get_norm_linear_subgraph(AntiOutlierConfig(anti_method='m2'))

        # 检查总共有3层 3*2=6
        assert len(result) == 6

        # 层0 input_layernorm：包含 q_proj, k_proj, v_proj
        key0_layernorm = 'model.language_model.layers.0.input_layernorm'
        assert key0_layernorm in result
        assert len(result[key0_layernorm]) == 3
        assert 'model.language_model.layers.0.self_attn.q_proj' in result[key0_layernorm]
        assert 'model.language_model.layers.0.self_attn.k_proj' in result[key0_layernorm]
        assert 'model.language_model.layers.0.self_attn.v_proj' in result[key0_layernorm]

        # 层0 post_layernorm：包含 gate_up_proj
        key0_post_layernorm = 'model.language_model.layers.0.post_attention_layernorm'
        assert key0_post_layernorm in result
        assert len(result[key0_post_layernorm]) == 1
        assert 'model.language_model.layers.0.mlp.gate_up_proj' in result[key0_post_layernorm]

        # 层2
        key2_layernorm = 'model.language_model.layers.2.input_layernorm'
        assert key2_layernorm in result
        assert len(result[key2_layernorm]) == 3
        key2_post_layernorm = 'model.language_model.layers.2.post_attention_layernorm'
        assert key2_post_layernorm in result
        assert len(result[key2_post_layernorm]) == 1

    def test_min_layer(self):
        """测试边界条件：模型最少层数"""
        adapter = self.create_adapter(1)
        from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig

        result = adapter.get_norm_linear_subgraph(AntiOutlierConfig(anti_method='m2'))

        assert len(result) == 2

        key = 'model.language_model.layers.0.input_layernorm'
        assert key in result
        assert len(result[key]) == 3
        assert 'model.language_model.layers.0.self_attn.q_proj' in result[key]
        assert 'model.language_model.layers.0.self_attn.k_proj' in result[key]
        assert 'model.language_model.layers.0.self_attn.v_proj' in result[key]

    def test_max_layers(self):
        """测试边界条件：模型最多层数"""
        adapter = self.create_adapter(999)
        from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig

        result = adapter.get_norm_linear_subgraph(AntiOutlierConfig(anti_method='m2'))

        # 检查最后一层
        key = 'model.language_model.layers.998.input_layernorm'
        assert key in result
        assert len(result[key]) == 3
        assert 'model.language_model.layers.998.self_attn.q_proj' in result[key]
        assert 'model.language_model.layers.998.self_attn.k_proj' in result[key]
        assert 'model.language_model.layers.998.self_attn.v_proj' in result[key]

    def test_invalid_layers_zero(self):
        """测试边界条件：模型层数小于条件值"""
        from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig

        adapter = self.create_adapter(0)
        with pytest.raises(ValueError) as excinfo:
            adapter.get_norm_linear_subgraph(AntiOutlierConfig(anti_method='m2'))
        assert "num_hidden_layers in text_config must be in the range 1 to 999." in str(excinfo.value).lower()

    def test_invalid_layers_exceed(self):
        """测试边界条件：模型层数大于条件值"""
        from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig

        adapter = self.create_adapter(1000)
        with pytest.raises(ValueError) as excinfo:
            adapter.get_norm_linear_subgraph(AntiOutlierConfig(anti_method='m2'))
        assert "num_hidden_layers in text_config must be in the range 1 to 999." in str(excinfo.value).lower()
