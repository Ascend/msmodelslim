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
import unittest

import pytest  # noqa: F401
from testing_utils.mock import mock_kia_library

mock_kia_library()

from msmodelslim.pytorch.llm_ptq.model.flux.flux import FluxAdapter  # noqa: E402
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig  # noqa: E402


class TestGetNormLinearSubgraph(unittest.TestCase):
    """测试 get_norm_linear_subgraph 函数的不同场景"""

    @staticmethod
    def create_adapter(num_layers):
        """创建模拟的预训练模型"""
        model = MagicMock()
        model.config = MagicMock()
        model.config.num_layers = num_layers
        adapter = FluxAdapter(model)
        return adapter

    def test_min_layer(self):
        """测试边界条件：模型最少层数"""
        adapter = self.create_adapter(1)
        result = adapter.get_norm_linear_subgraph(AntiOutlierConfig(anti_method='m4'))

        self.assertEqual(len(result), 6)
        key0_qkv = '0qkv_anti'
        self.assertIn(key0_qkv, result)
        self.assertEqual(len(result[key0_qkv]), 3)
        self.assertIn('transformer_blocks.0.attn.to_q', result[key0_qkv])
        self.assertIn('transformer_blocks.0.attn.to_k', result[key0_qkv])
        self.assertIn('transformer_blocks.0.attn.to_v', result[key0_qkv])

        key0_qkv_context = '0qkv_context_anti'
        self.assertIn(key0_qkv_context, result)
        self.assertEqual(len(result[key0_qkv_context]), 3)
        self.assertIn('transformer_blocks.0.attn.add_q_proj', result[key0_qkv_context])
        self.assertIn('transformer_blocks.0.attn.add_k_proj', result[key0_qkv_context])
        self.assertIn('transformer_blocks.0.attn.add_v_proj', result[key0_qkv_context])

        key0_out = '0out_anti'
        self.assertIn(key0_out, result)
        self.assertEqual(len(result[key0_out]), 1)
        self.assertIn('transformer_blocks.0.attn.to_out.0', result[key0_out])

        key0_out_context = '0out_context_anti'
        self.assertIn(key0_out_context, result)
        self.assertEqual(len(result[key0_out_context]), 1)
        self.assertIn('transformer_blocks.0.attn.to_add_out', result[key0_out_context])

        key0_ff = '0ff0_anti'
        self.assertIn(key0_ff, result)
        self.assertEqual(len(result[key0_ff]), 1)
        self.assertIn('transformer_blocks.0.ff.net.0.proj', result[key0_ff])

        key0_ff_context = '0ff0_context_anti'
        self.assertIn(key0_ff_context, result)
        self.assertEqual(len(result[key0_ff_context]), 1)
        self.assertIn('transformer_blocks.0.ff_context.net.0.proj', result[key0_ff_context])

    def test_num_layers_not_integer(self):
        """测试 double_layer_num 不是整数"""
        adapter = self.create_adapter("invalid")
        with self.assertRaises(TypeError) as context:
            adapter.get_norm_linear_subgraph(AntiOutlierConfig(anti_method='m4'))
        self.assertEqual(str(context.exception), "num_layers must be an integer.")

    def test_invalid_layers_zero(self):
        """测试边界条件：模型层数小于条件值"""
        adapter = self.create_adapter(0)
        with self.assertRaises(ValueError) as context:
            adapter.get_norm_linear_subgraph(AntiOutlierConfig(anti_method='m4'))
        self.assertEqual(str(context.exception), "num_layers must be in the range 1 to 999.")

    def test_invalid_layers_exceed(self):
        """测试边界条件：模型层数大于条件值"""
        adapter = self.create_adapter(1000)
        with self.assertRaises(ValueError) as context:
            adapter.get_norm_linear_subgraph(AntiOutlierConfig(anti_method='m4'))
        self.assertEqual(str(context.exception), "num_layers must be in the range 1 to 999.")


class TestModifySmoothArgs(unittest.TestCase):
    """测试 modify_smooth_args 函数的不同场景"""

    @staticmethod
    def create_adapter(num_layers):
        """创建模拟的预训练模型"""
        model = MagicMock()
        model.config = MagicMock()
        model.config.num_layers = num_layers
        adapter = FluxAdapter(model)
        return adapter

    @staticmethod
    def create_anti_config(method='m4'):
        cfg = AntiOutlierConfig(anti_method=method)
        return cfg

    def test_m4_method(self):
        """测试 anti_method 为 'm4' 时的逻辑"""
        adapter = self.create_adapter(3)
        cfg = self.create_anti_config()
        norm_name = "layer1"
        linear_names = "layer1_linear"
        args = [1, 2, 3]
        kwargs = {'param1': 'value1'}

        updated_args, updated_kwargs = adapter.modify_smooth_args(cfg, norm_name, linear_names, args, kwargs)

        # 验证 args 未被修改
        self.assertEqual(updated_args, args)
        # 验证 kwargs 是否正确更新
        self.assertEqual(updated_kwargs['alpha'], cfg.alpha)
        self.assertEqual(updated_kwargs['is_shift'], False)
        self.assertIn('param1', updated_kwargs)
        self.assertEqual(updated_kwargs['param1'], 'value1')

        assert kwargs["is_shift"] is False
        assert kwargs["alpha"] == cfg.alpha

    def test_non_m4_method(self):
        """测试 anti_method 不为 'm4' 时的逻辑"""
        adapter = self.create_adapter(3)
        cfg = self.create_anti_config(method='m1')
        norm_name = "layer1"
        linear_names = "layer1_linear"
        args = [1, 2, 3]
        kwargs = {'param1': 'value1'}

        updated_args, updated_kwargs = adapter.modify_smooth_args(cfg, norm_name, linear_names, args, kwargs)

        # 验证 args 未被修改
        self.assertEqual(updated_args, args)
        # 验证 kwargs 未被修改
        self.assertEqual(updated_kwargs, kwargs)

    def test_override_existing_kwargs(self):
        """测试覆盖已存在的 kwargs 参数"""
        adapter = self.create_adapter(3)
        cfg = self.create_anti_config()
        args, kwargs = adapter.modify_smooth_args(
            cfg=cfg, norm_name="norm.bias", linear_names="dummy", args=[], kwargs={"is_shift": True, "alpha": 1.0}
        )

        assert kwargs["is_shift"] is False  # 应覆盖原有值
        assert kwargs["alpha"] == cfg.alpha  # 应覆盖原有值
