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

from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn as nn

from testing_utils.mock import mock_kia_library

mock_kia_library()

from msmodelslim.pytorch.llm_ptq.model.base import DefaultModelAdapter  # noqa: E402
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig  # noqa: E402


class TestDefaultModelAdapter:
    """测试 get_norm_linear_subgraph 函数的不同场景"""

    @staticmethod
    def create_adapter(num_layers):
        """创建模拟的预训练模型"""
        model = MagicMock()
        model.config = MagicMock()
        model.config.num_hidden_layers = num_layers
        adapter = DefaultModelAdapter(model)
        return adapter

    def test_without_required_params(self):
        """测试缺少必要参数的情况"""
        adapter = self.create_adapter(3)
        with pytest.raises(ValueError) as excinfo:
            adapter.get_norm_linear_subgraph(AntiOutlierConfig(anti_method='m4'))
        assert "must be provided" in str(excinfo.value)

    @patch('msmodelslim.pytorch.llm_ptq.anti_outlier.graph_utils.extract_dag')
    def test_normal_case(self, mock_extract_dag):
        """测试正常情况"""
        # 设置模拟的 DAG 对象
        mock_dag = MagicMock()
        mock_dag.get_norm_linear_subgraph.return_value = {'norm1': ['linear1', 'linear2'], 'norm2': ['linear3']}
        mock_dag.get_linear_linear_subgraph.return_value = {'linear1': ['linear4']}
        mock_extract_dag.return_value = mock_dag

        adapter = self.create_adapter(3)
        dummy_input = torch.randn(1, 10, 768)
        norm_class = [nn.LayerNorm]

        result = adapter.get_norm_linear_subgraph(
            AntiOutlierConfig(anti_method='m4'), dummy_input=dummy_input, norm_class=norm_class
        )

        # 验证结果
        assert len(result) == 3
        assert 'norm1' in result
        assert 'norm2' in result
        assert 'linear1' in result
        assert result['norm1'] == ['linear1', 'linear2']
        assert result['norm2'] == ['linear3']
        assert result['linear1'] == ['linear4']

        # 验证 extract_dag 调用
        mock_extract_dag.assert_called_once_with(adapter.model, dummy_input, hook_nodes=norm_class, anti_method='m4')

    @patch('msmodelslim.pytorch.llm_ptq.anti_outlier.graph_utils.extract_dag')
    def test_without_linear_linear_subgraph(self, mock_extract_dag):
        """测试不需要 linear_linear_subgraph 的情况"""
        # 设置模拟的 DAG 对象
        mock_dag = MagicMock()
        mock_dag.get_norm_linear_subgraph.return_value = {'norm1': ['linear1', 'linear2']}
        mock_extract_dag.return_value = mock_dag

        adapter = self.create_adapter(3)
        dummy_input = torch.randn(1, 10, 768)
        norm_class = [nn.LayerNorm]

        result = adapter.get_norm_linear_subgraph(
            AntiOutlierConfig(anti_method='m1'),  # 使用非 m4/m6 方法
            dummy_input=dummy_input,
            norm_class=norm_class,
        )

        # 验证结果
        assert len(result) == 1
        assert 'norm1' in result
        assert result['norm1'] == ['linear1', 'linear2']

        # 验证 get_linear_linear_subgraph 没有被调用
        mock_dag.get_linear_linear_subgraph.assert_not_called()


class TestModifySmoothArgs:
    """测试 modify_smooth_args 函数的不同场景"""

    @staticmethod
    def create_adapter(num_layers):
        """创建模拟的预训练模型"""
        model = MagicMock()
        model.config = MagicMock()
        model.config.num_hidden_layers = num_layers
        adapter = DefaultModelAdapter(model)
        return adapter

    @staticmethod
    def create_anti_config(method='m4'):
        cfg = AntiOutlierConfig(anti_method=method)
        return cfg

    def test_default_behavior(self):
        """测试默认行为：不修改任何参数"""
        adapter = self.create_adapter(3)
        cfg = self.create_anti_config()
        original_args = [1, 2, 3]
        original_kwargs = {"key1": "value1", "key2": "value2"}

        args, kwargs = adapter.modify_smooth_args(
            cfg=cfg,
            norm_name="layer.norm",
            linear_names="dummy",
            args=original_args.copy(),
            kwargs=original_kwargs.copy(),
        )

        # 验证参数没有被修改
        assert args == original_args
        assert kwargs == original_kwargs
        assert "is_shift" not in kwargs
        assert "alpha" not in kwargs

    def test_with_empty_args(self):
        """测试空参数的情况"""
        adapter = self.create_adapter(3)
        cfg = self.create_anti_config()
        args, kwargs = adapter.modify_smooth_args(
            cfg=cfg, norm_name="layer.norm", linear_names="dummy", args=[], kwargs={}
        )

        # 验证空参数保持不变
        assert args == []
        assert kwargs == {}
        assert "is_shift" not in kwargs
        assert "alpha" not in kwargs

    def test_with_none_values(self):
        """测试包含None值的情况"""
        adapter = self.create_adapter(3)
        cfg = self.create_anti_config()
        original_args = [None, 1, None]
        original_kwargs = {"key1": None, "key2": "value2"}

        args, kwargs = adapter.modify_smooth_args(
            cfg=cfg,
            norm_name="layer.norm",
            linear_names="dummy",
            args=original_args.copy(),
            kwargs=original_kwargs.copy(),
        )

        # 验证None值保持不变
        assert args == original_args
        assert kwargs == original_kwargs
        assert "is_shift" not in kwargs
        assert "alpha" not in kwargs
