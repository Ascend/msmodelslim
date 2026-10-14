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
# pylint: disable=consider-using-from-import,redefined-builtin

import pytest
import torch  # noqa: F401
import torch.nn as nn
from unittest.mock import Mock
from unittest.mock import patch, MagicMock  # noqa: F401

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.models.model_utils import (
    StructurePair,
    ModelStructureBridge,
    AttnNormLinearPair,
    AttnLinearLinearPair,
    MLPNormLinearPair,
    MLPLinearLinearPair,
    get_module_by_name,
    set_module_by_name,
    TransformerStructurePairVisitor,
    RunnerStopExecution,
    clone_module_hooks,
    remove_after_substring,
)


class TestUtilityFunctions:
    def test_get_module_by_name_single_level(self):
        model = nn.Sequential()
        model.layer1 = nn.Linear(10, 5)

        result = get_module_by_name(model, "layer1")

        assert result == model.layer1

    def test_get_module_by_name_nested(self):
        model = nn.Sequential()
        model.block = nn.Sequential()
        model.block.layer = nn.Linear(10, 5)

        result = get_module_by_name(model, "block.layer")

        assert result == model.block.layer

    def test_get_module_by_name_deep_nested(self):
        model = nn.Sequential()
        model.transformer = nn.Sequential()
        model.transformer.layer = nn.Sequential()
        model.transformer.layer.attention = nn.Linear(10, 5)

        result = get_module_by_name(model, "transformer.layer.attention")

        assert result == model.transformer.layer.attention

    def test_set_module_by_name_single_level(self):
        model = nn.Sequential()
        model.layer1 = nn.Linear(10, 5)
        new_layer = nn.Linear(10, 8)

        set_module_by_name(model, "layer1", new_layer, clone_hooks=False)

        assert model.layer1 == new_layer

    def test_set_module_by_name_nested(self):
        model = nn.Sequential()
        model.block = nn.Sequential()
        model.block.layer = nn.Linear(10, 5)
        new_layer = nn.Linear(10, 8)

        set_module_by_name(model, "block.layer", new_layer, clone_hooks=False)

        assert model.block.layer == new_layer

    def test_set_module_by_name_with_hooks(self):
        model = nn.Sequential()
        old_layer = nn.Linear(10, 5)
        model.layer1 = old_layer

        # Register hooks on old layer
        def forward_hook(module, input, output):
            return None

        def backward_hook(module, grad_input, grad_output):
            return None

        old_layer.register_forward_hook(forward_hook)
        old_layer.register_backward_hook(backward_hook)

        new_layer = nn.Linear(10, 8)

        set_module_by_name(model, "layer1", new_layer, clone_hooks=True)

        assert model.layer1 == new_layer
        # Check that hooks were copied (basic check for hook dictionaries)
        assert hasattr(new_layer, '_forward_hooks')
        assert hasattr(new_layer, '_backward_hooks')

    def test_set_module_by_name_without_hooks(self):
        model = nn.Sequential()
        old_layer = nn.Linear(10, 5)
        model.layer1 = old_layer
        new_layer = nn.Linear(10, 8)

        set_module_by_name(model, "layer1", new_layer, clone_hooks=False)

        assert model.layer1 == new_layer


class TestComplexScenarios:
    def test_structure_pair_integration(self):
        """测试完整的结构对集成场景"""
        config = {"model_type": "test"}
        bridge = ModelStructureBridge(Mock(), config)

        # 创建不同类型的结构对
        attn_norm_pair = AttnNormLinearPair(config, "norm1", ["q", "k", "v"], "layer.0")
        attn_linear_pair = AttnLinearLinearPair(config, "v", ["o"], "layer.0")
        mlp_norm_pair = MLPNormLinearPair(config, "norm2", ["gate", "up"], "layer.0")
        mlp_linear_pair = MLPLinearLinearPair(config, "up", ["down"], "layer.0")

        # 注册所有结构对
        bridge.register_structure_pair(attn_norm_pair)
        bridge.register_structure_pair(attn_linear_pair)
        bridge.register_structure_pair(mlp_norm_pair)
        bridge.register_structure_pair(mlp_linear_pair)

        # 验证注册结果
        registry = bridge.get_structure_pairs()
        assert "AttnNormLinearPair" in registry
        assert "AttnLinearLinearPair" in registry
        assert "MLPNormLinearPair" in registry
        assert "MLPLinearLinearPair" in registry

    def test_visitor_pattern_integration(self):
        """测试访问者模式集成"""

        class TestVisitor(TransformerStructurePairVisitor):
            def __init__(self):
                self.visited = []

            def visit_attn_norm_linear_pair(self, pair):
                self.visited.append("attn_norm")
                return "attn_norm_result"

            def visit_mlp_linear_linear_pair(self, pair):
                self.visited.append("mlp_linear")
                return "mlp_linear_result"

        visitor = TestVisitor()
        config = {}

        attn_pair = AttnNormLinearPair(config, "norm", ["linear"], "prefix")
        mlp_pair = MLPLinearLinearPair(config, "pre", ["post"], "prefix")

        result1 = attn_pair.accept(visitor)
        result2 = mlp_pair.accept(visitor)

        assert result1 == "attn_norm_result"
        assert result2 == "mlp_linear_result"
        assert visitor.visited == ["attn_norm", "mlp_linear"]

    def test_module_name_operations_integration(self):
        """测试模块名称操作的集成场景"""
        # 创建复杂的嵌套模型
        model = nn.Sequential()
        model.transformer = nn.Sequential()
        model.transformer.layers = nn.ModuleList()

        layer = nn.Sequential()
        layer.attention = nn.Sequential()
        layer.attention.query = nn.Linear(128, 128)
        layer.attention.key = nn.Linear(128, 128)
        layer.mlp = nn.Sequential()
        layer.mlp.gate = nn.Linear(128, 512)

        model.transformer.layers.append(layer)

        # 测试获取模块
        query_module = get_module_by_name(model, "transformer.layers.0.attention.query")
        assert isinstance(query_module, nn.Linear)
        assert query_module.in_features == 128

        # 测试设置模块
        new_query = nn.Linear(256, 256)
        set_module_by_name(model, "transformer.layers.0.attention.query", new_query)

        updated_module = get_module_by_name(model, "transformer.layers.0.attention.query")
        assert updated_module == new_query
        assert updated_module.in_features == 256


def test_structure_pair_str_repr_and_contain():
    class DummyPair(StructurePair):
        _name = "dummy"

    pair = DummyPair("src", ["a", "b"], "prefix")
    assert str(pair) == pair.name
    assert repr(pair) == pair.name
    assert pair.contain("a") is True
    assert pair.contain("c") is False


def test_structure_pair_init_type_error():
    class DummyPair(StructurePair):
        _name = "dummy"

    with pytest.raises(ValueError):
        DummyPair(123, ["a"], "prefix")
    with pytest.raises(ValueError):
        DummyPair("src", "notalist", "prefix")


def test_model_structure_bridge_register_type_error():
    bridge = ModelStructureBridge(Mock())
    with pytest.raises(TypeError):
        bridge.register_structure_pair("not_a_pair")


def test_model_structure_bridge_get_layers_and_index():
    bridge = ModelStructureBridge(Mock())
    bridge._layers_name = "layers"
    assert bridge.get_layers() == "layers"
    assert bridge.get_layer_by_index(2) == "layers.2"


def test_model_structure_bridge_analyze_structure_not_implemented():
    bridge = ModelStructureBridge(Mock())
    with pytest.raises(NotImplementedError):
        bridge.analyze_structure()


def test_clone_module_hooks_typeerror_and_attributeerror():
    class DummyModule:
        def __init__(self):
            self._forward_hooks = {1: lambda *a, **k: None}

        def register_forward_hook(self, fn):
            raise TypeError()

    src = DummyModule()
    tgt = DummyModule()
    # TypeError should be silently ignored
    clone_module_hooks(src, tgt)
    # AttributeError should be silently ignored
    src2 = DummyModule()
    delattr(src2, '_forward_hooks')
    clone_module_hooks(src2, tgt)


def test_remove_after_substring():
    s = "abcdeabc"
    assert remove_after_substring(s, "cd") == "abcd"
    assert remove_after_substring(s, "xyz") == s


def test_runner_stop_execution_exception():
    with pytest.raises(RunnerStopExecution):
        raise RunnerStopExecution()
