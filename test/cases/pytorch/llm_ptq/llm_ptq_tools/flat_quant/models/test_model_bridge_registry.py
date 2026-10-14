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
# pylint: disable=abstract-method,useless-parent-delegation

import pytest
from unittest.mock import MagicMock
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.models.model_bridge_registry import (
    model_bridge_registry as global_registry,
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.models.model_bridge_registry import (
    ConfigMatcher,
    ModuleNameMatcher,
    CompositeMatcher,
    ModelBridgeRegistry,
    get_model_bridge,
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.models.model_bridge_registry import (
    ModelMatcher,  # noqa: F401
)  # noqa: F401
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.models.model_utils import ModelStructureBridge


# Dummy bridge for registration
def make_dummy_bridge():
    class DummyBridge(ModelStructureBridge):
        def __init__(self, model, config=None):
            super().__init__(model, config)

    return DummyBridge


# Dummy model/config for matcher
def make_dummy_model(config_attrs=None, module_names=None):
    class DummyConfig:
        pass

    class DummyModel:
        def __init__(self):
            self.config = DummyConfig()
            if config_attrs:
                for k, v in config_attrs.items():
                    setattr(self.config, k, v)
            self._modules = module_names or []

        def named_modules(self):
            return [(name, None) for name in self._modules]

    return DummyModel()


def test_config_matcher_exact_and_regex():
    # exact value
    matcher = ConfigMatcher({'model_type': 'qwen'})
    m = make_dummy_model({'model_type': 'qwen'})
    assert matcher.match(m)
    m2 = make_dummy_model({'model_type': 'other'})
    assert not matcher.match(m2)
    # regex
    matcher2 = ConfigMatcher({'model_type': '^qw.*'})
    m3 = make_dummy_model({'model_type': 'QWEN'})
    assert matcher2.match(m3)
    # list
    matcher3 = ConfigMatcher({'arch': ['A', 'B']})
    m4 = make_dummy_model({'arch': 'A'})
    assert matcher3.match(m4)
    m5 = make_dummy_model({'arch': 'C'})
    assert not matcher3.match(m5)
    # config is None
    m6 = MagicMock()
    m6.config = None
    assert not matcher.match(m6)
    # missing attr
    m7 = make_dummy_model({'foo': 'bar'})
    assert not matcher.match(m7)


def test_module_name_matcher_all_and_any():
    # match_all True
    matcher = ModuleNameMatcher(['foo', 'bar'], match_all=True)
    m = make_dummy_model(module_names=['foo', 'bar', 'baz'])
    assert matcher.match(m)
    m2 = make_dummy_model(module_names=['foo', 'baz'])
    assert not matcher.match(m2)
    # match_all False
    matcher2 = ModuleNameMatcher(['foo', 'bar'], match_all=False)
    assert matcher2.match(m)
    m3 = make_dummy_model(module_names=['baz'])
    assert not matcher2.match(m3)


def test_composite_matcher_and_or():
    m = make_dummy_model({'model_type': 'a'}, ['foo'])
    matcher1 = ConfigMatcher({'model_type': 'a'})
    matcher2 = ModuleNameMatcher(['foo'])
    # AND
    comp = CompositeMatcher([matcher1, matcher2], logic='AND')
    assert comp.match(m)
    comp2 = CompositeMatcher([matcher1, ModuleNameMatcher(['bar'])], logic='AND')
    assert not comp2.match(m)
    # OR
    comp3 = CompositeMatcher([matcher1, ModuleNameMatcher(['bar'])], logic='OR')
    assert comp3.match(m)
    # empty
    comp4 = CompositeMatcher([], logic='AND')
    assert not comp4.match(m)
    # logic异常
    with pytest.raises(ValueError):
        CompositeMatcher([matcher1], logic='XOR')


def test_model_bridge_registry_register_and_priority():
    registry = ModelBridgeRegistry()
    DummyBridge = make_dummy_bridge()
    matcher = ConfigMatcher({'model_type': 'a'})
    # 正常注册
    registry.register(DummyBridge, matcher, priority=1)
    assert registry._priority_registry[0][2] == DummyBridge
    # priority插入
    DummyBridge2 = make_dummy_bridge()
    registry.register(DummyBridge2, matcher, priority=2)
    assert registry._priority_registry[0][2] == DummyBridge2

    # 类型校验
    class NotBridge:
        pass

    with pytest.raises(TypeError):
        registry.register(NotBridge, matcher)


def test_model_bridge_registry_get_bridge_and_clear():
    registry = ModelBridgeRegistry()
    DummyBridge = make_dummy_bridge()
    matcher = ConfigMatcher({'model_type': 'a'})
    registry.register(DummyBridge, matcher, priority=1)
    m = make_dummy_model({'model_type': 'a'})
    bridge = registry.get_bridge(m)
    assert isinstance(bridge, DummyBridge)
    # 无匹配
    m2 = make_dummy_model({'model_type': 'b'})
    with pytest.raises(ValueError):
        registry.get_bridge(m2)
    # clear
    registry.clear()
    with pytest.raises(ValueError):
        registry.get_bridge(m)


def test_get_model_bridge_global():
    DummyBridge = make_dummy_bridge()
    matcher = ConfigMatcher({'model_type': 'gtest'})
    # 清理全局注册表
    global_registry.clear()
    global_registry.register(DummyBridge, matcher, priority=1)
    m = make_dummy_model({'model_type': 'gtest'})
    bridge = get_model_bridge(m)
    assert isinstance(bridge, DummyBridge)
    global_registry.clear()
