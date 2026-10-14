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
# pylint: disable=redefined-outer-name

import pytest
from unittest.mock import MagicMock

from ascend_utils.common.knowledge_distill.utils import replace_module


class MockModule:
    def __init__(self):
        self._modules = {}
        self.update_parameters_name = MagicMock()

    def __setattr__(self, key, value):
        if key != "_modules":
            object.__setattr__(self, key, value)
            if hasattr(self, '_modules') and isinstance(self._modules, dict):
                self._modules[key] = value

    def __setitem__(self, key, value):
        self._modules[key] = value
        object.__setattr__(self, str(key), value)

    def __getitem__(self, key):
        return self._modules.get(key)


@pytest.fixture
def mock_network():
    class MockNetwork(MockModule):
        def __init__(self):
            super().__init__()
            self.conv1 = MockModule()
            self.conv2 = MockModule()
            self.features = MockModule()

    return MockNetwork()


def test_replace_module_given_valid_name_when_backend_mindspore_then_update_success(mock_network):
    new_module = MockModule()
    name = "conv1"
    replace_module(mock_network, name, new_module, backend="mindspore")
    assert getattr(mock_network, name) is new_module
    new_module.update_parameters_name.assert_called_once_with("conv1.")


def test_replace_module_given_invalid_name_when_component_missing_then_do_nothing(mock_network):
    original_conv1 = mock_network.conv1
    new_module = MockModule()
    name = "layers.conv4"
    replace_module(mock_network, name, new_module, backend="mindspore")
    assert getattr(mock_network, "conv1", None) == original_conv1


def test_replace_module_given_valid_name_when_backend_other_then_update_without_param_update(mock_network):
    new_module = MockModule()
    name = "conv2"
    replace_module(mock_network, name, new_module, backend="other")
    assert getattr(mock_network, name) is new_module
    new_module.update_parameters_name.assert_not_called()


def test_replace_module_given_nested_name_when_backend_other_then_update_success(mock_network):
    new_module = MockModule()
    name = "features.conv3"
    replace_module(mock_network, name, new_module, backend="other")
    parent = getattr(mock_network, "features")
    assert getattr(parent, "conv3") is new_module
    new_module.update_parameters_name.assert_not_called()


def test_replace_module_given_invalid_index_when_backend_mindspore_then_do_nothing(mock_network):
    new_module = MockModule()
    name = "invalid_layer.0.conv6"
    replace_module(mock_network, name, new_module, backend="mindspore")
    assert not hasattr(mock_network, "invalid_layer")
