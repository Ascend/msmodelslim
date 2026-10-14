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

from pathlib import Path
from unittest.mock import Mock, patch
from importlib.metadata import EntryPoints, EntryPoint
import pytest

from msmodelslim.model.plugin_factory import PluginModelFactory, DEFAULT
from msmodelslim.model.plugin_factory.loader_interface import AdapterLoaderInterface
from msmodelslim.utils.exception import UnsupportedError, VersionError


class DummyAdapter:
    def __init__(self, model_type, model_path, trust_remote_code):
        self.model_type = model_type
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code


def make_entry_point(name):
    ep = Mock(spec=EntryPoint)
    ep.name = name
    ep.load.return_value = DummyAdapter
    return ep


@patch("msmodelslim.model.plugin_factory.plugin_model_factory.DependencyChecker.check_plugin")
@patch("msmodelslim.model.plugin_factory.plugin_model_factory.DependencyChecker.set_plugin")
@patch("msmodelslim.model.plugin_factory.plugin_model_factory.entry_points")
def test_create_valid_model(mock_plugin_entry_points, mock_set_plugin, mock_check_plugin):
    mock_check_plugin.return_value = None
    PluginModelFactory._model_map = None
    ep = make_entry_point("deepseek")
    eps = EntryPoints([ep])
    mock_plugin_entry_points.return_value.select.return_value = eps

    model = PluginModelFactory().create("deepseek", Path("/tmp/path"))

    ep.load.assert_called_once()
    mock_set_plugin.assert_called_once()
    mock_check_plugin.assert_called_once()
    assert isinstance(model, DummyAdapter)
    assert model.model_type == "deepseek"


@patch("msmodelslim.model.plugin_factory.plugin_model_factory.entry_points")
@patch("msmodelslim.model.plugin_factory.plugin_model_factory.get_logger")
@patch("msmodelslim.model.plugin_factory.plugin_model_factory.DependencyChecker.check_plugin")
@patch("msmodelslim.model.plugin_factory.plugin_model_factory.DependencyChecker.set_plugin")
def test_create_fallback_default(mock_set_plugin, mock_check_plugin, mock_logger, mock_adapter_entry_points):
    # Only default exists
    PluginModelFactory._model_map = None
    ep_default = make_entry_point(DEFAULT)
    eps = EntryPoints([ep_default])
    mock_adapter_entry_points.return_value.select.return_value = eps
    mock_check_plugin.return_value = None

    model = PluginModelFactory().create("not_exist", Path("/tmp/path"))

    mock_logger().warning.assert_called_once()
    mock_set_plugin.assert_called_once()
    assert model.model_type == "not_exist"


@patch("msmodelslim.model.plugin_factory.plugin_model_factory.entry_points")
def test_no_adapter_registered_should_raise(mock_entry_points):
    # No adapters at all
    PluginModelFactory._model_map = None
    eps = EntryPoints([])
    mock_entry_points.return_value.select.return_value = eps

    with pytest.raises(UnsupportedError):
        PluginModelFactory().create("not_exist", Path("/tmp/path"))


@patch("msmodelslim.model.plugin_factory.plugin_model_factory.DependencyChecker.check_plugin")
@patch("msmodelslim.model.plugin_factory.plugin_model_factory.entry_points")
def test_adapter_entrypoint_should_load_before_dependency_check(mock_adapter_entry_points, mock_check_plugin):
    PluginModelFactory._model_map = None

    ep_adapter = make_entry_point("deepseek")
    mock_adapter_entry_points.return_value.select.return_value = EntryPoints([ep_adapter])
    mock_check_plugin.side_effect = VersionError("dependency mismatch")

    with pytest.raises(VersionError):
        PluginModelFactory().create("deepseek", Path("/tmp/path"))

    ep_adapter.load.assert_called_once()


@patch("msmodelslim.model.plugin_factory.plugin_model_factory.entry_points")
def test_create_loader_entrypoint_should_use_loader_flow(mock_adapter_entry_points):
    PluginModelFactory._model_map = None

    class DummyLoader(AdapterLoaderInterface):
        def precheck(self, model_type: str, model_path: Path) -> None:
            return None

        def load(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
            return DummyAdapter(model_type, model_path, trust_remote_code)

    ep_loader = Mock(spec=EntryPoint)
    ep_loader.name = "loader-model"
    ep_loader.load.return_value = DummyLoader
    mock_adapter_entry_points.return_value.select.return_value = EntryPoints([ep_loader])

    model = PluginModelFactory().create("loader-model", Path("/tmp/path"))
    assert isinstance(model, DummyAdapter)
    assert model.model_type == "loader-model"
