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

from unittest.mock import Mock

import pytest
from msmodelslim.utils.plugin import TypedConfig

from msmodelslim.utils.exception import ToDoError
from msmodelslim.utils.plugin import plugin_utils


@TypedConfig.plugin_entry(entry_point_group="unit.plugins")
class _DemoConfig(TypedConfig):
    plugin_type: TypedConfig.TypeField = "demo"


class _DemoComponent:
    pass


def test_register_plugin_exposes_config_and_component_in_registry():
    result = plugin_utils.register_plugin(lambda: (_DemoConfig, _DemoComponent))

    assert result == (_DemoConfig, _DemoComponent)
    assert plugin_utils.list_registered_plugin_types("unit.plugins") == ["demo"]
    assert plugin_utils.load_plugin_config_class("unit.plugins", "demo") is _DemoConfig
    assert plugin_utils.load_plugin_component_class("unit.plugins", "demo") is _DemoComponent


def test_register_plugin_rejects_invalid_getter_result():
    with pytest.raises(ToDoError, match="must return \\(config_class, component_class\\)"):
        plugin_utils.register_plugin(lambda: (_DemoConfig,))


def test_get_entry_points_selects_requested_group(monkeypatch):
    selected = [object()]
    fake_entry_points = Mock()
    fake_entry_points.select.return_value = selected
    monkeypatch.setattr(plugin_utils, "entry_points", lambda: fake_entry_points)

    assert plugin_utils.get_entry_points("unit.group") == selected
    fake_entry_points.select.assert_called_once_with(group="unit.group")
