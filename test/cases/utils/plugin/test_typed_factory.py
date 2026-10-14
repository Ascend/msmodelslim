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
# pylint: disable=pointless-string-statement

"""
msmodelslim.utils.plugin.typed_factory 模块的单元测试（unittest TestCase + pytest 辅助）
"""

import unittest  # noqa: E402
from unittest.mock import patch  # noqa: E402

from pydantic import BaseModel  # noqa: E402

from msmodelslim.utils.exception import ToDoError, UnsupportedError  # noqa: E402
from msmodelslim.utils.plugin import TypedConfig, TypedFactory  # noqa: E402


@TypedConfig.plugin_entry(entry_point_group="test.group")
class TestPluginConfig(TypedConfig):
    """测试用配置类，继承 TypedConfig；单测中通过 mock 插件加载避免依赖真实 entry_points。"""

    kind: TypedConfig.TypeField
    value: int = 0


def _mock_load_plugin_config_class(entry_point_group: str, plugin_type: str):
    """单测用：对 test.group 直接返回 TestPluginConfig，不走真实插件加载。"""
    if entry_point_group == "test.group":
        return TestPluginConfig
    raise NotImplementedError("Only test.group is mocked in this test module.")


class TestTypedFactory(unittest.TestCase):
    """测试 TypedFactory 工厂类（一一对应 TypedFactory 代码类）"""

    def test_init_raise_todo_error_when_config_base_class_not_plugin_entry(self):
        """当 config_base_class 未经 @TypedConfig.plugin_entry 装饰时，应抛出 ToDoError"""

        class PlainConfig(BaseModel):
            kind: TypedConfig.TypeField
            value: int = 0

        with self.assertRaises(ToDoError) as cm:
            TypedFactory[object](config_base_class=PlainConfig)
        self.assertIn("must be decorated with", str(cm.exception))
        self.assertIn("plugin_entry", str(cm.exception))

    def test_create_raise_unsupported_error_when_config_type_mismatch(self):
        """当 config 不是指定基类实例时，应抛出 UnsupportedError"""

        class OtherConfig(BaseModel):
            x: int = 0

        factory = TypedFactory[object](config_base_class=TestPluginConfig)
        with self.assertRaises(UnsupportedError) as cm:
            factory.create(OtherConfig())
        msg = str(cm.exception)
        self.assertIn("Config must be an instance of TestPluginConfig", msg)

    @patch('msmodelslim.utils.plugin.typed_config.load_plugin_config_class', side_effect=_mock_load_plugin_config_class)
    def test_create_raise_todo_error_when_missing_type_field_value(self, mock_load_config):
        """当 config 中类型字段值为空时，应抛出 ToDoError"""

        factory = TypedFactory[object](config_base_class=TestPluginConfig)
        cfg = TestPluginConfig(kind="")

        with self.assertRaises(ToDoError) as cm:
            factory.create(cfg)

        msg = str(cm.exception)
        self.assertIn("Attr kind is required in the configuration", msg)

    @patch('msmodelslim.utils.plugin.typed_config.load_plugin_config_class', side_effect=_mock_load_plugin_config_class)
    def test_create_return_instance_when_plugin_accepts_config_positional(self, mock_load_config):
        """当插件类 __init__ 接受 (config, **kwargs) 形式时，应按关键字参数创建实例"""

        from msmodelslim.utils.plugin import typed_factory as typed_factory_module

        factory = TypedFactory[object](config_base_class=TestPluginConfig)
        cfg = TestPluginConfig(kind="service_a", value=5)

        class ServiceA:
            def __init__(self, config, **kwargs):
                self.config = config
                self.kwargs = kwargs

        def fake_load(group, plugin_type):
            self.assertEqual("test.group", group)
            self.assertEqual("service_a", plugin_type)
            return ServiceA

        with patch.object(typed_factory_module, "load_plugin_component_class", fake_load):
            inst = factory.create(cfg, flag=True)

        self.assertIsInstance(inst, ServiceA)
        self.assertIs(cfg, inst.config)
        self.assertEqual({"flag": True}, inst.kwargs)

    @patch('msmodelslim.utils.plugin.typed_config.load_plugin_config_class', side_effect=_mock_load_plugin_config_class)
    def test_create_return_instance_when_plugin_accepts_config_keyword(self, mock_load_config):
        """当插件类 __init__ 接受 config 关键字参数时，应按照关键字参数创建实例"""

        from msmodelslim.utils.plugin import typed_factory as typed_factory_module

        factory = TypedFactory[object](config_base_class=TestPluginConfig)
        cfg = TestPluginConfig(kind="service_b", value=7)

        class ServiceB:
            def __init__(self, config: TestPluginConfig, **kwargs):
                self.config = config
                self.kwargs = kwargs

        def fake_load(group, plugin_type):
            self.assertEqual("test.group", group)
            self.assertEqual("service_b", plugin_type)
            return ServiceB

        with patch.object(typed_factory_module, "load_plugin_component_class", fake_load):
            inst = factory.create(cfg, x="x", y=2)

        self.assertIsInstance(inst, ServiceB)
        self.assertIs(cfg, inst.config)
        self.assertEqual({"x": "x", "y": 2}, inst.kwargs)

    @patch('msmodelslim.utils.plugin.typed_config.load_plugin_config_class', side_effect=_mock_load_plugin_config_class)
    def test_create_raise_todo_error_when_plugin_accepts_only_field_args(self, mock_load_config):
        """当插件类既不接受 (config, **kwargs) 也不接受 (config=config, **kwargs) 时，应抛出 ToDoError"""

        from msmodelslim.utils.plugin import typed_factory as typed_factory_module

        factory = TypedFactory[object](config_base_class=TestPluginConfig)
        cfg = TestPluginConfig(kind="service_c", value=3)

        class ServiceC:
            def __init__(self, kind: str, value: int, **kwargs):
                self.kind = kind
                self.value = value
                self.kwargs = kwargs

        def fake_load(group, plugin_type):
            self.assertEqual("test.group", group)
            self.assertEqual("service_c", plugin_type)
            return ServiceC

        with patch.object(typed_factory_module, "load_plugin_component_class", fake_load):
            with self.assertRaises(ToDoError) as cm:
                factory.create(cfg, extra="ok")
        self.assertIn("Failed to instantiate", str(cm.exception))
        self.assertIn("ServiceC", str(cm.exception))
