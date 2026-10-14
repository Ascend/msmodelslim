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
msmodelslim.utils.plugin.typed_config 模块的单元测试（unittest TestCase + pytest 辅助）
"""

import importlib  # noqa: E402
import unittest  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import patch, MagicMock  # noqa: E402

from pydantic import BaseModel, ValidationError  # noqa: E402

from msmodelslim.utils.exception import ToDoError, UnsupportedError, SchemaValidateError  # noqa: E402
from msmodelslim.utils.plugin import TypedConfig  # noqa: E402
from msmodelslim.utils.plugin.plugin_utils import load_plugin_config_class  # noqa: E402

typed_config_module = importlib.import_module("msmodelslim.utils.plugin.typed_config")
plugin_utils_module = importlib.import_module("msmodelslim.utils.plugin.plugin_utils")


class TestTypedConfig(unittest.TestCase):
    """typed_config 模块测试样例统一收敛到一个类"""

    def test_detect_type_field_return_field_name_when_class_has_typefield(self):
        """当类中存在 TypeField 注解字段时，应返回该字段名"""

        class MyConfig(BaseModel):
            apiversion: TypedConfig.TypeField
            other: int = 0

        field_name = TypedConfig.detect_type_field(MyConfig)

        self.assertEqual("apiversion", field_name)

    def test_detect_type_field_raise_todo_error_when_no_typefield(self):
        """当类中不存在 TypeField 注解字段时，应抛出 ToDoError"""

        class MyConfig(BaseModel):
            apiversion: str

        with self.assertRaises(ToDoError) as cm:
            TypedConfig.detect_type_field(MyConfig)

        msg = str(cm.exception)
        self.assertIn("No type field found in MyConfig", msg)
        self.assertIn("TypedConfig.TypeField", msg)

    def test_load_plugin_config_class_return_class_when_entry_exists(self):
        """当 entry 存在且配置类符合规范（继承 TypedConfig 并使用 plugin_entry）时，应成功返回配置类（每次都重新加载，不使用缓存）"""

        @TypedConfig.plugin_entry(entry_point_group="test.group")
        class BaseCfg(TypedConfig):
            apiversion: TypedConfig.TypeField

        class PluginCfg(BaseCfg):
            pass

        class PluginComponent:
            pass

        def plugin_func():
            return PluginCfg, PluginComponent

        fake_entry = SimpleNamespace(
            name="v1",
            load=MagicMock(return_value=plugin_func),
        )

        with patch.object(
            plugin_utils_module,
            "get_entry_points",
            return_value=[fake_entry],
        ):
            cls = load_plugin_config_class(
                entry_point_group="test.group",
                plugin_type="v1",
            )

        self.assertIs(PluginCfg, cls)
        # 再次加载会重新调用 load（不使用缓存）
        with patch.object(
            plugin_utils_module,
            "get_entry_points",
            return_value=[fake_entry],
        ):
            cls2 = load_plugin_config_class(
                entry_point_group="test.group",
                plugin_type="v1",
            )
        self.assertIs(PluginCfg, cls2)
        # 验证 load 被调用了两次（因为不使用缓存）
        self.assertEqual(fake_entry.load.call_count, 2)

    def test_load_plugin_config_class_raise_todo_error_when_config_not_basemodel(self):
        """当配置类不是 BaseModel 子类时，应抛出 ToDoError（每次都重新加载，不使用缓存）"""

        class NotBaseModel:
            pass

        class PluginComponent:
            pass

        def plugin_func():
            return NotBaseModel, PluginComponent

        fake_entry = SimpleNamespace(
            name="v1",
            load=MagicMock(return_value=plugin_func),
        )

        with patch.object(
            plugin_utils_module,
            "get_entry_points",
            return_value=[fake_entry],
        ):
            with self.assertRaises(ToDoError) as cm:
                load_plugin_config_class(
                    entry_point_group="test.group",
                    plugin_type="v1",
                )

        msg = str(cm.exception)
        self.assertIn("Config class NotBaseModel is not a subclass of BaseModel", msg)

        # 再次调用时会重新尝试加载（不使用缓存）
        fake_entry.load.reset_mock()
        with patch.object(
            plugin_utils_module,
            "get_entry_points",
            return_value=[fake_entry],
        ):
            with self.assertRaises(ToDoError) as cm2:
                load_plugin_config_class(
                    entry_point_group="test.group",
                    plugin_type="v1",
                )

        msg2 = str(cm2.exception)
        self.assertIn("Config class NotBaseModel is not a subclass of BaseModel", msg2)
        # 验证 load 被再次调用（因为不使用缓存）
        fake_entry.load.assert_called_once()

    def test_load_plugin_config_class_raise_unsupported_error_when_no_entry_found(self):
        """当没有找到对应 entry 时，应抛出 UnsupportedError"""

        with patch.object(
            plugin_utils_module,
            "get_entry_points",
            return_value=[],
        ):
            with self.assertRaises(UnsupportedError) as cm:
                load_plugin_config_class(
                    entry_point_group="test.group",
                    plugin_type="v-not-exist",
                )

        msg = str(cm.exception)
        self.assertIn("No plugin found for type 'v-not-exist'", msg)
        self.assertIn("Please install plugin before using", msg)

    def test_load_plugin_config_class_raise_todo_error_when_entry_load_raises(self):
        """当 entry.load 抛出异常时，应抛出 ToDoError（每次都重新加载，不使用缓存）"""

        def _raise_error():
            raise RuntimeError("load failed")

        fake_entry = SimpleNamespace(
            name="v1",
            load=_raise_error,
        )

        with patch.object(
            plugin_utils_module,
            "get_entry_points",
            return_value=[fake_entry],
        ):
            with self.assertRaises(ToDoError) as cm:
                load_plugin_config_class(
                    entry_point_group="test.group",
                    plugin_type="v1",
                )

        msg = str(cm.exception)
        self.assertIn("load failed", msg)
        self.assertIn("failed to load", msg)

    def test_typed_config_return_plugin_instance_when_data_has_type_field(self):
        """当数据包含类型字段时，应返回插件子类实例"""

        @TypedConfig.plugin_entry(entry_point_group="test.group")
        class BaseCfg(TypedConfig):
            apiversion: TypedConfig.TypeField
            value: int = 0

        class PluginCfg(BaseCfg):
            extra: int = 1

        class PluginComponent:
            pass

        def plugin_func():
            return PluginCfg, PluginComponent

        fake_entry = SimpleNamespace(
            name="v1",
            load=MagicMock(return_value=plugin_func),
        )
        with patch.object(
            plugin_utils_module,
            "get_entry_points",
            return_value=[fake_entry],
        ):
            obj = BaseCfg.model_validate({"apiversion": "v1", "value": 10})

        # validator 应该被触发，返回的是插件子类实例
        self.assertIsInstance(obj, PluginCfg)
        self.assertIsInstance(obj, BaseCfg)  # PluginCfg 是 BaseCfg 的子类
        self.assertIsInstance(obj, TypedConfig)  # BaseCfg 继承自 TypedConfig
        self.assertEqual("v1", obj.apiversion)
        self.assertEqual(10, obj.value)

    def test_typed_config_raise_validation_error_when_missing_type_field(self):
        """当数据缺少类型字段时，会使用基类的验证，抛出验证错误"""

        @TypedConfig.plugin_entry(entry_point_group="test.group")
        class BaseCfg(TypedConfig):
            apiversion: TypedConfig.TypeField

        # 可能抛出 SchemaValidateError（项目中的 patch）或 ValidationError
        with self.assertRaises((SchemaValidateError, ValidationError)) as cm:
            BaseCfg.model_validate({})

        msg = str(cm.exception)
        self.assertIn("Field required", msg)

    def test_typed_config_model_validate_handles_non_dict_input(self):
        """当 model_validate 输入数据不是字典时，应使用基类的验证逻辑"""

        @TypedConfig.plugin_entry(entry_point_group="test.group")
        class BaseCfg(TypedConfig):
            apiversion: TypedConfig.TypeField

        # 非字典输入会使用基类的验证，可能抛出 SchemaValidateError 或 ValidationError
        with self.assertRaises((SchemaValidateError, ValidationError)):
            BaseCfg.model_validate("not-a-dict")

    def test_typed_config_validator_passes_all_data_to_plugin_class(self):
        """验证 validator 将所有原始数据传递给插件类"""

        @TypedConfig.plugin_entry(entry_point_group="test.group")
        class BaseCfg(TypedConfig):
            apiversion: TypedConfig.TypeField
            value: int = 0
            extra_field: str = "default"

        class PluginCfg(BaseCfg):
            new_field: int = 100

        class PluginComponent:
            pass

        def plugin_func():
            return PluginCfg, PluginComponent

        fake_entry = SimpleNamespace(
            name="v1",
            load=MagicMock(return_value=plugin_func),
        )
        with patch.object(
            plugin_utils_module,
            "get_entry_points",
            return_value=[fake_entry],
        ):
            obj = BaseCfg.model_validate({"apiversion": "v1", "value": 20, "extra_field": "custom", "new_field": 200})

        # validator 应该被触发
        self.assertIsInstance(obj, PluginCfg)
        self.assertEqual("v1", obj.apiversion)
        self.assertEqual(20, obj.value)
        self.assertEqual("custom", obj.extra_field)
        self.assertEqual(200, obj.new_field)

    def test_typed_config_validator_raises_error_when_plugin_not_found(self):
        """当找不到对应的插件类时，应抛出 UnsupportedError"""

        @TypedConfig.plugin_entry(entry_point_group="test.group")
        class BaseCfg(TypedConfig):
            apiversion: TypedConfig.TypeField

        with patch.object(
            plugin_utils_module,
            "get_entry_points",
            return_value=[],
        ):
            with self.assertRaises(UnsupportedError) as cm:
                BaseCfg.model_validate({"apiversion": "non-existent"})

        msg = str(cm.exception)
        self.assertIn("No plugin found for type 'non-existent'", msg)

    def test_typed_config_validator_handles_empty_type_field(self):
        """当类型字段值为空字符串时，不会加载插件，使用基类验证"""

        @TypedConfig.plugin_entry(entry_point_group="test.group")
        class BaseCfg(TypedConfig):
            apiversion: TypedConfig.TypeField

        # 空字符串被视为 falsy，不会触发插件加载，使用基类验证
        # 空字符串是有效的字符串值，所以会正常创建实例
        with patch.object(
            plugin_utils_module,
            "load_plugin_config_class",
        ) as mock_load:
            obj = BaseCfg.model_validate({"apiversion": ""})

        # 验证没有调用插件加载
        mock_load.assert_not_called()
        # 验证返回的是基类实例
        self.assertEqual(type(obj).__name__, "BaseCfg")
        self.assertIsInstance(obj, TypedConfig)
        self.assertEqual("", obj.apiversion)

    def test_typed_config_nested_field_validation(self):
        """测试嵌套 BaseModel 字段的验证，确保子字段能正常验证"""

        @TypedConfig.plugin_entry(entry_point_group="test.group")
        class StrategyConfig(TypedConfig):
            type: TypedConfig.TypeField
            value: int = 0

        class PluginStrategyConfig(StrategyConfig):
            extra: int = 1

        class PluginComponent:
            pass

        class TuningPlanConfig(BaseModel):
            strategy: StrategyConfig

        def plugin_func():
            return PluginStrategyConfig, PluginComponent

        # 创建假的 entry point 来模拟插件加载
        fake_entry = SimpleNamespace(
            name="plugin",
            load=MagicMock(return_value=plugin_func),
        )

        with patch.object(
            plugin_utils_module,
            "get_entry_points",
            return_value=[fake_entry],
        ):
            obj = TuningPlanConfig.model_validate({"strategy": {"type": "plugin", "value": 10}})

        # 验证嵌套字段能正常验证和创建实例
        # 注意：嵌套字段验证时，Pydantic 的验证机制会触发 model_validator
        # 应该能正常创建插件类实例
        self.assertIsInstance(obj.strategy, StrategyConfig)
        self.assertIsInstance(obj.strategy, TypedConfig)
        self.assertEqual("plugin", obj.strategy.type)
        self.assertEqual(10, obj.strategy.value)

    def test_plugin_entry_decorator_requires_typedconfig_subclass(self):
        """装饰器要求类必须继承 TypedConfig"""

        # 尝试在不继承 TypedConfig 的类上使用装饰器，应抛出 ToDoError
        with self.assertRaises(ToDoError) as cm:

            @TypedConfig.plugin_entry(entry_point_group="test.group")
            class InvalidCfg(BaseModel):
                apiversion: TypedConfig.TypeField

        msg = str(cm.exception)
        self.assertIn("must inherit from TypedConfig", msg)
        self.assertIn("InvalidCfg", msg)
        self.assertIn("@TypedConfig.plugin_entry", msg)

    def test_typed_config_without_decorator_works_as_normal_base_model(self):
        """没有使用装饰器的 TypedConfig 子类应作为普通的 BaseModel 工作"""

        class NormalCfg(TypedConfig):
            apiversion: str
            value: int = 0

        # 应该能正常创建实例，不会触发插件加载
        obj = NormalCfg.model_validate({"apiversion": "v1", "value": 10})
        self.assertIsInstance(obj, NormalCfg)
        self.assertIsInstance(obj, TypedConfig)
        self.assertEqual("v1", obj.apiversion)
        self.assertEqual(10, obj.value)

    def test_plugin_entry_requires_typefield(self):
        """使用 @TypedConfig.plugin_entry 装饰器但缺少 TypeField 时，应抛出 ToDoError"""

        # 使用 @TypedConfig.plugin_entry 装饰器但没有 TypeField 注解，应该报错
        # 错误会在装饰器执行时（类定义时）抛出
        with self.assertRaises(ToDoError) as cm:

            @TypedConfig.plugin_entry(entry_point_group="test.group")
            class InvalidCfg(TypedConfig):
                apiversion: str  # 不是 TypeField
                value: int = 0

        msg = str(cm.exception)
        self.assertIn("No type field found in InvalidCfg", msg)
        self.assertIn("TypedConfig.TypeField", msg)
