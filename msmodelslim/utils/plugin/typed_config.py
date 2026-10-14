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
# pylint: disable=bad-staticmethod-argument,pointless-string-statement

"""
类型化配置工具，提供基类和装饰器协作实现插件式数据类

使用示例:
    from msmodelslim.utils.plugin import TypedConfig

    @TypedConfig.plugin_entry(entry_point_group="msmodelslim.quant_config.plugins")
    class QuantConfig(TypedConfig):
        apiversion: TypedConfig.TypeField  # 通过类型注解指定为类型字段
        spec: Any = None
"""
from typing import Type, cast, Any, Annotated, get_args, get_origin, ClassVar  # noqa: E402

from pydantic import BaseModel, ConfigDict, TypeAdapter, model_validator  # noqa: E402

from msmodelslim.utils.exception import ToDoError  # noqa: E402
from msmodelslim.utils.plugin.plugin_utils import load_plugin_config_class  # noqa: E402


class TypedConfig(BaseModel):
    """
    插件式数据类的基类，使用 BaseModel 校验机制

    插件基类应继承此类，并通过 @TypedConfig.plugin_entry 装饰器绑定 entry_point_group
    从插件模块加载插件类并输出子类实例
    """

    # 类型字段标记
    TYPE_FIELD_MARKER: ClassVar[str] = "plugin_type_field"

    # 类型字段类型别名
    TypeField: ClassVar[type] = Annotated[str, TYPE_FIELD_MARKER]

    # 允许任意额外属性
    model_config = ConfigDict(extra='allow')

    @staticmethod
    def detect_type_field(cls: Type[BaseModel]) -> str:
        """
        通过类型注解检测类中的类型字段

        只检查使用 TypeField 注解的字段

        Args:
            cls: 要检测的类（可以是 TypedConfig 子类或任意 BaseModel 子类）

        Returns:
            类型字段名
        """

        annotations = getattr(cls, '__annotations__', {})

        for field_name, annotation in annotations.items():
            if field_name.startswith('_'):
                continue

            # 检查是否是 Annotated[str, TYPE_FIELD_MARKER] 类型
            origin = get_origin(annotation)
            if origin is Annotated:
                args = get_args(annotation)
                if args and len(args) >= 2 and TypedConfig.TYPE_FIELD_MARKER in args[1:]:
                    return field_name

        raise ToDoError(
            f"No type field found in {cls.__name__}. ",
            action="Please use TypedConfig.TypeField annotation to mark the type field, "
            "e.g., 'apiversion: TypedConfig.TypeField'",
        )

    @staticmethod
    def plugin_entry(entry_point_group: str):
        """
        静态方法装饰器：为插件基类绑定插件入口点组（entry_point_group）

        插件基类应继承 TypedConfig 类，通过此装饰器绑定 entry_point_group，
        用于从插件模块动态加载插件类。

        Args:
            entry_point_group: entry_point 组名，例如 "msmodelslim.quant_config.plugins"

        Returns:
            装饰器函数

        使用示例:
            from msmodelslim.utils.plugin import TypedConfig

            @TypedConfig.plugin_entry(entry_point_group="msmodelslim.quant_config.plugins")
            class QuantConfig(TypedConfig):
                apiversion: TypedConfig.TypeField  # 通过类型注解指定为类型字段
                spec: Any = None
        """

        def decorator(cls: Type[TypedConfig]) -> Type[TypedConfig]:
            """装饰器内部函数，绑定 entry_point_group 到类属性"""
            if not issubclass(cls, TypedConfig):
                raise ToDoError(
                    f"Class {cls.__name__} must inherit from TypedConfig to use @TypedConfig.plugin_entry decorator"
                )

            # 如果使用了 plugin_entry 装饰器，必须能检测到 type_field，否则报错
            # detect_type_field 本身会抛出 ToDoError，直接让它抛出即可
            type_field = TypedConfig.detect_type_field(cls)

            # 绑定 entry_point_group 和 type_field 到类属性，避免在运行时重复检测
            cls._entry_point_group = entry_point_group
            cls._type_field = type_field
            return cls

        return decorator

    @model_validator(mode='wrap')
    @classmethod
    def _validate_plugin(cls: Type['TypedConfig'], value: Any, handler: Any) -> 'TypedConfig':
        """
        使用 model_validator mode=wrap 根据类型字段动态加载插件类

        如果类绑定了 entry_point_group 且数据包含类型字段，
        则从插件模块加载对应的插件类并使用 TypeAdapter 创建子类实例

        Args:
            value: 输入的数据值（dict、BaseModel 或其他）
            handler: pydantic 提供的原始验证处理器

        Returns:
            TypedConfig 实例（可能是插件子类实例）
        """
        # 插件类（_entry_point_group 不在 __dict__ 中）直接使用原始 handler，避免递归
        if '_entry_point_group' not in cls.__dict__:
            return cast('TypedConfig', handler(value))

        # 基类：如果使用了 @TypedConfig.plugin_entry 装饰器，_entry_point_group 和 _type_field 一定在 __dict__ 中
        # 装饰器已经检查过并缓存了这些值，直接使用即可
        entry_point_group = cls._entry_point_group
        type_field = cls._type_field

        # 从数据中获取类型字段值
        if isinstance(value, dict):
            plugin_type = value.get(type_field)
        elif isinstance(value, BaseModel):
            plugin_type = getattr(value, type_field, None)
        else:
            plugin_type = None

        # 如果数据中没有类型字段值，让 pydantic 的验证来处理（会抛出字段缺失错误）
        # 或者如果类型字段值为空字符串，也使用基类验证
        if not plugin_type:
            return cast('TypedConfig', handler(value))

        # 加载插件配置类（load_plugin_config_class 已经验证了配置类是 BaseModel 的子类，找不到会抛出异常）
        plugin_class = load_plugin_config_class(entry_point_group, plugin_type)

        # 若返回的就是当前类（如单测 mock 直接返回本类），直接走默认校验，避免递归
        if plugin_class is cls:
            return cast('TypedConfig', handler(value))

        # 使用 TypeAdapter 来验证并创建插件类实例
        # TypeAdapter 可以处理 dict、BaseModel 等输入，并返回对应的实例
        adapter = TypeAdapter(plugin_class)
        instance = adapter.validate_python(value)

        # 类型转换：plugin_class 是 cls 的子类，instance 是 plugin_class 的实例，因此也是 TypedConfig 的实例
        return cast('TypedConfig', instance)
