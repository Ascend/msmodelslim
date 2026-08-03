#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

import sys
import traceback
from importlib.metadata import entry_points
from typing import Type, Tuple, Dict, Callable, List, get_args

from pydantic import BaseModel

from msmodelslim.utils.exception import ToDoError, UnsupportedError, ModelslimTimeoutError
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.timeout import with_timeout

# 插件加载/执行超时秒数，防止恶意或异常插件长时间占用
PLUGIN_LOAD_TIMEOUT_SECONDS = 5


def get_entry_points(group_name: str):
    """获取 entry_points，兼容不同 Python 版本"""
    if sys.version_info >= (3, 10):
        return entry_points().select(group=group_name)
    return entry_points().get(group_name, [])


# 全局插件注册表: {entry_point_group: {plugin_type: (config_class, component_class)}}
_PLUGIN_REGISTRY: Dict[str, Dict[str, Tuple[Type[BaseModel], Type]]] = {}


def _register_plugin_pair(
    entry_point_group: str, plugin_type: str, config_class: Type[BaseModel], component_class: Type
) -> None:
    """内部接口：将配置类和组件类写入插件表，对外注册统一通过 register_plugin。"""
    if entry_point_group not in _PLUGIN_REGISTRY:
        _PLUGIN_REGISTRY[entry_point_group] = {}

    _PLUGIN_REGISTRY[entry_point_group][plugin_type] = (config_class, component_class)


def _plugin_type_from_config_class(config_class: Type[BaseModel], type_field: str) -> str:
    """从配置类的 type 字段取 plugin_type：优先 FieldInfo.default，否则从 Literal 取第一个值。"""
    if not hasattr(config_class, "model_fields") or type_field not in config_class.model_fields:
        raise ToDoError(f"Config class {config_class.__name__} has no field {type_field!r} (required for plugin_type).")
    field_info = config_class.model_fields[type_field]  # Pydantic v2 FieldInfo
    default = getattr(field_info, "default", None)  # 字段声明时的默认值，如 apiversion: Literal["x"] = "x"
    if default is not None:
        return str(default)
    # 无 default 时从 Literal 注解取第一个值
    ann = config_class.__annotations__.get(type_field)
    if ann is not None:
        args = get_args(ann)
        if args:
            return str(args[0])
    raise ToDoError(
        f"Config class {config_class.__name__} field {type_field!r} has no default and no Literal for plugin_type."
    )


def list_registered_plugin_types(entry_point_group: str) -> List[str]:
    """列出已在内存注册表中注册的插件 ``type`` 名称（不含尚未触发的 setuptools entry_point）。"""
    return sorted(_PLUGIN_REGISTRY.get(entry_point_group, {}).keys())


def register_plugin(plugin_getter: Callable[[], Tuple[Type[BaseModel], Type]]) -> Tuple[Type[BaseModel], Type]:
    """
    直接传入 get_plugin 函数完成注册。entry_point_group 与 plugin_type 从返回的
    配置类上读取（配置类须为 TypedConfig 子类且基类使用 @TypedConfig.plugin_entry 绑定）。

    Args:
        plugin_getter: get_plugin 函数，无参，返回 (Config 类, 组件类)

    Returns:
        (config_class, component_class) 元组
    """
    if not callable(plugin_getter) or isinstance(plugin_getter, type):
        raise ToDoError(f"Plugin getter must be a function, got {type(plugin_getter).__name__}")

    result = plugin_getter()
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise ToDoError(f"Plugin function must return (config_class, component_class), got {result!r}")
    config_class, component_class = result

    if not issubclass(config_class, BaseModel):
        raise ToDoError(f"Config class {config_class.__name__} is not a subclass of BaseModel")

    entry_point_group = getattr(config_class, "_entry_point_group", None)
    type_field = getattr(config_class, "_type_field", None)
    if entry_point_group is None or type_field is None:
        raise ToDoError(
            f"Config class {config_class.__name__} must inherit from a TypedConfig base "
            f"decorated with @TypedConfig.plugin_entry (has _entry_point_group and _type_field)."
        )
    plugin_type = _plugin_type_from_config_class(config_class, type_field)

    _register_plugin_pair(entry_point_group, plugin_type, config_class, component_class)
    get_logger().debug(
        "[plugin_utils] Registered plugin %r in group %r via register_plugin(getter)", plugin_type, entry_point_group
    )
    return config_class, component_class


def load_plugin_function(entry_point_group: str, plugin_type: str) -> Tuple[Type[BaseModel], Type]:
    """加载函数式插件，返回配置类和组件类并注册到插件表。"""
    # 首先检查注册表中是否已有
    if entry_point_group in _PLUGIN_REGISTRY:
        if plugin_type in _PLUGIN_REGISTRY[entry_point_group]:
            config_class, component_class = _PLUGIN_REGISTRY[entry_point_group][plugin_type]
            get_logger().debug("[plugin_utils] Load plugin %r from memory (group %r)", plugin_type, entry_point_group)
            return config_class, component_class
    # 从 entry_points 加载插件函数，由 register_plugin(getter) 从配置类读取 group/type 并注册
    # 同一 group 下同一 name 可能来自多个包，get_entry_points 会返回多个同名 entry，此处取第一个并告警
    candidates = [e for e in get_entry_points(entry_point_group) if e.name == plugin_type]
    if len(candidates) > 1:
        get_logger().warning(
            "[plugin_utils] Multiple packages register plugin name %r in group %r (count=%d), using first.",
            plugin_type,
            entry_point_group,
            len(candidates),
        )
    for entry in candidates:

        def _load_and_register(ep=entry):
            plugin_func = ep.load()
            return register_plugin(plugin_func)

        try:
            config_class, component_class = with_timeout(PLUGIN_LOAD_TIMEOUT_SECONDS, _load_and_register)
            # register_plugin 从 config_class 推导的 plugin_type 须与当前请求一致
            derived_type = _plugin_type_from_config_class(config_class, config_class._type_field)
            if derived_type != plugin_type:
                get_logger().warning(
                    "[plugin_utils] Plugin type mismatch: entry name %r vs config type %r", plugin_type, derived_type
                )
            get_logger().debug("[plugin_utils] Load plugin %r from group %r success!", plugin_type, entry_point_group)
            return config_class, component_class

        except ModelslimTimeoutError as e:
            error_msg = f"Plugin load/execute timed out after {PLUGIN_LOAD_TIMEOUT_SECONDS}s"
            get_logger().error(
                "[plugin_utils] Timeout loading plugin %r from group %r: %s",
                plugin_type,
                entry_point_group,
                e,
            )
            # 直接向上抛出 ModelslimTimeoutError，让调用方统一感知为超时错误
            raise ModelslimTimeoutError(error_msg) from e
        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            get_logger().error(
                "[plugin_utils] Failed to load plugin function %r from group %r: %r", plugin_type, entry_point_group, e
            )
            raise ToDoError(
                f"Plugin function for type '{plugin_type}' in group '{entry_point_group}' failed to load:\n{error_msg}"
            ) from e
    raise UnsupportedError(
        f"No plugin found for type '{plugin_type}' in group '{entry_point_group}'.",
        action="Please install plugin before using.",
    )


def load_plugin_config_class(entry_point_group: str, plugin_type: str) -> Type[BaseModel]:
    """加载插件配置类（插件须函数式注册，返回配置类与组件类元组）。"""
    config_class, _ = load_plugin_function(entry_point_group, plugin_type)
    return config_class


def load_plugin_component_class(entry_point_group: str, plugin_type: str) -> Type:
    """加载插件组件类（插件须函数式注册，返回配置类与组件类元组）。"""
    _, component_class = load_plugin_function(entry_point_group, plugin_type)
    return component_class
