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
类型化工厂类：根据 config 的 TypeField 动态加载对象类并用 config 初始化

使用示例:
    from msmodelslim.utils.plugin.typed_factory import TypedFactory
    from msmodelslim.core.quant_service.interface import QuantServiceConfig
    from msmodelslim.core.quant_service import IQuantService

    # 创建工厂实例（entry_point_group 从 QuantServiceConfig 的 plugin_entry 读取）
    factory = TypedFactory[IQuantService](config_base_class=QuantServiceConfig)

    # 使用 config 创建对象，返回类型为 IQuantService
    service = factory.create(config, **extra_kwargs)
"""
from typing import Type, TypeVar, Generic  # noqa: E402

from pydantic import BaseModel  # noqa: E402

from msmodelslim.utils.exception import ToDoError, UnsupportedError  # noqa: E402
from msmodelslim.utils.logging import get_logger  # noqa: E402
from msmodelslim.utils.plugin.plugin_utils import load_plugin_component_class  # noqa: E402
from msmodelslim.utils.plugin.typed_config import TypedConfig  # noqa: E402

T = TypeVar('T', bound=object)


class TypedFactory(Generic[T]):
    """
    类型化工厂类：根据 config 的 TypeField 动态加载对象类并用 config 初始化

    使用示例:
        from msmodelslim.utils.plugin.typed_factory import TypedFactory
        from msmodelslim.core.quant_service.interface import QuantServiceConfig
        from msmodelslim.core.quant_service import IQuantService

        # 创建工厂实例（entry_point_group 从 config_base_class 的 plugin_entry 读取）
        factory = TypedFactory[IQuantService](config_base_class=QuantServiceConfig)
        service = factory.create(quant_service_config, **extra_kwargs)
    """

    def __init__(self, config_base_class: Type[BaseModel]):
        """初始化工厂，entry_point_group 从 config_base_class（须 @TypedConfig.plugin_entry）读取。"""
        self.config_base_class = config_base_class
        self.type_field = TypedConfig.detect_type_field(config_base_class)
        self.entry_point_group = getattr(config_base_class, "_entry_point_group", None)
        if not self.entry_point_group:
            raise ToDoError(
                f"Config base class {config_base_class.__name__} must be decorated with "
                f"@TypedConfig.plugin_entry(entry_point_group=...) to provide _entry_point_group."
            )

    def create(self, config: BaseModel, **kwargs) -> T:
        """根据 config 创建对象实例，config 须含 TypeField，**kwargs 传给 __init__。"""
        # 验证 config 类型
        if not isinstance(config, self.config_base_class):
            raise UnsupportedError(
                f"Config must be an instance of {self.config_base_class.__name__}, got {type(config).__name__}"
            )

        # 从 config 中获取类型字段值
        plugin_type = getattr(config, self.type_field, None)
        if not plugin_type:
            raise ToDoError(f"Attr {self.type_field} is required in the configuration")

        plugin_class = load_plugin_component_class(self.entry_point_group, plugin_type)
        try:
            instance = plugin_class(config, **kwargs)
        except TypeError:
            try:
                instance = plugin_class(config=config, **kwargs)
            except Exception as e:
                raise ToDoError(f"Failed to instantiate {plugin_class.__name__} with config. Error: {str(e)}. ") from e
        get_logger().debug(
            "[typed_factory] Successfully created %r instance from config with %r=%r",
            plugin_class.__name__,
            self.type_field,
            plugin_type,
        )
        return instance
