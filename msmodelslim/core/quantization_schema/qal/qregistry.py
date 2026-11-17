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
# pylint: disable=logging-fstring-interpolation

import inspect
import logging
from typing import Type, Callable, Tuple, Any, Dict, Optional, List

from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.exception import ToDoError, UnsupportedError


class QAPI:
    """
    QAPI代表了一个API的声明，并持有了该声明所有的相关实现。

    实现以dispatch_key为key，实现函数为value，存储在impl_map中。

    """

    def __init__(self, func_name: str, dispatch_key: Any, signature: Tuple[Type]):
        self.func_name = func_name
        self.dispatch_key = dispatch_key
        self.signature = signature
        self.impl_map: Dict[Type, Callable] = {}

    def add_impl(self, impl: Callable, dispatch_key: Any) -> None:
        """
        添加一个实现到impl_map中。

        """
        if dispatch_key in self.impl_map:
            get_logger().warning(
                f"[Core] Overwriting existing implementation for {self.func_name} with dispatch key: {dispatch_key}"
            )

        self.impl_map[dispatch_key] = impl


class QFuncRegistry:
    """
    通用的函数注册和分发系统，支持任意函数名和签名。

    声明API时，通过装饰器@QFuncRegistry.register_api装饰特定函数进行注册，注册时需要指定dispatch_key和signature（可选）。

    实现API时，通过装饰器@QFuncRegistry.register装饰特定函数进行注册，注册时需要指定api_name、dispatch_key和func_signature（可选）。

    调用API时，可以通过QFuncRegistry.dispatch接口进行调用，该接口会根据dispatch_key分发到对应的实现函数。

    该设计参考了torch的kernel分发机制，但进行了一定程度的本地化。

    """

    _registered_api: Dict[str, QAPI] = {}

    @classmethod
    def register_api(
        cls, dispatch_key: Any, api_name: Optional[str] = None, signature: Optional[Tuple[Type]] = None
    ) -> Callable:
        """

        声明API时，通过装饰器进行注册。

        example:
            @QFuncRegistry.register_api(dispatch_key=(QDType, QParam))
            def quantize(tensor: torch.Tensor, q_param: QParam, q_dtype: QDType) -> QTensor:
                pass

        """

        def decorator(func: Callable) -> Callable:
            actual_api_name = api_name if api_name is not None else func.__name__

            func_signature = signature if signature is not None else tuple(inspect.signature(func).parameters.keys())

            if actual_api_name in cls._registered_api:
                raise ToDoError(
                    f"API '{actual_api_name}' is already registered.", action="Please avoid duplicate registration."
                )

            cls._registered_api[actual_api_name] = QAPI(actual_api_name, dispatch_key, func_signature)

            get_logger().debug("[Core] Register API %r with dispatch key: %r", actual_api_name, dispatch_key)

            return func

        return decorator

    @classmethod
    def register(cls, dispatch_key: Any, api_name: str) -> Callable:
        """
        实现API时，通过装饰器进行注册。

        example:
            @QFuncRegistry.register(dispatch_key=(QInt8DType, QPerTensorParam), api_name="quantize")
            def quantize_int8_pertensor_symmetric(tensor: torch.Tensor, q_param: QParam, q_dtype: QDType) -> QTensor:
                pass

        """

        def decorator(func: Callable) -> Callable:
            # 如果API还没有注册，报错
            if api_name not in cls._registered_api:
                raise ToDoError(
                    f"API '{api_name}' is not registered. ",
                    action="Please register the API first using QFuncRegistry.register_api.",
                )

            # 检查函数签名与QAPI声明的signature是否兼容
            qapi = cls._registered_api[api_name]
            func_signature = tuple(inspect.signature(func).parameters.keys())

            if func_signature != qapi.signature:
                raise ToDoError(
                    f"Function signature mismatch: "
                    f"API \"{api_name}\" has signature {qapi.signature}, "
                    f"but registered func has signature {func_signature},",
                    action="Please check function signature.",
                )

            # 添加实现到对应的QAPI中
            cls._registered_api[api_name].add_impl(func, dispatch_key)

            get_logger().debug(
                "[Core] Register API implementation %r for %r with dispatch key: %r",
                func.__name__,
                api_name,
                dispatch_key,
            )

            return func

        return decorator

    @classmethod
    def dispatch(cls, api_name: str, dispatch_key: Any, *args, **kwargs) -> Any:
        """
        分发函数调用到对应的注册函数。

        Args:
            api_name: 函数名
            dispatch_key: 分发key
            *args: 函数参数
            **kwargs: 函数关键字参数

        Raises:
            NotImplementedError: 如果相应的API没有注册或是相应的DispatchKey版本没有实现

        """
        if api_name not in cls._registered_api:
            raise UnsupportedError(
                f"API '{api_name}' is not registered, available APIs: {list(cls._registered_api.keys())}",
                action=f"Please choose one in {list(cls._registered_api.keys())}",
            )

        qapi = cls._registered_api[api_name]

        if dispatch_key not in qapi.impl_map:
            get_logger().error(
                f"[Core] No implementation found for API {api_name} with dispatch key: {dispatch_key},\n"
                f"available dispatch keys are:\n"
                f"%s",
                "\n".join(f"- {item}" for item in list(qapi.impl_map.keys())),
            )

            raise UnsupportedError(
                f"No implementation found for function '{api_name}' with dispatch key: {dispatch_key}",
                action=f"Please chose one in {list(qapi.impl_map.keys())}",
            )

        impl_func = qapi.impl_map[dispatch_key]
        return impl_func(*args, **kwargs)


class QABC:
    def __init__(self, abc_class: Type, dispatch_key: Any):
        self.abc_class = abc_class
        self.dispatch_key = dispatch_key
        self.impl_map: Dict[Any, Type] = {}

    def add_impl(self, impl: Type, dispatch_key: Any) -> None:
        """
        添加一个实现到impl_map中。

        """
        if dispatch_key in self.impl_map:
            logging.warning(
                f"Overwriting existing implementation for {self.abc_class} with dispatch key: {dispatch_key}"
            )

        self.impl_map[dispatch_key] = impl


class QABCRegistry:
    _registered_abc: Dict[Type, QABC] = {}

    @classmethod
    def register_abc(
        cls,
        dispatch_key: Any,
    ) -> Callable:
        def decorator(wrapp_cls: Type) -> Type:
            if wrapp_cls in cls._registered_abc:
                raise ToDoError(f"ABC {wrapp_cls} is already registered", action="Please remove one")
            cls._registered_abc[wrapp_cls] = QABC(abc_class=wrapp_cls, dispatch_key=dispatch_key)
            get_logger().debug("[Core] Register ABC %r with dispatch key: %r", wrapp_cls, dispatch_key)
            return wrapp_cls

        return decorator

    @classmethod
    def register(cls, dispatch_key: Any, abc_class: Type, *args, **kwargs) -> Callable:
        def decorator(wrapp_cls: Type) -> Type:
            if abc_class not in cls._registered_abc:
                raise ToDoError(f"ABC {abc_class} is not registered.", action=f"Please register {abc_class} first.")
            abc = cls._registered_abc[abc_class]
            if not issubclass(wrapp_cls, abc.abc_class):
                raise ToDoError(
                    f"Class {wrapp_cls.__name__} is not a subclass of {abc.abc_class}",
                    action=f"Please make sure {wrapp_cls.__name__} is a subclass of {abc.abc_class}",
                )
            abc.add_impl(wrapp_cls, dispatch_key)
            get_logger().debug(
                "[Core] Register ABC implementation %r for %r with dispatch key: %r",
                wrapp_cls.__name__,
                abc_class,
                dispatch_key,
            )
            return wrapp_cls

        return decorator

    @classmethod
    def multi_register(cls, dispatch_key: List[Any], abc_type: Type, *args, **kwargs) -> Callable:
        def decorator(wrapp_cls: Type) -> Type:
            if abc_type not in cls._registered_abc:
                raise ToDoError(f"ABC {abc_type} is not registered", action=f"Please register {abc_type} first")
            abc = cls._registered_abc[abc_type]
            if not issubclass(wrapp_cls, abc.abc_class):
                raise ToDoError(
                    f"Class {wrapp_cls.__name__} is not a subclass of {abc_type}",
                    action=f"Please make sure {wrapp_cls.__name__} is a subclass of {abc_type}",
                )
            for key in dispatch_key:
                abc.add_impl(wrapp_cls, key)
                get_logger().debug(
                    "[Core] Register ABC implementation %r for %r with dispatch key: %r",
                    wrapp_cls.__name__,
                    abc_type,
                    key,
                )
            return wrapp_cls

        return decorator

    @classmethod
    def create(cls, abc_type: Type, dispatch_key: Any, *args, **kwargs) -> Any:
        if abc_type not in cls._registered_abc:
            raise UnsupportedError(
                f"ABC {abc_type} is not registered, available ABCs: {list(cls._registered_abc.keys())}",
                action=f"Please choose one in {list(cls._registered_abc.keys())}",
            )

        abc = cls._registered_abc[abc_type]

        if dispatch_key not in abc.impl_map:
            get_logger().error(
                f"[Core] No implementation found for ABC {abc_type} with dispatch key: {dispatch_key},\n"
                f"available dispatch keys are:\n"
                f"%s",
                "\n".join(f"- {item}" for item in list(abc.impl_map.keys())),
            )
            raise UnsupportedError(
                f"No implementation found for ABC {abc_type} with dispatch key: {dispatch_key}",
                action=f"Please choose one in {list(abc.impl_map.keys())}",
            )

        return abc.impl_map[dispatch_key](*args, **kwargs)
