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

from __future__ import annotations

import traceback
import types
from functools import wraps
from typing import TYPE_CHECKING, Tuple

from typing_extensions import Type

if TYPE_CHECKING:
    from typing import Any, Callable, Optional

from msmodelslim.utils.config import msmodelslim_config
from msmodelslim.utils.exception import UnexpectedError, ModelslimError
from msmodelslim.utils.logging import get_logger

ACTION_REPORT = (
    f'Please report this issue to the msModelSlim developers. '
    f'Repository: {msmodelslim_config.urls.repository} '
    f'Q&A: {msmodelslim_config.urls.question_and_answer}'
)

ACTION_REPORT_IF_TIPS_FAIL = (
    f'If the tips below cannot resolve your issue, please report this issue to the msModelSlim developers. '
    f'Repository: {msmodelslim_config.urls.repository} '
    f'Q&A: {msmodelslim_config.urls.question_and_answer}'
)


def get_action_report() -> str:
    if UnexpectedError.tips:
        return ACTION_REPORT_IF_TIPS_FAIL
    return ACTION_REPORT


def _rebuild_exception_handler_callable(
    func: Callable[..., Any],
    receiver: Optional[Any],
    handler: _ExceptionHandler,
) -> _ExceptionHandlerCallable:
    if receiver is not None:
        return _ExceptionHandlerCallable(types.MethodType(func, receiver), handler)
    return _ExceptionHandlerCallable(func, handler)


class _ExceptionHandlerCallable:
    """Picklable decorator result; replaces the former nested local wrapper."""

    def __init__(self, func: Callable[..., Any], handler: _ExceptionHandler):
        self._handler = handler
        if isinstance(func, types.MethodType):
            # Store unbound function + receiver explicitly; bound methods are awkward to pickle
            # and functools.wraps on them can pull in non-picklable getset_descriptor objects.
            self._func = func.__func__
            self._self: Optional[Any] = func.__self__
        else:
            self._func = func
            self._self = None

    def __reduce__(self):
        return _rebuild_exception_handler_callable, (self._func, self._self, self._handler)

    def __call__(self, *args, **kwargs):
        try:
            if self._self is not None:
                return self._func(self._self, *args, **kwargs)
            return self._func(*args, **kwargs)
        except ModelslimError:
            # 已是msModelSlim异常，直接抛出
            raise
        except self._handler._err_cls as e:
            # 关键字匹配（若设置）
            if not self._handler._keyword or self._handler._keyword in str(e):
                args_to_raise = self._handler._set_args if self._handler._set_args else (str(e),)
                raise self._handler._ms_err_cls(*args_to_raise, action=self._handler._action) from e
            raise
        except Exception:
            # 其他类型的异常直接抛出
            raise


class _ExceptionHandler:
    """Decorator and context manager for converting third-party exceptions."""

    def __init__(
        self,
        set_args_tuple: Tuple[Any, ...],
        err_cls_local: Type[Exception],
        ms_err_cls_local: Type[ModelslimError],
        keyword_local: str,
        action_local: str,
    ):
        self._set_args = set_args_tuple
        self._err_cls = err_cls_local
        self._ms_err_cls = ms_err_cls_local
        self._keyword = keyword_local
        self._action = action_local

    def __call__(self, func: Callable[..., Any]) -> _ExceptionHandlerCallable:
        return _ExceptionHandlerCallable(func, self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is None:
            return False
        # 已是msModelSlim异常，透传
        if isinstance(exc, ModelslimError):
            return False
        # 仅处理指定第三方异常
        if isinstance(exc, self._err_cls):
            if not self._keyword or self._keyword in str(exc):
                args_to_raise = self._set_args if self._set_args else (str(exc),)
                # 在上下文管理器中转换并抛出新的msModelSlim异常
                raise self._ms_err_cls(*args_to_raise, action=self._action) from exc
        # 其他异常不处理，交由上层
        return False


def exception_handler(
    *set_args,
    err_cls: Type[Exception] = Exception,
    ms_err_cls: Type[ModelslimError] = ModelslimError,
    keyword: str = '',
    action: str = '',
):
    """
    异常知识库构建装饰器/上下文管理器，用于将第三方异常的经验和解决方案沉淀到代码中。

    该工具用于构建异常知识库，将已知的第三方异常（如文件不存在、网络超时、权限错误等）
    转换为msModelSlim自定义异常，并提供针对性的解决建议。既可作为装饰器使用，也可作为
    上下文管理器使用，行为一致。

    Args:
        *set_args: 传递给自定义异常的参数，通常包含具体的错误描述
        err_cls: 要捕获的第三方异常类型，如FileNotFoundError、TimeoutError等
        ms_err_cls: 转换后的msModelSlim异常类型，如ConfigError、InvalidModelError等
        keyword: 异常消息中必须包含的关键字，用于精确匹配特定的异常场景
        action: 针对性的解决建议，如"请检查文件路径"、"请检查网络连接"等

    Returns:
        一个既可作为装饰器又可作为上下文管理器的对象

    Example:
        # 作为装饰器
        @exception_handler("配置文件不存在", ms_err_cls=ConfigError,
                           err_cls=FileNotFoundError,
                           keyword="No such file",
                           action="请检查配置文件路径是否正确")
        def load_config():
            pass

        # 作为上下文管理器
        with exception_handler("权限不足", ms_err_cls=SecurityError,
                               err_cls=PermissionError,
                               action="请使用管理员权限运行或检查文件权限"):
            write_output()


    Note:
        - 主要用于构建异常知识库，将运维经验沉淀到代码中
        - 通过精确的异常匹配和针对性的解决建议，提升用户体验
        - 随着项目发展，可以不断丰富异常知识库的内容
        - 建议为每种常见的第三方异常场景都建立相应的知识库条目
        - 装饰器与上下文两种用法行为一致；若遇到已是 ModelslimError 的异常将直接透传
    """
    return _ExceptionHandler(set_args, err_cls, ms_err_cls, keyword, action)


def exception_catcher(func):
    """
    应用级异常捕获装饰器，用于统一处理函数中的异常并记录日志。

    该装饰器主要用于应用入口或顶层函数，作为兜底机制捕获所有未被处理的异常，
    记录详细的错误日志，并将非msModelSlim异常转换为UnexpectedError，同时提供标准的错误报告信息。

    Args:
        func: 要装饰的函数

    Returns:
        装饰后的函数

    Example:
        # 应用入口使用：兜底捕获所有异常
        @exception_catcher
        def main():
            # 应用主逻辑
            pass

        # 顶层API函数使用
        @exception_catcher
        def api_endpoint():
            # API处理逻辑
            pass

        # 与组件级装饰器组合使用（exception_catcher应该在最外层）
        @exception_catcher
        @exception_handler(err_cls=ValueError, ms_err_cls=ConfigError)
        def complex_workflow():
            # 复杂的工作流程
            pass

    Note:
        - 适用于应用级别的兜底异常处理
        - 会记录所有异常的详细信息到日志中，包括完整的错误栈
        - 对于非msModelSlim异常，会转换为UnexpectedError并包含标准的错误报告链接
        - 建议在应用入口、API端点或顶层函数上使用此装饰器
        - 当与exception_handler组合使用时，exception_catcher应该放在最外层
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ModelslimError as e:
            # 记录msModelSlim异常
            get_logger().error('%r', e)
            raise
        except Exception as e:
            # 先转换为UnexpectedError，再记录日志和错误栈
            unexpected_error = UnexpectedError(action=get_action_report())
            get_logger().error('%r', unexpected_error)
            get_logger().error('Original exception: %r', e)
            get_logger().error('Traceback:\n%s', traceback.format_exc())
            raise unexpected_error from e

    return wrapper
