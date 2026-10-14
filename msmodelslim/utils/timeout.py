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
# pylint: disable=no-member

import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import wraps
from typing import Callable, TypeVar

from msmodelslim.utils.exception import ModelslimTimeoutError, ToDoError

_F = TypeVar("_F", bound=Callable)


def _is_signal_timeout_available() -> bool:
    """是否可使用 signal.SIGALRM 做超时（仅 Unix 主线程）。"""
    if sys.platform == "win32":
        return False
    if threading.current_thread() is not threading.main_thread():
        return False
    return hasattr(signal, "SIGALRM")


def _run_with_signal_timeout(seconds: float, func: Callable, *args, **kwargs):
    """在 Unix 主线程内使用 SIGALRM 执行带超时的调用。"""

    def _handler(signum, frame):
        raise ModelslimTimeoutError(f"Execution timed out after {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            return func(*args, **kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    finally:
        signal.signal(signal.SIGALRM, old_handler)


def _run_with_thread_timeout(seconds: float, func: Callable, *args, **kwargs):
    """使用线程池 + future.result(timeout) 做超时（跨平台；超时后线程仍会继续执行）。"""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=seconds)
        except FuturesTimeoutError as e:
            raise ModelslimTimeoutError(f"Execution timed out after {seconds} seconds") from e


def with_timeout(seconds: float, func: Callable, *args, **kwargs):
    """
    在限定秒数内执行 func(*args, **kwargs)，超时则抛出 ModelslimTimeoutError

    - Unix 主线程：使用 signal.SIGALRM，超时后会中断当前执行。
    - Windows 或非主线程：使用线程池 + result(timeout)，超时后主流程返回并抛错，被调函数所在线程可能仍在运行。

    Args:
        seconds: 超时秒数，须 > 0
        func: 无参或与 *args/**kwargs 兼容的可调用对象
        *args: 传给 func 的位置参数
        **kwargs: 传给 func 的关键字参数

    Returns:
        func 的返回值

    Raises:
        ModelslimTimeoutError: 超过 seconds 未返回时抛出
        ToDoError: seconds <= 0 时抛出，表示调用方配置错误
    """
    if seconds <= 0:
        # 非正超时时间属于调用方配置错误，用 ToDoError 归类
        raise ToDoError("timeout seconds must be positive")
    if _is_signal_timeout_available():
        return _run_with_signal_timeout(seconds, func, *args, **kwargs)
    return _run_with_thread_timeout(seconds, func, *args, **kwargs)


def timeout(seconds: float) -> Callable[[_F], _F]:
    """
    超时装饰器：被装饰函数在限定秒数内未返回则抛出 ModelslimTimeoutError
    可作为 utils 的通用机制，用于插件加载、外部调用等需要防止长时间占用的场景。

    - Unix 主线程：使用 signal 超时。
    - Windows 或非主线程：使用线程池超时（被装饰函数会在子线程中执行）。

    Args:
        seconds: 超时秒数

    Returns:
        装饰器
    """

    def decorator(f: _F) -> _F:
        @wraps(f)
        def wrapper(*args, **kwargs):
            return with_timeout(seconds, f, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
