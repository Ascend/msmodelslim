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

import inspect
import logging
from functools import wraps
from logging import Logger
from contextlib import contextmanager
from typing_extensions import Any, Callable, Type, Union

from msmodelslim.utils.exception import SchemaValidateError, ToDoError


class MsgConst:
    """
    Class for log messages const
    """

    SPECIAL_CHAR = ["\n", "\r", "\u007f", "\b", "\f", "\t", "\u000b", "%08", "%0a", "%0b", "%0c", "%0d", "%7f"]


class DistributedFilter(logging.Filter):
    def filter(self, record):
        # DEBUG 级别显示所有 rank，其他级别只显示 rank 0
        # 避免torch在顶层自动注册 ，从而触发CANN环境检查，故放在行间并通过E402忽略pre-commit检测
        import torch.distributed as dist  # noqa: E402

        if record.levelno == logging.DEBUG:
            return True
        return dist.get_rank() == 0 if dist.is_initialized() else True


def filter_special_chars(func):
    @wraps(func)
    def func_level(msg, *args):
        for char in MsgConst.SPECIAL_CHAR:
            if isinstance(msg, str):
                msg = msg.replace(char, ' ')
        return func(msg, *args)

    return func_level


def filter_logger(cur_logger: Logger):
    setattr(cur_logger, 'critical', filter_special_chars(cur_logger.critical))
    setattr(cur_logger, 'debug', filter_special_chars(cur_logger.debug))
    setattr(cur_logger, 'error', filter_special_chars(cur_logger.error))
    setattr(cur_logger, 'info', filter_special_chars(cur_logger.info))
    setattr(cur_logger, 'warning', filter_special_chars(cur_logger.warning))

    # 添加分布式过滤器
    if not any(isinstance(f, DistributedFilter) for f in cur_logger.filters):
        cur_logger.addFilter(DistributedFilter())


def get_logger(name: str = ''):
    """
    获取指定名称的日志记录器。

    如果名称为空，返回当前全局日志记录器（初始为根日志记录器，可能已被logger_setter更改）。
    如果指定了名称，会获取或创建对应名称的日志记录器并应用特殊字符过滤功能。

    Args:
        name (str, optional): 日志记录器的名称。默认为空字符串。

    Returns:
        Logger: 配置好的日志记录器实例。

    Examples:
        >>> # 获取当前全局日志记录器（初始为根日志记录器，可能已被logger_setter更改）
        >>> current_logger = get_logger()

        >>> # 获取指定名称的日志记录器（如果已存在则返回已存在的）
        >>> custom_logger = get_logger("msmodelslim.utils.logging")

        >>> # 使用日志记录器
        >>> custom_logger.info("这是一条信息日志")
    """
    if not name:
        return logger
    cur_logger = logging.getLogger(name)
    filter_logger(cur_logger)
    return cur_logger


def get_root_logger():
    root_logger = logging.getLogger(__name__.split('.', maxsplit=1)[0])
    root_logger.propagate = False
    root_logger.setLevel(logging.INFO)
    filter_logger(root_logger)
    if not root_logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)
    return root_logger


logger = get_root_logger()

LOG_LEVEL = {
    "notset": logging.NOTSET,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARN,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.FATAL,
    "critical": logging.CRITICAL,
}

LOGGER_FUNC = {
    "debug": logger.debug,
    "info": logger.info,
    "warn": logger.warning,
    "warning": logger.warning,
    "error": logger.error,
    "critical": logger.critical,
}


def set_logger_level(level="info"):
    if not isinstance(level, str):
        raise SchemaValidateError(
            f"level must be str, not {type(level)}", action='Please make sure log level is a string'
        )
    if level.lower() in LOG_LEVEL:
        get_root_logger().setLevel(LOG_LEVEL.get(level.lower()))
    else:
        get_root_logger().warning("Set %r log level failed.", level)


def progress_bar(iterable, desc: str = None, total: int = -1, interval: int = 1):
    if total == -1 and hasattr(iterable, "__len__"):
        total = len(iterable)

    format_str = "" if desc is None else (desc + ": ")
    if isinstance(total, int) and total > 0:
        format_str += "[%d/{}]".format(total)
    else:
        format_str += "[%d]"

    if not (isinstance(interval, int) and interval > 0):
        interval = 1

    prev_terminator = logging.StreamHandler.terminator
    logging.StreamHandler.terminator = '\r'
    for item_id, item in enumerate(iterable, start=1):
        if item_id % interval == 0:
            logger.info(format_str, item_id)
        yield item
    logging.StreamHandler.terminator = prev_terminator
    logger.info("")


@contextmanager
def clean_output():
    """Context manager to temporarily disable log formatting for clean output"""
    # Get all loggers and their handlers
    loggers_handlers = []

    # Find all loggers with handlers
    for name in logging.Logger.manager.loggerDict:
        logger_obj = logging.getLogger(name)
        if logger_obj.handlers:
            for handler in logger_obj.handlers:
                if hasattr(handler, 'formatter') and handler.formatter:
                    loggers_handlers.append((handler, handler.formatter))
                    # Set a minimal formatter
                    handler.setFormatter(logging.Formatter('%(message)s'))

    try:
        yield
    finally:
        # Restore original formatters
        for handler, original_formatter in loggers_handlers:
            handler.setFormatter(original_formatter)


class LoggerSetter:
    """
    日志设置器，支持作为装饰器和上下文管理器使用。

    这个类可以为函数、类或代码块自动设置日志记录器。

    **装饰器场景**：
    当作为装饰器使用时，会为被装饰的对象获取或创建一个专用的日志记录器，
    记录器名称格式为：{prefix}[.{subfix}]。
    在对象执行期间，全局logger会被临时替换为该专用记录器，执行完毕后恢复。
    每次调用时都会根据被装饰的对象实时计算日志记录器路径。

    **上下文管理器场景**：
    当作为上下文管理器使用时，会为代码块获取或创建一个专用的日志记录器，
    记录器名称格式为：{prefix}[.{subfix}]。
    在代码块执行期间，全局logger会被临时替换为该专用记录器，执行完毕后恢复。
    每次使用时都会重新计算日志记录器路径。

    Args:
        prefix (str, optional): 日志记录器名称的前缀。默认为空字符串。
        subfix (str, optional): 日志记录器名称的后缀。默认为空字符串。

    Returns:
        LoggerSetter: 日志设置器实例，可以作为装饰器或上下文管理器使用。

    Raises:
        ToDoError: 当装饰器应用于不支持的对象类型时抛出。

    Examples:
        >>> # 装饰器场景 - 函数
        >>> @LoggerSetter(prefix="msmodelslim.utils.logging")
        >>> def my_function():
        >>>     get_logger().info("函数执行中...")  # 使用 msmodelslim.utils.logging 记录器
        >>>     return "结果"

        >>> # 装饰器场景 - 类
        >>> @LoggerSetter(prefix="msmodelslim.utils.logging", subfix="default")
        >>> class MyClass:
        >>>     def method1(self):
        >>>         get_logger().info("方法1执行中...")  # 使用 msmodelslim.utils.logging.default 记录器

        >>> # 上下文管理器场景 - 指定前缀
        >>> with LoggerSetter(prefix="msmodelslim.utils.logging"):
        >>>     get_logger().info("使用自定义记录器")  # 使用 msmodelslim.utils.logging 记录器
        >>> get_logger().info("恢复原来的记录器")  # 使用原来的记录器

        >>> # 上下文管理器场景 - 不指定前缀，使用调用模块名
        >>> with LoggerSetter():
        >>>     get_logger().info("使用调用模块的记录器")  # 使用当前模块名记录器

        >>> # 上下文管理器场景 - 指定记录器名称
        >>> with LoggerSetter(prefix="msmodelslim.utils.logging", subfix="database"):
        >>>     get_logger().info("数据库操作日志")  # 使用 msmodelslim.utils.logging.database 记录器
    """

    def __init__(self, prefix: str = '', subfix: str = ''):
        self.prefix = prefix
        self.subfix = subfix
        self._original_logger = None

    def __call__(self, obj: Union[Type[Any], Callable]):
        """
        装饰器调用方法，支持装饰函数和类
        """
        if inspect.isclass(obj):
            # 如果是类，直接遍历类的 __dict__ 来获取用户自定义方法
            for name, member in obj.__dict__.items():
                # 过滤掉内置属性
                if name in ['__class__', '__dict__', '__doc__', '__module__', '__weakref__']:
                    continue

                # 检查是否是静态方法或类方法
                if isinstance(member, staticmethod):
                    # 对于静态方法，需要保持其静态方法特性
                    wrapped_func = self._wrap_function_with_logger(member.__func__, self._get_target_logger(obj))
                    setattr(obj, name, staticmethod(wrapped_func))
                elif isinstance(member, classmethod):
                    # 对于类方法，需要保持其类方法特性
                    wrapped_func = self._wrap_function_with_logger(member.__func__, self._get_target_logger(obj))
                    setattr(obj, name, classmethod(wrapped_func))
                elif inspect.isfunction(member):
                    # 普通方法或函数
                    setattr(obj, name, self._wrap_function_with_logger(member, self._get_target_logger(obj)))
            return obj
        elif inspect.isfunction(obj):
            # 如果是函数，包装并设置日志记录器
            return self._wrap_function_with_logger(obj, self._get_target_logger(obj))
        else:
            raise ToDoError(
                f'decorator only support function or class, not {type(obj)}',
                action='Please make sure apply LoggerSetter to a function or class',
            )

    def __enter__(self):
        """上下文管理器进入方法"""
        global logger
        self._original_logger = logger
        logger = self._get_target_logger()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出方法"""
        global logger
        logger = self._original_logger

    @staticmethod
    def _wrap_function_with_logger(func: Callable, target_logger: Logger):
        """为函数包装日志记录器"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            # 直接管理全局 logger，使用传入的 target_logger
            global logger
            original_logger = logger
            try:
                logger = target_logger
                return func(*args, **kwargs)
            finally:
                logger = original_logger

        return wrapper

    @staticmethod
    def _get_calling_module_name():
        """获取调用模块的名称"""
        # 获取调用栈，跳过当前方法
        frame = inspect.currentframe()
        try:
            # 跳过内部调用链：_get_calling_module_name -> _get_logger_name -> _get_target_logger -> __enter__
            for _ in range(4):  # 跳过 _get_calling_module_name, _get_logger_name, _get_target_logger, __enter__
                frame = frame.f_back
                if frame is None:
                    break

            if frame is not None:
                module_name = frame.f_globals.get('__name__', '')
                if module_name and module_name != '__main__':
                    return module_name
        finally:
            # 清理 frame 引用以避免循环引用
            del frame

        # 如果无法获取模块名，返回根日志记录器名
        return __name__.split('.', maxsplit=1)[0]

    def _get_logger_name(self, obj: Union[Type[Any], Callable] = None):
        """获取日志记录器名称"""
        path = self.prefix
        if not path:
            if obj is not None:
                # 装饰器模式：使用被装饰对象的模块名
                path = obj.__module__
            else:
                # 上下文管理器模式：使用调用模块的名称
                path = self._get_calling_module_name()
        if self.subfix:
            path += f'.{self.subfix}'
        return path

    def _get_target_logger(self, obj: Union[Type[Any], Callable] = None):
        """获取目标日志记录器"""
        # 每次都重新计算日志记录器，不进行缓存
        logger_name = self._get_logger_name(obj)
        return get_logger(logger_name)


def logger_setter(prefix: str = '', subfix: str = ''):
    """
    日志设置器，支持作为装饰器和上下文管理器使用。

    这个函数可以为函数、类或代码块自动设置日志记录器。

    **装饰器场景**：
    当作为装饰器使用时，会为被装饰的对象获取或创建一个专用的日志记录器，
    记录器名称格式为：{prefix}[.{subfix}]。
    在对象执行期间，全局logger会被临时替换为该专用记录器，执行完毕后恢复。
    每次调用时都会根据被装饰的对象实时计算日志记录器路径。

    **上下文管理器场景**：
    当作为上下文管理器使用时，会为代码块获取或创建一个专用的日志记录器，
    记录器名称格式为：{prefix}[.{subfix}]。
    在代码块执行期间，全局logger会被临时替换为该专用记录器，执行完毕后恢复。
    每次使用时都会重新计算日志记录器路径。

    Args:
        prefix (str, optional): 日志记录器名称的前缀。默认为空字符串。
        subfix (str, optional): 日志记录器名称的后缀。默认为空字符串。

    Returns:
        LoggerSetter: 日志设置器实例，可以作为装饰器或上下文管理器使用。

    Raises:
        ToDoError: 当装饰器应用于不支持的对象类型时抛出。

    Examples:
        >>> # 装饰器场景 - 函数
        >>> @logger_setter(prefix="msmodelslim.utils.logging")
        >>> def my_function():
        >>>     get_logger().info("函数执行中...")  # 使用 msmodelslim.utils.logging 记录器
        >>>     return "结果"

        >>> # 装饰器场景 - 类
        >>> @logger_setter(prefix="msmodelslim.utils.logging", subfix="default")
        >>> class MyClass:
        >>>     def method1(self):
        >>>         get_logger().info("方法1执行中...")  # 使用 msmodelslim.utils.logging.default 记录器

        >>> # 上下文管理器场景 - 指定前缀
        >>> with logger_setter(prefix="msmodelslim.utils.logging"):
        >>>     get_logger().info("使用自定义记录器")  # 使用 msmodelslim.utils.logging 记录器
        >>> get_logger().info("恢复原来的记录器")  # 使用原来的记录器

        >>> # 上下文管理器场景 - 不指定前缀，使用调用模块名
        >>> with logger_setter():
        >>>     get_logger().info("使用调用模块的记录器")  # 使用当前模块名记录器

        >>> # 上下文管理器场景 - 指定记录器名称
        >>> with logger_setter(prefix="msmodelslim.utils.logging", subfix="database"):
        >>>     get_logger().info("数据库操作日志")  # 使用 msmodelslim.utils.logging.database 记录器
    """
    return LoggerSetter(prefix, subfix)
