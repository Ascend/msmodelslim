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

import importlib
import sys
import time

import pytest

from msmodelslim.utils.exception import ModelslimTimeoutError, ToDoError
from msmodelslim.utils.timeout import timeout as timeout_decorator
from msmodelslim.utils.timeout import with_timeout

timeout_mod = importlib.import_module("msmodelslim.utils.timeout")


def test_with_timeout_return_result_when_not_timeout():
    """当函数在超时时间内完成时，应返回原始结果"""

    def fast_func():
        return "ok"

    result = with_timeout(1.0, fast_func)
    assert result == "ok"


def test_with_timeout_raise_timeout_error_when_seconds_non_positive():
    """当 seconds <= 0 时，应立即抛出 ToDoError"""
    with pytest.raises(ToDoError, match="timeout seconds must be positive"):
        with_timeout(0, lambda: None)
    with pytest.raises(ToDoError, match="timeout seconds must be positive"):
        with_timeout(-1, lambda: None)


def test_with_timeout_raise_timeout_error_when_thread_timeout(monkeypatch):
    """线程池分支：超时时抛出 ModelslimTimeoutError"""
    monkeypatch.setattr(timeout_mod, "_is_signal_timeout_available", lambda: False)

    def slow_func():
        time.sleep(0.1)
        return "done"

    with pytest.raises(ModelslimTimeoutError, match="Execution timed out"):
        with_timeout(0.01, slow_func)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows 不支持 SIGALRM 信号分支")
def test_with_timeout_raise_timeout_error_when_signal_timeout(monkeypatch):
    """signal 分支：超时时抛出 ModelslimTimeoutError"""
    assert hasattr(timeout_mod.signal, "SIGALRM")
    monkeypatch.setattr(timeout_mod, "_is_signal_timeout_available", lambda: True)

    def slow_func():
        time.sleep(0.1)
        return "done"

    with pytest.raises(ModelslimTimeoutError, match="Execution timed out"):
        with_timeout(0.01, slow_func)


def test_timeout_decorator_raise_timeout_error_when_timeout(monkeypatch):
    """装饰器形式：超时时抛出 ModelslimTimeoutError"""
    monkeypatch.setattr(timeout_mod, "_is_signal_timeout_available", lambda: False)

    @timeout_decorator(0.01)
    def slow_func():
        time.sleep(0.1)
        return "done"

    with pytest.raises(ModelslimTimeoutError, match="Execution timed out"):
        slow_func()
