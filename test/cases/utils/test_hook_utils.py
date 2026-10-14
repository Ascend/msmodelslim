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
msmodelslim.utils.hook_util 模块的单元测试（pytest 版）
"""

from unittest.mock import Mock  # noqa: E402
import pytest  # noqa: E402

from msmodelslim.utils.hook_utils import (  # noqa: E402
    HookManager,
    add_before_hook,
    restore_target,
)
from msmodelslim.utils.hook_utils import (  # noqa: E402
    add_after_hook,  # noqa: F401
    add_error_hook,  # noqa: F401
    restore_all_hooks,  # noqa: F401
)  # noqa: F401


class TestHookManager:
    @staticmethod
    def test_add_before_hook_executes_before_function(mock_self):
        """测试before钩子在目标函数执行前被调用"""
        mock_hook = Mock()

        mock_self.manager.add_before_hook(mock_self.target, mock_hook)

        mock_self.test_class.test_method(1)

        mock_hook.assert_called_once()
        args, kwargs = mock_hook.call_args
        func, call_kwargs = args

        assert call_kwargs == {"a": 1, "b": 2}

    @staticmethod
    def test_add_after_hook_executes_after_function(mock_self):
        """测试after钩子在目标函数执行后被调用，并且能修改返回值"""
        execution_order = []

        def before_hook(*args, **kwargs):
            execution_order.append("before")

        def after_hook(func, kwargs, result):
            execution_order.append("after")
            return result * 2

        mock_self.manager.add_before_hook(mock_self.target, before_hook)
        mock_self.manager.add_after_hook(mock_self.target, after_hook)

        result = mock_self.test_class.test_method(1)

        assert execution_order == ["before", "after"]
        assert result == 6  # (1+2)*2=6

    @staticmethod
    def test_error_hook_triggers_on_exception(mock_self):
        """测试当目标函数抛出异常时，error钩子被调用"""
        mock_error_hook = Mock()

        def faulty_method(a, b):
            raise ValueError("Test error")

        mock_self.test_class.test_method = faulty_method

        mock_self.manager.add_error_hook(mock_self.target, mock_error_hook)

        with pytest.raises(ValueError, match="Test error"):
            mock_self.test_class.test_method(1, 2)

        mock_error_hook.assert_called_once()
        args, _ = mock_error_hook.call_args
        _, call_kwargs, error = args
        assert call_kwargs == {"a": 1, "b": 2}
        assert isinstance(error, ValueError)

    @staticmethod
    def test_restore_target_returns_original_function(mock_self):
        """测试restore_target方法能恢复原始函数，移除所有钩子"""
        mock_before = Mock()
        mock_after = Mock()

        mock_self.manager.add_before_hook(mock_self.target, mock_before)
        mock_self.manager.add_after_hook(mock_self.target, mock_after)
        mock_self.test_class.test_method(1)

        assert mock_before.called
        assert mock_after.called

        mock_self.manager.restore_target(mock_self.target)

        mock_before.reset_mock()
        mock_after.reset_mock()
        result = mock_self.test_class.test_method(1)

        assert not mock_before.called
        assert not mock_after.called
        assert result == 3

    @staticmethod
    def test_restore_all_hooks_removes_all(mock_self):
        """测试restore_all方法能恢复所有被hook的目标"""

        class AnotherClass:
            @staticmethod
            def another_method():
                return "original"

        another_instance = AnotherClass()
        second_target = (another_instance, "another_method")

        mock_self.manager.add_before_hook(mock_self.target, Mock())
        mock_self.manager.add_before_hook(second_target, Mock())

        mock_self.manager.restore_all()

        assert mock_self.manager.hooked_targets == {}
        assert mock_self.manager.original_functions == {}

    @staticmethod
    def test_global_functions_work_with_manager(mock_self):
        """测试全局函数接口能正确与管理器交互"""
        mock_hook = Mock()

        add_before_hook(mock_self.target, mock_hook)

        mock_self.test_class.test_method(1)
        assert mock_hook.called

        restore_target(mock_self.target)
        mock_hook.reset_mock()

        mock_self.test_class.test_method(1)
        assert not mock_hook.called

    @pytest.fixture
    def mock_self(self):
        mock = Mock()
        mock.manager = HookManager()

        class TestClass:
            @staticmethod
            def test_method(a, b=2):
                return a + b

        mock.test_class = TestClass()
        mock.target = (mock.test_class, "test_method")
        return mock
