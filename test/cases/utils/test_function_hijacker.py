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
# pylint: disable=consider-using-generator,keyword-arg-before-vararg,pointless-string-statement

"""
msmodelslim.utils.function_hijacker 模块的单元测试
"""

import unittest  # noqa: E402
from unittest.mock import patch, MagicMock  # noqa: E402

from msmodelslim.utils.function_hijacker import (  # noqa: E402
    _get_target_id,
    _create_kwargs_wrapper,
    FunctionHijacker,
    hijack_function,
    restore_function,
    restore_all_hijacked,
    get_original_function,
    _hijacker,
)


def original_func_for_test(a, b=10, c=None):
    """用于测试的原始函数"""
    return a + b + (c or 0)


def replacement_func_for_test(a, b=10, c=None):
    """用于测试的替换函数"""
    return a * b * (c or 1)


def original_func_with_args_kwargs(*args, **kwargs):
    """用于测试带*args和**kwargs的原始函数"""
    return sum(args) + sum(kwargs.values())


def replacement_func_with_args_kwargs(**kwargs):
    """用于测试带*args和**kwargs的替换函数"""
    raw_args = kwargs.get('_raw_args', ())
    if raw_args:
        args_sum = sum([x for x in raw_args if isinstance(x, (int, float))])
    else:
        args_sum = sum([x for x in kwargs.get('args', ()) if isinstance(x, (int, float))])

    kwargs_sum = sum([v for v in kwargs.get('kwargs', {}).values() if isinstance(v, (int, float))])
    other_sum = sum(
        [v for k, v in kwargs.items() if k not in ['args', 'kwargs', '_raw_args'] and isinstance(v, (int, float))]
    )

    return args_sum + kwargs_sum + other_sum


def replacement_func_for_inspection_error(**kwargs):
    """用于测试检查错误的替换函数"""
    raw_args = kwargs.get('_raw_args', ())
    numeric_args = [x for x in raw_args if isinstance(x, (int, float))]
    numeric_kwargs = [v for k, v in kwargs.items() if k != '_raw_args' and isinstance(v, (int, float))]
    return sum(numeric_args) + sum(numeric_kwargs)


def replacement_func_for_bind_error(**kwargs):
    """用于测试绑定错误的替换函数"""
    raw_args = kwargs.get('_raw_args', ())
    if len(raw_args) >= 2:
        return raw_args[0] + raw_args[1]
    else:
        return kwargs.get('a', 0) + kwargs.get('b', 0)


def replacement_func_for_complex_func(**kwargs):
    """用于测试复杂函数的替换函数"""
    raw_args = kwargs.get('_raw_args', ())
    kw_arg = kwargs.get('kw_arg', 1)
    extra = kwargs.get('extra', 0)

    numeric_raw_args = [x for x in raw_args if isinstance(x, (int, float))]

    if raw_args:
        first_arg = raw_args[0]
        return first_arg * kw_arg * (sum(numeric_raw_args) + 1) * (extra + 1)
    else:
        return 0


def complex_func_for_test(a: int, b: str = "default", *args, **kwargs) -> str:
    """用于测试复杂函数的原始函数"""
    return f"{a}-{b}"


def replacement_func_for_complex_test(**kwargs):
    """用于测试复杂函数的替换函数"""
    a = kwargs.get('a', 0)
    b = kwargs.get('b', 'default')
    return f"replaced-{a}-{b}"


class TestTargetId(unittest.TestCase):
    """测试_get_target_id函数"""

    def test_get_target_id_basic(self):
        """测试基本的target_id生成"""

        class TestContainer:
            def test_func(self):
                pass

        container = TestContainer()
        target = (container, 'test_func')
        target_id = _get_target_id(target)

        self.assertIsInstance(target_id, str)
        self.assertIn(f"{id(container)}:", target_id)
        self.assertIn(":test_func", target_id)

    def test_get_target_id_different_objects(self):
        """测试不同对象的target_id唯一性"""

        class TestContainer:
            def func1(self):
                pass

            def func2(self):
                pass

        container1 = TestContainer()
        container2 = TestContainer()

        target1 = (container1, 'func1')
        target2 = (container2, 'func1')
        target3 = (container1, 'func2')

        id1 = _get_target_id(target1)
        id2 = _get_target_id(target2)
        id3 = _get_target_id(target3)

        self.assertNotEqual(id1, id2)
        self.assertNotEqual(id1, id3)


class TestCreateKwargsWrapper(unittest.TestCase):
    """测试_create_kwargs_wrapper函数"""

    def test_wrapper_with_inspectable_function(self):
        """测试可检查函数的包装器"""
        wrapper = _create_kwargs_wrapper(original_func_for_test, replacement_func_for_test)

        result = wrapper(2, 3, 4)
        self.assertEqual(result, 24)

        result = wrapper(a=2, b=3, c=4)
        self.assertEqual(result, 24)

        result = wrapper(a=2, b=3)
        self.assertEqual(result, 6)

    def test_wrapper_with_args(self):
        """测试包含位置参数的包装器"""
        wrapper = _create_kwargs_wrapper(original_func_with_args_kwargs, replacement_func_with_args_kwargs)

        result = wrapper(1, 2, 3, x=4, y=5)
        self.assertEqual(result, 15)

    def test_wrapper_with_inspection_error(self):
        """测试检查时出错的情况"""
        original_func = MagicMock()
        original_func.__name__ = 'test_func'

        with patch('inspect.signature', side_effect=TypeError("Cannot inspect")):
            wrapper = _create_kwargs_wrapper(original_func, replacement_func_for_inspection_error)

            result = wrapper(x=1, y=2)
            self.assertEqual(result, 3)

    def test_wrapper_with_inspection_error_with_args(self):
        """测试检查时出错且包含位置参数的情况"""
        original_func = MagicMock()
        original_func.__name__ = 'test_func'

        with patch('inspect.signature', side_effect=TypeError("Cannot inspect")):
            wrapper = _create_kwargs_wrapper(original_func, replacement_func_for_inspection_error)

            result = wrapper(1, 2, 3, x=4, y=5)
            self.assertEqual(result, 15)

    def test_wrapper_with_bind_error(self):
        """测试绑定参数时出错的情况"""

        def original_func(a, b):
            return a + b

        wrapper = _create_kwargs_wrapper(original_func, replacement_func_for_bind_error)

        result = wrapper(1, 2, 3)
        self.assertEqual(result, 3)

    def test_wrapper_with_inspect_signature_error(self):
        """测试inspect.signature出错的情况"""

        def original_func(a, b):
            return a + b

        def replacement_func(**kwargs):
            return kwargs.get('a', 0) + kwargs.get('b', 0)

        with patch('inspect.signature', side_effect=ValueError("Cannot inspect")):
            wrapper = _create_kwargs_wrapper(original_func, replacement_func)

            result = wrapper(1, 2)
            self.assertIsInstance(result, int)

    def test_wrapper_with_no_args(self):
        """测试无参数函数的包装器"""

        def original_func_no_args():
            return 42

        def replacement_func_no_args():
            return 84

        wrapper = _create_kwargs_wrapper(original_func_no_args, replacement_func_no_args)

        result = wrapper()
        self.assertEqual(result, 84)


class TestFunctionHijacker(unittest.TestCase):
    """测试FunctionHijacker类"""

    def setUp(self):
        """每个测试方法前的设置"""
        self.hijacker = FunctionHijacker()

        class TestContainer:
            @staticmethod
            def original_func(x, y=10):
                return x + y

            @staticmethod
            def another_func(z):
                return z * 2

        self.container = TestContainer()

    def test_initialization(self):
        """测试初始化"""
        hijacker = FunctionHijacker()
        self.assertIsInstance(hijacker.original_functions, dict)
        self.assertIsInstance(hijacker.hijacked_targets, dict)
        self.assertEqual(len(hijacker.original_functions), 0)
        self.assertEqual(len(hijacker.hijacked_targets), 0)

    def test_hijack_function_basic(self):
        """测试基本的函数劫持"""

        def replacement_func(x, y=10):
            return x * y

        target = (self.container, 'original_func')
        self.hijacker.hijack_function(target, replacement_func)

        result = self.container.original_func(2, 3)
        self.assertEqual(result, 6)

        original_func = self.hijacker.get_original_function(target)
        self.assertIsNotNone(original_func)
        self.assertEqual(original_func(2, 3), 5)

    def test_hijack_function_update(self):
        """测试更新已劫持的函数"""

        def replacement_func1(x, y=10):
            return x * y

        def replacement_func2(x, y=10):
            return x**y

        target = (self.container, 'original_func')

        self.hijacker.hijack_function(target, replacement_func1)
        result1 = self.container.original_func(2, 3)
        self.assertEqual(result1, 6)

        self.hijacker.hijack_function(target, replacement_func2)
        result2 = self.container.original_func(2, 3)
        self.assertEqual(result2, 8)

    def test_restore_function(self):
        """测试恢复函数"""

        def replacement_func(x, y=10):
            return x * y

        target = (self.container, 'original_func')

        self.hijacker.hijack_function(target, replacement_func)
        hijacked_result = self.container.original_func(2, 3)
        self.assertEqual(hijacked_result, 6)

        self.hijacker.restore_function(target)
        restored_result = self.container.original_func(2, 3)
        self.assertEqual(restored_result, 5)

    def test_restore_function_not_found(self):
        """测试恢复不存在的函数"""
        target = (self.container, 'nonexistent_func')

        self.hijacker.restore_function(target)

    def test_restore_all(self):
        """测试恢复所有函数"""

        def replacement_func1(x, y=10):
            return x * y

        def replacement_func2(z):
            return z**2

        target1 = (self.container, 'original_func')
        target2 = (self.container, 'another_func')

        self.hijacker.hijack_function(target1, replacement_func1)
        self.hijacker.hijack_function(target2, replacement_func2)

        self.assertEqual(self.container.original_func(2, 3), 6)
        self.assertEqual(self.container.another_func(4), 16)

        self.hijacker.restore_all()

        self.assertEqual(self.container.original_func(2, 3), 5)
        self.assertEqual(self.container.another_func(4), 8)

    def test_get_original_function(self):
        """测试获取原函数"""

        def replacement_func(x, y=10):
            return x * y

        target = (self.container, 'original_func')

        original = self.hijacker.get_original_function(target)
        self.assertIsNone(original)

        self.hijacker.hijack_function(target, replacement_func)
        original = self.hijacker.get_original_function(target)
        self.assertIsNotNone(original)
        self.assertEqual(original(2, 3), 5)

    def test_hijack_nonexistent_attribute(self):
        """测试劫持不存在的属性"""

        def replacement_func():
            return "replaced"

        target = (self.container, 'nonexistent_attr')
        with self.assertRaises(AttributeError):
            self.hijacker.hijack_function(target, replacement_func)

    def test_complex_function_with_kwargs(self):
        """测试复杂的函数劫持（带kwargs）"""

        def original_func(a, b=1, *args, **kwargs):
            numeric_args = [arg for arg in args if isinstance(arg, (int, float))]
            numeric_kwargs = [v for v in kwargs.values() if isinstance(v, (int, float))]
            return a + b + sum(numeric_args) + sum(numeric_kwargs)

        def replacement_func(**kwargs):
            a = kwargs.get('a', 0)
            b = kwargs.get('b', 1)
            args_tuple = kwargs.get('args', ())
            kwargs_dict = kwargs.get('kwargs', {})

            numeric_args = [x for x in args_tuple if isinstance(x, (int, float))]
            numeric_kwargs = [v for v in kwargs_dict.values() if isinstance(v, (int, float))]

            return a * b * (sum(numeric_args) + 1) * (sum(numeric_kwargs) + 1)

        self.container.original_func = original_func

        target = (self.container, 'original_func')
        self.hijacker.hijack_function(target, replacement_func)

        result = self.container.original_func(2, 3, 4, 5, x=6, y=7)
        self.assertEqual(result, 840)

    def test_hijack_builtin_function(self):
        """测试劫持内置函数（通过模拟）"""

        class MockBuiltin:
            def __init__(self):
                self.builtin_func = len

        mock_obj = MockBuiltin()

        def replacement_func(*args, **kwargs):
            return 42

        target = (mock_obj, 'builtin_func')
        self.hijacker.hijack_function(target, replacement_func)

        result = mock_obj.builtin_func([1, 2, 3])
        self.assertEqual(result, 42)

    def test_hijacker_with_no_hijacks(self):
        """测试没有劫持时的恢复操作"""
        self.hijacker.restore_all()

        original = self.hijacker.get_original_function(('dummy', 'dummy'))
        self.assertIsNone(original)


class TestGlobalFunctions(unittest.TestCase):
    """测试全局函数"""

    def setUp(self):
        """每个测试方法前的设置"""
        _hijacker.original_functions.clear()
        _hijacker.hijacked_targets.clear()

        class TestContainer:
            @staticmethod
            def original_func(x, y=10):
                return x + y

        self.container = TestContainer()

    def test_global_hijack_function(self):
        """测试全局hijack_function函数"""

        def replacement_func(x, y=10):
            return x * y

        target = (self.container, 'original_func')
        hijack_function(target, replacement_func)

        result = self.container.original_func(2, 3)
        self.assertEqual(result, 6)

    def test_global_restore_function(self):
        """测试全局restore_function函数"""

        def replacement_func(x, y=10):
            return x * y

        target = (self.container, 'original_func')

        hijack_function(target, replacement_func)
        self.assertEqual(self.container.original_func(2, 3), 6)

        restore_function(target)
        self.assertEqual(self.container.original_func(2, 3), 5)

    def test_global_restore_all_hijacked(self):
        """测试全局restore_all_hijacked函数"""

        def replacement_func1(x, y=10):
            return x * y

        def replacement_func2(x, y=10):
            return x**y

        target1 = (self.container, 'original_func')
        target2 = (self.container, 'original_func')  # noqa: F841

        hijack_function(target1, replacement_func1)
        hijack_function(target1, replacement_func2)

        self.assertEqual(self.container.original_func(2, 3), 8)

        restore_all_hijacked()
        self.assertEqual(self.container.original_func(2, 3), 5)

    def test_global_get_original_function(self):
        """测试全局get_original_function函数"""

        def replacement_func(x, y=10):
            return x * y

        target = (self.container, 'original_func')

        original = get_original_function(target)
        self.assertIsNone(original)

        hijack_function(target, replacement_func)
        original = get_original_function(target)
        self.assertIsNotNone(original)
        self.assertEqual(original(2, 3), 5)

    def test_global_functions_with_multiple_targets(self):
        """测试多个目标的全局函数"""

        def replacement_func1(x, y=10):
            return x * y

        def replacement_func2(x, y=10):
            return x + y + 100

        class MultiContainer:
            @staticmethod
            def func1(x, y=10):
                return x + y

            @staticmethod
            def func2(x, y=10):
                return x - y

        container = MultiContainer()

        target1 = (container, 'func1')
        target2 = (container, 'func2')

        hijack_function(target1, replacement_func1)
        hijack_function(target2, replacement_func2)

        self.assertEqual(container.func1(2, 3), 6)
        self.assertEqual(container.func2(10, 3), 113)

        restore_all_hijacked()

        self.assertEqual(container.func1(2, 3), 5)
        self.assertEqual(container.func2(10, 3), 7)

    def test_global_functions_edge_cases(self):
        """测试全局函数的边界情况"""
        restore_all_hijacked()


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def test_empty_container(self):
        """测试空容器"""

        class EmptyContainer:
            pass

        container = EmptyContainer()
        target = (container, 'nonexistent')

        def replacement():
            return "replaced"

        with self.assertRaises(AttributeError):
            hijack_function(target, replacement)

    def test_none_values_in_wrapper(self):
        """测试包装器中的None值处理"""

        def original_func(x=None):
            return x or 0

        def replacement_func(x=None):
            return (x or 0) + 10

        wrapper = _create_kwargs_wrapper(original_func, replacement_func)

        result = wrapper(x=None)
        self.assertEqual(result, 10)

        result = wrapper()
        self.assertEqual(result, 10)

    def test_function_with_no_parameters(self):
        """测试无参数函数"""

        def original_func():
            return 42

        def replacement_func():
            return 84

        wrapper = _create_kwargs_wrapper(original_func, replacement_func)

        result = wrapper()
        self.assertEqual(result, 84)

    def test_function_with_various_arg_types(self):
        """测试包含各种参数类型的函数"""

        def replacement_func(**kwargs):
            args_val = kwargs.get('args', ())
            kwargs_dict = kwargs.get('kwargs', {})

            if args_val:
                first_arg = args_val[0]
                kw_arg_val = kwargs_dict.get('kw_arg', 1)
                extra_val = kwargs_dict.get('extra', 0)

                return (
                    first_arg
                    * kw_arg_val
                    * (sum([x for x in args_val if isinstance(x, (int, float))]) + 1)
                    * (extra_val + 1)
                )
            else:
                return 0

        wrapper = _create_kwargs_wrapper(original_func_with_args_kwargs, replacement_func)

        result = wrapper(2, kw_arg=3, extra=5)
        self.assertEqual(result, 108)

    def test_inspect_signature_with_complex_function(self):
        """测试复杂函数的签名检查"""
        wrapper = _create_kwargs_wrapper(complex_func_for_test, replacement_func_for_complex_test)

        result = wrapper(42, "test")
        self.assertEqual(result, "replaced-42-test")

        result = wrapper(42)
        self.assertEqual(result, "replaced-42-default")


if __name__ == '__main__':
    unittest.main()
