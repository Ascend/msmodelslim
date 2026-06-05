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

import unittest

from msmodelslim.ir.qal.qregistry import QAPI, QFuncRegistry, QABCRegistry
from msmodelslim.utils.exception import ToDoError, UnsupportedError


class TestQAPI(unittest.TestCase):
    """测试 QAPI 类"""

    def test_creation_should_set_attributes(self):
        """测试创建时设置属性"""
        api = QAPI("test_func", "key1", ("param1", "param2"))
        self.assertEqual(api.func_name, "test_func")
        self.assertEqual(api.dispatch_key, "key1")
        self.assertEqual(api.signature, ("param1", "param2"))
        self.assertEqual(api.impl_map, {})

    def test_add_impl_should_store_implementation(self):
        """测试添加实现"""
        api = QAPI("test_func", "key1", ("param1",))

        def impl(x):
            return x

        api.add_impl(impl, "key1")
        self.assertIn("key1", api.impl_map)
        self.assertEqual(api.impl_map["key1"], impl)

    def test_add_impl_should_overwrite_existing(self):
        """测试覆盖已有实现"""
        api = QAPI("test_func", "key1", ("param1",))

        def impl1(x):
            return x

        def impl2(x):
            return x + 1

        api.add_impl(impl1, "key1")
        api.add_impl(impl2, "key1")
        self.assertEqual(api.impl_map["key1"], impl2)


class TestQFuncRegistry(unittest.TestCase):
    """测试 QFuncRegistry 类"""

    def test_register_api_should_register_function(self):
        """测试注册 API"""
        # 使用唯一名称避免冲突
        api_name = "_test_unique_api_func_1"

        @QFuncRegistry.register_api(dispatch_key=str, api_name=api_name)
        def my_func(a: int, b: int) -> int:
            return a + b

        self.assertIn(api_name, QFuncRegistry._registered_api)
        api = QFuncRegistry._registered_api[api_name]
        self.assertEqual(api.func_name, api_name)
        self.assertEqual(api.dispatch_key, str)

        # 清理
        del QFuncRegistry._registered_api[api_name]

    def test_register_api_duplicate_should_raise_error(self):
        """测试重复注册 API 抛出异常"""
        api_name = "_test_unique_api_func_2"

        @QFuncRegistry.register_api(dispatch_key=str, api_name=api_name)
        def my_func(a: int) -> int:
            return a

        with self.assertRaises(ToDoError):

            @QFuncRegistry.register_api(dispatch_key=str, api_name=api_name)
            def my_func2(a: int) -> int:
                return a

        # 清理
        del QFuncRegistry._registered_api[api_name]

    def test_register_impl_should_add_to_existing_api(self):
        """测试注册实现"""
        api_name = "_test_unique_api_func_3"

        @QFuncRegistry.register_api(dispatch_key=str, api_name=api_name)
        def my_func(a: int) -> int:
            return a

        @QFuncRegistry.register(dispatch_key="key1", api_name=api_name)
        def impl1(a: int) -> int:
            return a * 2

        api = QFuncRegistry._registered_api[api_name]
        self.assertIn("key1", api.impl_map)

        # 清理
        del QFuncRegistry._registered_api[api_name]

    def test_register_impl_for_unregistered_api_should_raise_error(self):
        """测试为未注册的 API 注册实现抛出异常"""
        with self.assertRaises(ToDoError):

            @QFuncRegistry.register(dispatch_key="key1", api_name="_nonexistent_api_12345")
            def impl1(a: int) -> int:
                return a

    def test_register_impl_with_mismatched_signature_should_raise_error(self):
        """测试签名不匹配时抛出异常"""
        api_name = "_test_unique_api_func_4"

        @QFuncRegistry.register_api(dispatch_key=str, api_name=api_name)
        def my_func(a: int, b: int) -> int:
            return a + b

        with self.assertRaises(ToDoError):

            @QFuncRegistry.register(dispatch_key="key1", api_name=api_name)
            def impl1(a: int) -> int:  # 签名不匹配
                return a

        # 清理
        del QFuncRegistry._registered_api[api_name]

    def test_dispatch_should_call_correct_implementation(self):
        """测试分发到正确的实现"""
        api_name = "_test_unique_api_func_5"

        @QFuncRegistry.register_api(dispatch_key=str, api_name=api_name)
        def my_func(a: int) -> int:
            return QFuncRegistry.dispatch(api_name, "key1", a)

        @QFuncRegistry.register(dispatch_key="key1", api_name=api_name)
        def impl1(a: int) -> int:
            return a * 2

        @QFuncRegistry.register(dispatch_key="key2", api_name=api_name)
        def impl2(a: int) -> int:
            return a * 3

        self.assertEqual(my_func(5), 10)  # key1 -> 5 * 2
        # 直接调用 dispatch 测试不同 key
        result = QFuncRegistry.dispatch(api_name, "key2", 5)
        self.assertEqual(result, 15)  # key2 -> 5 * 3

        # 清理
        del QFuncRegistry._registered_api[api_name]

    def test_dispatch_with_unregistered_api_should_raise_error(self):
        """测试分发未注册的 API 抛出异常"""
        with self.assertRaises(UnsupportedError):
            QFuncRegistry.dispatch("_nonexistent_api_67890", "key1", 1)

    def test_dispatch_with_unregistered_key_should_raise_error(self):
        """测试分发未注册的 key 抛出异常"""
        api_name = "_test_unique_api_func_6"

        @QFuncRegistry.register_api(dispatch_key=str, api_name=api_name)
        def my_func(a: int) -> int:
            return a

        @QFuncRegistry.register(dispatch_key="key1", api_name=api_name)
        def impl1(a: int) -> int:
            return a

        with self.assertRaises(UnsupportedError):
            QFuncRegistry.dispatch(api_name, "nonexistent_key", 1)

        # 清理
        del QFuncRegistry._registered_api[api_name]


class TestQABCRegistry(unittest.TestCase):
    """测试 QABCRegistry 类"""

    def test_register_abc_should_register_class(self):
        """测试注册 ABC"""

        class _TestMyABC1:
            pass

        decorated = QABCRegistry.register_abc(dispatch_key="key1")
        result = decorated(_TestMyABC1)
        self.assertEqual(result, _TestMyABC1)
        self.assertIn(_TestMyABC1, QABCRegistry._registered_abc)

        # 清理
        del QABCRegistry._registered_abc[_TestMyABC1]

    def test_register_abc_duplicate_should_raise_error(self):
        """测试重复注册 ABC 抛出异常"""

        class _TestMyABC2:
            pass

        @QABCRegistry.register_abc(dispatch_key="key1")
        class _TestRegisteredABC2(_TestMyABC2):
            pass

        # 尝试用同一个类对象再次注册
        with self.assertRaises(ToDoError):
            QABCRegistry.register_abc(dispatch_key="key2")(_TestRegisteredABC2)

        # 清理
        del QABCRegistry._registered_abc[_TestRegisteredABC2]

    def test_register_impl_should_add_to_existing_abc(self):
        """测试注册实现"""

        class _TestMyABC3:
            pass

        @QABCRegistry.register_abc(dispatch_key="key1")
        class _TestRegisteredABC3(_TestMyABC3):
            pass

        class _TestMyImpl3(_TestRegisteredABC3):
            pass

        @QABCRegistry.register(dispatch_key="key1", abc_class=_TestRegisteredABC3)
        class _TestImpl3(_TestMyImpl3):
            pass

        abc = QABCRegistry._registered_abc[_TestRegisteredABC3]
        self.assertIn("key1", abc.impl_map)

        # 清理
        del QABCRegistry._registered_abc[_TestRegisteredABC3]

    def test_register_impl_for_unregistered_abc_should_raise_error(self):
        """测试为未注册的 ABC 注册实现抛出异常"""

        class _TestMyABC4:
            pass

        with self.assertRaises(ToDoError):

            @QABCRegistry.register(dispatch_key="key1", abc_class=_TestMyABC4)
            class _TestImpl4(_TestMyABC4):
                pass

    def test_register_impl_with_non_subclass_should_raise_error(self):
        """测试非子类注册实现抛出异常"""

        class _TestMyABC5:
            pass

        @QABCRegistry.register_abc(dispatch_key="key1")
        class _TestRegisteredABC5(_TestMyABC5):
            pass

        class _TestNotASubclass:
            pass

        with self.assertRaises(ToDoError):

            @QABCRegistry.register(dispatch_key="key1", abc_class=_TestRegisteredABC5)
            class _TestImpl5(_TestNotASubclass):
                pass

        # 清理
        del QABCRegistry._registered_abc[_TestRegisteredABC5]

    def test_create_should_instantiate_registered_class(self):
        """测试创建已注册类的实例"""

        class _TestMyABC6:
            def __init__(self, value):
                self.value = value

        @QABCRegistry.register_abc(dispatch_key="key1")
        class _TestRegisteredABC6(_TestMyABC6):
            pass

        @QABCRegistry.register(dispatch_key="key1", abc_class=_TestRegisteredABC6)
        class _TestImpl6(_TestRegisteredABC6):
            pass

        instance = QABCRegistry.create(_TestRegisteredABC6, "key1", 42)
        self.assertIsInstance(instance, _TestImpl6)
        self.assertEqual(instance.value, 42)

        # 清理
        del QABCRegistry._registered_abc[_TestRegisteredABC6]

    def test_create_with_unregistered_abc_should_raise_error(self):
        """测试创建未注册 ABC 的实例抛出异常"""

        class _TestMyABC7:
            pass

        with self.assertRaises(UnsupportedError):
            QABCRegistry.create(_TestMyABC7, "key1")

    def test_create_with_unregistered_key_should_raise_error(self):
        """测试使用未注册的 key 创建实例抛出异常"""

        class _TestMyABC8:
            pass

        @QABCRegistry.register_abc(dispatch_key="key1")
        class _TestRegisteredABC8(_TestMyABC8):
            pass

        with self.assertRaises(UnsupportedError):
            QABCRegistry.create(_TestRegisteredABC8, "nonexistent_key")

        # 清理
        del QABCRegistry._registered_abc[_TestRegisteredABC8]

    def test_multi_register_should_register_for_multiple_keys(self):
        """测试多键注册"""

        class _TestMyABC9:
            pass

        @QABCRegistry.register_abc(dispatch_key="key1")
        class _TestRegisteredABC9(_TestMyABC9):
            pass

        @QABCRegistry.multi_register(dispatch_key=["key1", "key2", "key3"], abc_type=_TestRegisteredABC9)
        class _TestImpl9(_TestRegisteredABC9):
            pass

        abc = QABCRegistry._registered_abc[_TestRegisteredABC9]
        self.assertIn("key1", abc.impl_map)
        self.assertIn("key2", abc.impl_map)
        self.assertIn("key3", abc.impl_map)

        # 清理
        del QABCRegistry._registered_abc[_TestRegisteredABC9]


if __name__ == '__main__':
    unittest.main()
