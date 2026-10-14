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
msmodelslim.utils.logging 模块的单元测试
"""

import logging  # noqa: E402
from unittest import TestCase  # noqa: E402

from msmodelslim.utils.logging import logger_setter, get_logger, set_logger_level  # noqa: E402


class TestLoggerSetter(TestCase):
    """测试 logger_setter 装饰器和上下文管理器"""

    def test_decorator_when_decorate_function_with_prefix_then_use_specified_logger(self):
        """测试装饰器功能：当装饰函数并指定前缀时，函数内部应使用指定的logger名称"""

        @logger_setter(prefix="test.module")
        def test_function():
            logger = get_logger()
            # 验证logger名称
            self.assertEqual(logger.name, "test.module")

        test_function()

    def test_decorator_when_decorate_function_without_prefix_then_use_module_name(self):
        """测试装饰器功能：当装饰函数但不指定前缀时，应使用函数所在模块名作为logger名称"""

        @logger_setter()
        def test_function():
            logger = get_logger()
            # 验证logger名称使用当前模块名
            self.assertIn("test_logging", logger.name)

        test_function()

    def test_decorator_when_decorate_class_with_prefix_and_subfix_then_all_methods_use_specified_logger(self):
        """测试装饰器功能：当装饰类并指定前缀和后缀时，类的所有方法都应使用指定的logger名称"""

        @logger_setter(prefix="test.class", subfix="methods")
        class TestClass:
            def method1(self):
                logger = get_logger()
                # 返回logger名称供外部验证
                return logger.name

            @staticmethod
            def static_method():
                logger = get_logger()
                # 返回logger名称供外部验证
                return logger.name

            @classmethod
            def class_method(cls):
                logger = get_logger()
                # 返回logger名称供外部验证
                return logger.name

        # 创建实例并测试方法
        obj = TestClass()
        self.assertEqual(obj.method1(), "test.class.methods")
        self.assertEqual(TestClass.static_method(), "test.class.methods")
        self.assertEqual(TestClass.class_method(), "test.class.methods")

    def test_context_manager_when_use_with_prefix_then_use_specified_logger_and_restore_after_exit(self):
        """测试上下文管理器功能：使用前缀时应在内部使用指定logger，退出后恢复原始logger"""
        # 在上下文管理器外部
        original_logger = get_logger()
        original_name = original_logger.name

        # 在上下文管理器内部
        with logger_setter(prefix="test.context"):
            logger = get_logger()
            # 验证logger名称
            self.assertEqual(logger.name, "test.context")

        # 验证恢复原始logger
        restored_logger = get_logger()
        self.assertEqual(restored_logger.name, original_name)

    def test_context_manager_when_use_with_prefix_and_subfix_then_use_combined_logger_name(self):
        """测试上下文管理器功能：使用前缀和后缀时logger名称应为prefix.subfix格式"""
        # 在上下文管理器外部
        original_logger = get_logger()
        original_name = original_logger.name

        # 在上下文管理器内部，使用前缀和后缀
        with logger_setter(prefix="test.app", subfix="database"):
            logger = get_logger()
            # 验证logger名称
            self.assertEqual(logger.name, "test.app.database")

        # 验证恢复原始logger
        restored_logger = get_logger()
        self.assertEqual(restored_logger.name, original_name)

    def test_context_manager_when_nested_use_then_each_level_has_own_logger_and_restore_correctly(self):
        """测试嵌套上下文管理器：每层应使用自己的logger，退出时正确恢复到上一层的logger"""
        original_logger = get_logger()
        original_name = original_logger.name

        with logger_setter(prefix="outer"):
            outer_logger = get_logger()
            self.assertEqual(outer_logger.name, "outer")

            with logger_setter(prefix="inner", subfix="nested"):
                inner_logger = get_logger()
                self.assertEqual(inner_logger.name, "inner.nested")

            # 验证回到外层
            current_logger = get_logger()
            self.assertEqual(current_logger.name, "outer")

        # 验证回到最外层
        final_logger = get_logger()
        self.assertEqual(final_logger.name, original_name)

    def test_context_manager_when_no_prefix_specified_then_use_calling_module_name(self):
        """测试上下文管理器功能：不指定前缀时应使用调用模块名作为logger名称"""
        original_logger = get_logger()
        original_name = original_logger.name

        # 在上下文管理器内部，不指定 prefix
        with logger_setter():
            logger = get_logger()
            # 验证logger名称使用当前模块名
            self.assertIn("test_logging", logger.name)

        # 验证恢复原始logger
        restored_logger = get_logger()
        self.assertEqual(restored_logger.name, original_name)

    def test_context_manager_when_only_subfix_specified_then_use_calling_module_with_subfix(self):
        """测试上下文管理器功能：只指定后缀时logger名称应为模块名.subfix格式"""
        original_logger = get_logger()
        original_name = original_logger.name

        # 在上下文管理器内部，只指定 subfix
        with logger_setter(subfix="test"):
            logger = get_logger()
            # 验证logger名称使用当前模块名.test
            self.assertTrue(logger.name.endswith(".test"))

        # 验证恢复原始logger
        restored_logger = get_logger()
        self.assertEqual(restored_logger.name, original_name)

    def test_decorator_when_specify_prefix_then_use_specified_prefix(self):
        """测试装饰器功能：指定前缀时应使用指定的前缀作为logger名称"""

        @logger_setter(prefix="msmodelslim.utils.logging")
        def test_function_with_prefix():
            logger = get_logger()
            # 验证logger名称
            self.assertEqual(logger.name, "msmodelslim.utils.logging")

        test_function_with_prefix()

    def test_decorator_when_specify_subfix_then_use_function_module_with_subfix(self):
        """测试装饰器功能：指定后缀时logger名称应为函数模块名.subfix格式"""

        @logger_setter(subfix="custom")
        def test_function_with_subfix():
            logger = get_logger()
            # 验证logger名称使用当前模块名.custom
            self.assertTrue(logger.name.endswith(".custom"))

        test_function_with_subfix()


class TestLoggingFunctions(TestCase):
    """测试其他日志相关函数"""

    def test_get_logger_when_no_name_specified_then_return_default_logger(self):
        """测试get_logger函数：不指定名称时应返回默认logger，指定名称时应返回对应名称的logger"""
        # 测试获取默认 logger
        logger = get_logger()
        self.assertIsInstance(logger, logging.Logger)

        # 测试获取指定名称的 logger
        named_logger = get_logger("test.named")
        self.assertIsInstance(named_logger, logging.Logger)
        self.assertEqual(named_logger.name, "test.named")

    def test_set_logger_level_when_valid_level_then_set_successfully(self):
        """测试set_logger_level函数：设置有效级别时应成功，设置无效级别时应抛出异常"""
        # 测试设置有效的日志级别
        set_logger_level("debug")
        logger = get_logger()
        self.assertEqual(logger.level, logging.DEBUG)

        set_logger_level("info")
        logger = get_logger()
        self.assertEqual(logger.level, logging.INFO)

        set_logger_level("warning")
        logger = get_logger()
        self.assertEqual(logger.level, logging.WARNING)

        set_logger_level("error")
        logger = get_logger()
        self.assertEqual(logger.level, logging.ERROR)

        set_logger_level("critical")
        logger = get_logger()
        self.assertEqual(logger.level, logging.CRITICAL)

        # 测试设置无效的日志级别
        with self.assertRaises(Exception):
            set_logger_level(123)  # 非字符串类型


if __name__ == "__main__":
    import unittest

    unittest.main()
