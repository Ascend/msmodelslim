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

import unittest

from msmodelslim.utils.exception import ModelslimError, UnexpectedError


class TestModelslimError(unittest.TestCase):
    """测试 ModelslimError 基类"""

    def test_str_when_error_with_message_then_return_formatted_string(self):
        """测试__str__方法：当错误包含消息时，应返回格式化的字符串"""
        error = ModelslimError("Something unexpected happened")

        expected_str = "Code: 0, Message: Something unexpected happened"
        self.assertEqual(str(error), expected_str)

    def test_str_when_error_with_message_and_action_then_return_formatted_string_with_tip(self):
        """测试__str__方法：当错误包含消息和解决推荐时，应返回包含TIP的格式化字符串"""
        error = ModelslimError("Python version not compatible", action="Please upgrade to Python 3.8 or higher")

        expected_str = "Code: 0, Message: Python version not compatible\nTIP: Please upgrade to Python 3.8 or higher"
        self.assertEqual(str(error), expected_str)

    def test_str_when_error_without_message_then_use_default_message(self):
        """测试__str__方法：当错误不包含消息时，应使用默认消息"""
        error = ModelslimError()

        expected_str = "Code: 0, Message: modelslim error"
        self.assertEqual(str(error), expected_str)

    def test_str_when_error_with_empty_message_then_use_default_message(self):
        """测试__str__方法：当错误消息为空字符串时，应使用默认消息"""
        error = ModelslimError("")

        expected_str = "Code: 0, Message: modelslim error"
        self.assertEqual(str(error), expected_str)

    def test_str_when_error_with_none_message_then_use_default_message(self):
        """测试__str__方法：当错误消息为None时，应使用默认消息"""
        error = ModelslimError(None)

        expected_str = "Code: 0, Message: modelslim error"
        self.assertEqual(str(error), expected_str)

    def test_repr_when_error_with_message_then_return_formatted_repr(self):
        """测试__repr__方法：当错误包含消息时，应返回格式化的repr字符串"""
        error = ModelslimError("Something unexpected happened")

        expected_repr = "[ModelslimError] Code: 0, Message: Something unexpected happened"
        self.assertEqual(repr(error), expected_repr)

    def test_repr_when_error_with_message_and_action_then_return_formatted_repr_with_tip(self):
        """测试__repr__方法：当错误包含消息和解决推荐时，应返回包含TIP的格式化repr字符串"""
        error = ModelslimError("Python version not compatible", action="Please upgrade to Python 3.8 or higher")

        expected_repr = "[ModelslimError] Code: 0, Message: Python version not compatible, TIP: Please upgrade to Python 3.8 or higher"
        self.assertEqual(repr(error), expected_repr)

    def test_repr_when_error_without_message_then_use_default_message(self):
        """测试__repr__方法：当错误不包含消息时，应使用默认消息"""
        error = ModelslimError()

        expected_repr = "[ModelslimError] Code: 0, Message: modelslim error"
        self.assertEqual(repr(error), expected_repr)

    def test_repr_when_error_with_empty_message_then_use_default_message(self):
        """测试__repr__方法：当错误消息为空字符串时，应使用默认消息"""
        error = ModelslimError("")

        expected_repr = "[ModelslimError] Code: 0, Message: modelslim error"
        self.assertEqual(repr(error), expected_repr)

    def test_repr_when_error_with_none_message_then_use_default_message(self):
        """测试__repr__方法：当错误消息为None时，应使用默认消息"""
        error = ModelslimError(None)

        expected_repr = "[ModelslimError] Code: 0, Message: modelslim error"
        self.assertEqual(repr(error), expected_repr)

    def test_create_exception_when_valid_parameters_then_create_custom_error_class(self):
        """测试create_exception类方法：使用有效参数时应成功创建自定义错误类"""
        CustomError = ModelslimError.create_exception("CustomError", 999, "Custom error message")

        # 测试创建的错误类
        error = CustomError("Test message")
        self.assertEqual(error.code, 999)
        self.assertEqual(error.default_message, "Custom error message")
        self.assertEqual(str(error), "Code: 999, Message: Test message")
        self.assertEqual(repr(error), "[CustomError] Code: 999, Message: Test message")


class TestUnexpectedError(unittest.TestCase):
    """测试 UnexpectedError 类及其 TIPS 功能"""

    def tearDown(self):
        """每个测试用例执行完后清空 TIPS，避免污染"""
        UnexpectedError.clear_tips()

    def test_unexpected_error_attributes(self):
        """测试 UnexpectedError 的基本属性"""
        error = UnexpectedError("test msg")
        self.assertEqual(error.code, 500)
        self.assertEqual(error.default_message, "Unexpected error.")
        self.assertIn("Code: 500", str(error))
        self.assertIn("Message: test msg", str(error))

    def test_inject_tips_single_str(self):
        """测试注入单条字符串 TIP"""
        UnexpectedError.inject_tips("Tip 1")
        self.assertEqual(UnexpectedError.tips, ["Tip 1"])

    def test_inject_tips_iterable(self):
        """测试注入可迭代对象 TIPS"""
        UnexpectedError.inject_tips(["Tip 2", "Tip 3"])
        self.assertEqual(UnexpectedError.tips, ["Tip 2", "Tip 3"])

    def test_multiple_inject_tips(self):
        """测试多次注入 TIPS 累积顺序"""
        UnexpectedError.inject_tips("Tip A")
        UnexpectedError.inject_tips(["Tip B", "Tip C"])
        self.assertEqual(UnexpectedError.tips, ["Tip A", "Tip B", "Tip C"])

    def test_clear_tips(self):
        """测试清空 TIPS"""
        UnexpectedError.inject_tips("To be cleared")
        UnexpectedError.clear_tips()
        self.assertEqual(UnexpectedError.tips, [])
        UnexpectedError.inject_tips("New tip")
        self.assertEqual(UnexpectedError.tips, ["New tip"])

    def test_str_display_with_various_tips(self):
        """测试 __str__ 各种 TIP 组合展示情况"""
        UnexpectedError.inject_tips("Class Tip 1")
        error = UnexpectedError("msg")
        expected_str = "Code: 500, Message: msg\nTIP: Class Tip 1"
        self.assertEqual(str(error), expected_str)

        UnexpectedError.clear_tips()
        error_with_action = UnexpectedError("msg", action="Instance Action")
        expected_str_action = "Code: 500, Message: msg\nTIP: Instance Action"
        self.assertEqual(str(error_with_action), expected_str_action)

        UnexpectedError.inject_tips(["Class Tip 1", "Class Tip 2"])
        error_both = UnexpectedError("msg", action="Instance Action")
        expected_str_both = "Code: 500, Message: msg\nTIP: Instance Action\nTIP: Class Tip 1\nTIP: Class Tip 2"
        self.assertEqual(str(error_both), expected_str_both)

    def test_repr_display_with_various_tips(self):
        """测试 __repr__ 各种 TIP 组合展示情况"""
        UnexpectedError.inject_tips("Class Tip")
        error = UnexpectedError("msg", action="Instance Action")
        expected_repr = "[UnexpectedError] Code: 500, Message: msg, TIP: Instance Action, TIP: Class Tip"
        self.assertEqual(repr(error), expected_repr)

    def test_inheritance_compatibility(self):
        """测试继承兼容性"""
        error = UnexpectedError("msg")
        self.assertTrue(isinstance(error, UnexpectedError))
        self.assertTrue(isinstance(error, ModelslimError))
        self.assertTrue(isinstance(error, Exception))


if __name__ == "__main__":
    unittest.main()
