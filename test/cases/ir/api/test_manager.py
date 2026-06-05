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
from unittest.mock import patch

from msmodelslim.ir.api.manager import FuncMark, APIManager
from msmodelslim.utils.exception import VersionError


class TestFuncMark(unittest.TestCase):
    """测试 FuncMark 数据类"""

    def test_creation_should_set_attributes(self):
        """测试创建时设置属性"""
        mark = FuncMark(min_version="1.0.0", max_version="2.0.0")
        self.assertEqual(mark.min_version, "1.0.0")
        self.assertEqual(mark.max_version, "2.0.0")


class TestAPIManager(unittest.TestCase):
    """测试 APIManager 类"""

    def setUp(self):
        """测试前重置状态"""
        self._original_initialized = APIManager.INITIALIZED
        self._original_module = APIManager.API_MODULE
        self._original_marked = APIManager.MARKED_FUNC.copy()
        APIManager.INITIALIZED = False
        APIManager.API_MODULE = None
        APIManager.MARKED_FUNC = {}

    def tearDown(self):
        """测试后恢复状态"""
        APIManager.INITIALIZED = self._original_initialized
        APIManager.API_MODULE = self._original_module
        APIManager.MARKED_FUNC = self._original_marked

    def test_max_version_should_be_set(self):
        """测试 MAX_VERSION 常量"""
        self.assertEqual(APIManager.MAX_VERSION, "9999.9999.9999")

    def test_init_module_should_set_initialized_flag(self):
        """测试 init_module 设置初始化标志"""
        APIManager.init_module()
        self.assertTrue(APIManager.INITIALIZED)

    def test_init_module_should_not_reinitialize(self):
        """测试 init_module 不会重复初始化"""
        APIManager.init_module()
        first_module = APIManager.API_MODULE
        APIManager.init_module()
        self.assertIs(APIManager.API_MODULE, first_module)

    @patch('msmodelslim.ir.api.manager.importlib.import_module')
    def test_init_module_should_handle_import_error(self, mock_import):
        """测试 init_module 处理导入错误"""
        mock_import.side_effect = ImportError("test error")
        APIManager.init_module()
        self.assertTrue(APIManager.INITIALIZED)
        self.assertIsNone(APIManager.API_MODULE)

    def test_get_module_should_return_loaded_module(self):
        """测试 get_module 返回加载的模块"""
        module = APIManager.get_module()
        # 如果 api_main 可以导入，应该返回模块
        if module is not None:
            self.assertTrue(hasattr(module, '__name__'))

    def test_get_version_should_return_version_string(self):
        """测试 get_module 返回版本字符串"""
        version = APIManager.get_version()
        self.assertIsInstance(version, str)

    @patch.object(APIManager, 'get_version', return_value="1.0.0")
    def test_check_version_should_warn_for_out_of_range(self, mock_version):
        """测试 check_version 对超出范围的版本发出警告"""
        APIManager.MARKED_FUNC["test_func"] = FuncMark(min_version="2.0.0", max_version="3.0.0")
        # 应该发出警告但不抛出异常
        APIManager.check_version()

    @patch.object(APIManager, 'get_version', return_value="2.5.0")
    def test_check_version_should_not_warn_for_in_range(self, mock_version):
        """测试 check_version 对范围内版本不发出警告"""
        APIManager.MARKED_FUNC["test_func"] = FuncMark(min_version="2.0.0", max_version="3.0.0")
        # 不应该发出警告
        APIManager.check_version()

    def test_mark_require_version_should_register_function(self):
        """测试 mark_require_version 注册函数"""

        @APIManager.mark_require_version(min_version="1.0.0", max_version="2.0.0")
        def test_func():
            pass

        self.assertIn("test_func", APIManager.MARKED_FUNC)
        mark = APIManager.MARKED_FUNC["test_func"]
        self.assertEqual(mark.min_version, "1.0.0")
        self.assertEqual(mark.max_version, "2.0.0")

    @patch.object(APIManager, 'get_version', return_value="1.5.0")
    def test_mark_require_version_should_allow_valid_version(self, mock_version):
        """测试 mark_require_version 允许有效版本"""

        @APIManager.mark_require_version(min_version="1.0.0", max_version="2.0.0")
        def test_func():
            return "success"

        result = test_func()
        self.assertEqual(result, "success")

    @patch.object(APIManager, 'get_version', return_value="3.0.0")
    def test_mark_require_version_should_raise_for_version_too_high(self, mock_version):
        """测试 mark_require_version 对过高版本抛出异常"""

        @APIManager.mark_require_version(min_version="1.0.0", max_version="2.0.0")
        def test_func():
            pass

        with self.assertRaises(VersionError):
            test_func()

    @patch.object(APIManager, 'get_version', return_value="0.5.0")
    def test_mark_require_version_should_raise_for_version_too_low(self, mock_version):
        """测试 mark_require_version 对过低版本抛出异常"""

        @APIManager.mark_require_version(min_version="1.0.0", max_version="2.0.0")
        def test_func():
            pass

        with self.assertRaises(VersionError):
            test_func()

    def test_mark_require_version_default_max_should_be_max_version(self):
        """测试 mark_require_version 默认 max_version 为 MAX_VERSION"""

        @APIManager.mark_require_version(min_version="1.0.0")
        def test_func():
            pass

        mark = APIManager.MARKED_FUNC["test_func"]
        self.assertEqual(mark.max_version, APIManager.MAX_VERSION)


if __name__ == '__main__':
    unittest.main()
