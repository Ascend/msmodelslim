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

import torch

from msmodelslim.ir.api.base_api import calculate_qparam, quantize, dequantize, fake_quantize
from msmodelslim.ir.qal import QDType, QParam, QScope, QScheme, QStorage, QFuncRegistry

# 导入实现以确保注册


class TestCalculateQparam(unittest.TestCase):
    """测试 calculate_qparam 函数"""

    def test_should_be_registered_as_api(self):
        """测试 calculate_qparam 已注册为 API"""
        self.assertIn("calculate_qparam", QFuncRegistry._registered_api)

    def test_should_raise_for_unsupported_dispatch_key(self):
        """测试使用不支持的 dispatch key 时抛出异常"""
        min_val = torch.tensor([0.0])
        max_val = torch.tensor([1.0])
        # PLACEHOLDER 类型没有实现
        with self.assertRaises(Exception):
            calculate_qparam(min_val, max_val, QDType.PLACEHOLDER, QScope.PER_TENSOR, True)


class TestQuantize(unittest.TestCase):
    """测试 quantize 函数"""

    def test_should_be_registered_as_api(self):
        """测试 quantize 已注册为 API"""
        self.assertIn("quantize", QFuncRegistry._registered_api)

    def test_should_raise_for_unsupported_dispatch_key(self):
        """测试使用不支持的 dispatch key 时抛出异常"""
        tensor = QStorage(QDType.FLOAT, torch.tensor([1.0]))
        scheme = QScheme(scope=QScope.PLACEHOLDER, dtype=QDType.PLACEHOLDER, symmetric=True)
        q_param = QParam(scheme=scheme)
        with self.assertRaises(Exception):
            quantize(tensor, q_param)


class TestDequantize(unittest.TestCase):
    """测试 dequantize 函数"""

    def test_should_be_registered_as_api(self):
        """测试 dequantize 已注册为 API"""
        self.assertIn("dequantize", QFuncRegistry._registered_api)

    def test_should_raise_for_unsupported_dispatch_key(self):
        """测试使用不支持的 dispatch key 时抛出异常"""
        tensor = QStorage(QDType.INT8, torch.tensor([1], dtype=torch.int8))
        scheme = QScheme(scope=QScope.PLACEHOLDER, dtype=QDType.PLACEHOLDER, symmetric=True)
        q_param = QParam(scheme=scheme)
        with self.assertRaises(Exception):
            dequantize(tensor, q_param)


class TestFakeQuantize(unittest.TestCase):
    """测试 fake_quantize 函数"""

    def test_should_be_registered_as_api(self):
        """测试 fake_quantize 已注册为 API"""
        self.assertIn("fake_quantize", QFuncRegistry._registered_api)

    def test_should_raise_for_unsupported_dispatch_key(self):
        """测试使用不支持的 dispatch key 时抛出异常"""
        tensor = QStorage(QDType.FLOAT, torch.tensor([1.0]))
        scheme = QScheme(scope=QScope.PLACEHOLDER, dtype=QDType.PLACEHOLDER, symmetric=True)
        q_param = QParam(scheme=scheme)
        with self.assertRaises(Exception):
            fake_quantize(tensor, q_param)


if __name__ == '__main__':
    unittest.main()
