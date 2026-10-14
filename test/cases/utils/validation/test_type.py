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
msmodelslim.utils.validation.type 模块的单元测试
"""
from typing import Mapping  # noqa: E402
from typing import Dict, List, Tuple  # noqa: E402, F401
import pytest  # noqa: E402
from msmodelslim.utils.exception import SchemaValidateError  # noqa: E402
from msmodelslim.utils.validation.type import (  # noqa: E402
    type_to_str,
    check_type,
    check_number,
    check_int,
    check_element_type,
    check_character,
    check_dict_character,
    check_dict_element,
    check_mapping_element,
)


# ------------------------------ 测试 type_to_str ------------------------------
def test_type_to_str():
    # 测试单个类型
    assert type_to_str(int) == "int"
    assert type_to_str(str) == "str"

    # 测试元组类型
    assert type_to_str((int, str)) == "int or str"
    assert type_to_str((list, tuple, dict)) == "list or tuple or dict"


# ------------------------------ 测试 check_type ------------------------------
def test_check_type_normal():
    # 正常情况 - 单个类型
    check_type(10, int, "number")
    check_type("test", str, "text")
    check_type([1, 2, 3], list, "list")

    # 正常情况 - 元组类型
    check_type(10, (int, float), "number")
    check_type(3.14, (int, float), "number")
    check_type(True, (int, bool), "flag")


def test_check_type_invalid():
    # 无效类型 - 单个类型
    with pytest.raises(SchemaValidateError) as excinfo:
        check_type("10", int, "number")
    assert "must be int, not str" in str(excinfo.value)

    # 无效类型 - 元组类型
    with pytest.raises(SchemaValidateError) as excinfo:
        check_type("10", (int, float), "number")
    assert "must be int or float, not str" in str(excinfo.value)

    # 布尔值不应被视为整数
    with pytest.raises(SchemaValidateError) as excinfo:
        check_type(True, int, "flag")
    assert "must be int, not bool" in str(excinfo.value)


# ------------------------------ 测试 check_number ------------------------------
def test_check_number_normal():
    # 正常整数
    check_number(10, param_name="int_num")
    check_number(10, min_value=5, param_name="int_num")
    check_number(10, max_value=15, param_name="int_num")
    check_number(10, min_value=5, max_value=15, param_name="int_num")

    # 正常浮点数
    check_number(3.14, param_name="float_num")
    check_number(3.14, min_value=3.0, param_name="float_num")
    check_number(3.14, max_value=4.0, param_name="float_num")


def test_check_number_invalid():
    # 类型错误
    with pytest.raises(SchemaValidateError) as excinfo:
        check_number("10", param_name="number")
    assert "must be int or float, not str" in str(excinfo.value)

    # 小于最小值
    with pytest.raises(SchemaValidateError) as excinfo:
        check_number(3, min_value=5, param_name="number")
    assert "is smaller than 5" in str(excinfo.value)

    # 大于最大值
    with pytest.raises(SchemaValidateError) as excinfo:
        check_number(17, max_value=15, param_name="number")
    assert "is larger than 15" in str(excinfo.value)


# ------------------------------ 测试 check_int ------------------------------
def test_check_int_normal():
    check_int(10, param_name="int_num")
    check_int(10, min_value=5, param_name="int_num")
    check_int(10, max_value=15, param_name="int_num")


def test_check_int_invalid():
    # 不是整数
    with pytest.raises(SchemaValidateError) as excinfo:
        check_int(3.14, param_name="int_num")
    assert "must be int, not float" in str(excinfo.value)

    # 小于最小值
    with pytest.raises(SchemaValidateError) as excinfo:
        check_int(3, min_value=5, param_name="int_num")
    assert "is smaller than 5" in str(excinfo.value)


# ------------------------------ 测试 check_element_type ------------------------------
def test_check_element_type_normal():
    # 列表元素检查
    check_element_type([1, 2, 3], int, param_name="int_list")
    check_element_type(["a", "b", "c"], str, param_name="str_list")

    # 元组元素检查
    check_element_type((1, 2, 3), int, param_name="int_tuple")
    check_element_type(("a", "b", "c"), str, param_name="str_tuple")


def test_check_element_type_invalid():
    # 列表包含无效元素
    with pytest.raises(SchemaValidateError) as excinfo:
        check_element_type([1, "2", 3], int, param_name="int_list")
    assert "Element in int_list is invalid." in str(excinfo.value)

    # 错误的容器类型
    with pytest.raises(SchemaValidateError) as excinfo:
        check_element_type("not a list", int, param_name="int_list")
    assert "must be list or tuple, not str" in str(excinfo.value)


# ------------------------------ 测试 check_character ------------------------------
def test_check_character_normal():
    # 有效字符
    check_character("valid_string_123")
    check_character("valid with space")
    check_character("contains/slash")
    check_character("contains.dots")
    check_character("contains-hyphens")
    check_character("contains_underscores")
    check_character("contains~tilde")
    check_character("contains*star")
    check_character('"quoted string"')
    check_character("<tag>")
    check_character("a=b")
    check_character("[list]")
    check_character("(tuple)")
    check_character("{dict}")
    check_character("a,b")
    check_character("a:b")

    # 有效列表
    check_character(["valid", "list", "elements"])

    # 有效嵌套列表
    check_character(["valid", ["nested", "list"], "elements"])


def test_check_character_invalid():
    # 包含无效字符
    with pytest.raises(SchemaValidateError) as excinfo:
        check_character("invalid@character")
    assert "contains invalid characters" in str(excinfo.value)

    # 列表中包含无效字符
    with pytest.raises(SchemaValidateError) as excinfo:
        check_character(["valid", "invalid@element"])
    assert "contains invalid characters" in str(excinfo.value)

    # 递归深度超出限制
    deep_list = []
    current = deep_list
    for _ in range(101):
        current.append([])
        current = current[0]

    with pytest.raises(SchemaValidateError) as excinfo:
        check_character(deep_list)
    assert "Recursion depth of value exceeds limitation" in str(excinfo.value)


# ------------------------------ 测试 check_dict_character ------------------------------
def test_check_dict_character_normal():
    # 正常字典
    check_dict_character({"name": "test", "id": 123, "valid": True})
    # 嵌套字典
    check_dict_character({"level1": {"level2": {"level3": "value"}}})
    # 带列表值的字典
    check_dict_character({"list": ["item1", "item2"]})


# ------------------------------ 测试 check_dict_element ------------------------------
def test_check_dict_element_normal():
    # 所有值都是整数
    check_dict_element({"a": 1, "b": 2, "c": 3}, int, "int_dict")

    # 所有值都是字符串
    check_dict_element({"name": "test", "desc": "description"}, str, "str_dict")


def test_check_dict_element_invalid():
    # 非字典类型
    with pytest.raises(SchemaValidateError) as excinfo:
        check_dict_element([1, 2, 3], int, "not_dict")
    assert "must be dict, not list" in str(excinfo.value)

    # 字典值类型不匹配
    with pytest.raises(SchemaValidateError) as excinfo:
        check_dict_element({"a": 1, "b": "two", "c": 3}, int, "int_dict")
    assert "Param of dict int_dict[b] should be int" in str(excinfo.value)


# ------------------------------ 测试 check_mapping_element ------------------------------
def test_check_mapping_element_normal():
    # 测试标准字典
    check_mapping_element({"a": 1, "b": 2}, int, "int_mapping")

    # 测试其他Mapping类型（如果有）
    class SimpleMapping(Mapping):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, key):
            return self.data[key]

        def __iter__(self):
            return iter(self.data)

        def __len__(self):
            return len(self.data)

    check_mapping_element(SimpleMapping({"x": "a", "y": "b"}), str, "str_mapping")


def test_check_mapping_element_invalid():
    # 非Mapping类型
    with pytest.raises(SchemaValidateError) as excinfo:
        check_mapping_element([1, 2, 3], int, "not_mapping")
    assert "must be Mapping, not list" in str(excinfo.value)

    # Mapping值类型不匹配
    with pytest.raises(SchemaValidateError) as excinfo:
        check_mapping_element({"a": 1, "b": "two", "c": 3}, int, "int_mapping")
    assert "Param of Mapping int_mapping[b] should be int" in str(excinfo.value)
