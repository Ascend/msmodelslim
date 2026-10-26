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

import pytest

from ascend_utils.common import security


def test_check_type_pass_when_valid_types():
    security.check_type(12, value_type=int)
    security.check_type(1.0, value_type=(int, float), additional_check_func=lambda xx: xx > 0)
    security.check_type(["aa", "bb"], value_type=list, additional_check_func=lambda xx: isinstance(xx, str))


def test_check_type_given_int_when_str_then_error():
    with pytest.raises(TypeError):
        # TypeError: test must be str, not int.
        security.check_type(12, value_type=str, param_name="test")


def test_check_type_given_int_when_list_tuple_then_error():
    with pytest.raises(TypeError):
        # TypeError: test must be list or tuple, not float.
        security.check_type(1.2, value_type=(list, tuple), param_name="test")


def test_check_type_given_check_when_fail_then_error():
    with pytest.raises(ValueError):
        # ValueError: Value of test is invalid.
        security.check_type(-1, value_type=(int, float), additional_check_func=lambda xx: xx > 0, param_name="test")


def test_check_type_given_list_tuple_check_when_fail_then_error():
    with pytest.raises(ValueError):
        # ValueError: Element in test is invalid.
        security.check_type(
            ["aa", 11, "bb"], value_type=list, additional_check_func=lambda xx: isinstance(xx, str), param_name="test"
        )


def test_check_number_pass_when_number():
    security.check_number(12, value_type=int)
    security.check_number(12.0, value_type=float, min_value=0)
    security.check_number(1, value_type=(int, float), max_value=12)


def test_check_number_given_str_when_any_then_error():
    with pytest.raises(TypeError):
        # TypeError: test must be int, not str.
        security.check_number("12", value_type=int, param_name="test")


def test_check_number_given_valid_when_min_max_invalid_then_error():
    with pytest.raises(ValueError):
        # ValueError: test = 0 is smaller than 1.
        security.check_number(0, value_type=(int, float), min_value=1, param_name="test")
    with pytest.raises(ValueError):
        # ValueError: test = 4.2 is larger than 1.
        security.check_number(4.2, value_type=(int, float), max_value=1, param_name="test")


def test_check_int_pass_when_int():
    security.check_int(12)
    security.check_int(12, min_value=0)
    security.check_int(1, max_value=12)


def test_check_int_given_float_when_any_then_error():
    with pytest.raises(TypeError):
        # TypeError: test must be int, not float.
        security.check_int(12.0, param_name="test")


def test_check_element_type_pass_when_element_types_matched():
    security.check_element_type([12], element_type=int)
    security.check_element_type(["aa", "bb"], element_type=str)
    security.check_element_type([], element_type=int)
    security.check_element_type([1.0, 2, 3], element_type=(int, float))


def test_check_element_type_given_str_when_any_then_error():
    with pytest.raises(TypeError):
        # TypeError: test must be list or tuple, not str.
        security.check_element_type("12", element_type=str, param_name="test")


def test_check_element_type_given_invalid_element_when_any_then_error():
    with pytest.raises(ValueError):
        # ValueError: Element in test is invalid. Should be all int.
        security.check_element_type(["12", "13"], element_type=int, param_name="test")
    with pytest.raises(ValueError):
        # ValueError: Element in test is invalid. Should be all int or float.
        security.check_element_type(("12", "13", 14), element_type=(int, float), param_name="test")


def test_check_character_pass_when_valid_characters():
    security.check_character(11)
    security.check_character("11")
    security.check_character(["11", "12"])
    security.check_character(["11", 12])
    security.check_character(["11", [11, "12"]])


def test_check_character_raise_value_error_when_invalid_characters():
    with pytest.raises(ValueError):
        # ValueError: test contains invalid characters.
        security.check_character("$", param_name="test")
    with pytest.raises(ValueError):
        # ValueError: test contains invalid characters.
        security.check_character(["$"], param_name="test")


def test_check_dict_character_pass_when_valid_dict():
    security.check_dict_character({})
    security.check_dict_character({'aa': 11})
    security.check_dict_character({'aa': {'bb': 12}})
    security.check_dict_character({'aa': {'bb': 'asdaf'}})
    security.check_dict_character({'aa': {'bb': '~'}})
    security.check_dict_character({'aa': {'bb': ['~', 11]}})


def test_check_dict_character_given_int_when_any_then_error():
    with pytest.raises(TypeError):
        # TypeError: test must be dict, not int.
        security.check_dict_character(12, param_name="test")


def test_check_dict_character_given_invalid_when_any_then_error():
    with pytest.raises(ValueError):
        # ValueError: Length of test key exceeds limitation 12.
        security.check_dict_character({'aa': {'bb' * 12: 'cc'}}, key_max_len=12, param_name="test")
    with pytest.raises(ValueError):
        # ValueError: test contains invalid characters.
        security.check_dict_character({'aa': {'bb': '#!@'}}, param_name="test")
    with pytest.raises(ValueError):
        # ValueError: test contains invalid characters.
        security.check_dict_character({'aa': {'bb': '/asfd\\'}}, param_name="test")
    with pytest.raises(ValueError):
        # ValueError: test contains invalid characters.
        security.check_dict_character({'aa': {'bb': ['vv', '$']}}, param_name="test")
