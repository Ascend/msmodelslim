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

from types import SimpleNamespace

import pytest

from msmodelslim.utils.validation.pydantic import (
    at_least_one_element,
    at_least_one_key,
    allow_empty_dict,
    allow_empty_list,
    config_constraint,
    in_range,
    is_string_list,
    validate_str_length,
)


def test_at_least_one_element_returns_value_or_raises_value_error_with_field_name():
    info = SimpleNamespace(field_name="items")

    assert at_least_one_element([1, 2], info) == [1, 2]

    with pytest.raises(ValueError, match="items must have at least one element"):
        at_least_one_element([], info)


def test_in_range_returns_value_or_raises_value_error_with_field_name_and_bounds():
    validator = in_range(min_val=0.0, max_val=1.0)
    info = SimpleNamespace(field_name="ratio")

    assert validator(0.5, info) == 0.5

    with pytest.raises(ValueError, match="ratio must be less than or equal to 1.0, got 2.0"):
        validator(2.0, info)


def test_config_constraint_attaches_documentation_when_decorating_validator():
    @config_constraint("自定义配置约束")
    def validator(value: int) -> int:
        return value

    assert validator.config_constraint == "自定义配置约束"


def test_config_constraint_formats_values_when_format_arguments_are_provided():
    @config_constraint("长度 ≤ {max_len}", max_len=64)
    def validator(value: str) -> str:
        return value

    assert validator.config_constraint == "长度 ≤ 64"


def test_config_constraint_keeps_literal_braces_when_format_arguments_are_provided():
    @config_constraint("集合 {{a,b}}，最多 {limit} 项", limit=3)
    def validator(value: list) -> list:
        return value

    assert validator.config_constraint == "集合 {a,b}，最多 3 项"


def test_config_constraint_raises_when_format_argument_is_missing():
    with pytest.raises(KeyError, match="max_len"):
        config_constraint("长度 ≤ {max_len}", min_len=1)


def test_builtin_after_validator_exposes_documented_constraint_when_registered():
    assert at_least_one_element.config_constraint == "列表至少包含 1 项"
    assert is_string_list.config_constraint == "列表元素必须全部为字符串"
    assert allow_empty_list.config_constraint == "列表（允许为空列表）"
    assert allow_empty_dict.config_constraint == "对象（允许为空对象）"
    assert at_least_one_key.config_constraint == "对象至少包含 1 个键值对"


@pytest.mark.parametrize(
    ("max_len", "expected"),
    [
        (200, "字符串长度 ≤ 200"),
        (64, "字符串长度 ≤ 64"),
    ],
)
def test_validate_str_length_sets_documented_constraint_when_factory_called(max_len, expected):
    assert validate_str_length(max_len).config_constraint == expected


@pytest.mark.parametrize(
    ("min_val", "max_val", "expected"),
    [
        (1, 1000, "≥1；≤1000"),
        (None, 10, "≤10"),
        (0, None, "≥0"),
        (None, None, "不限范围"),
    ],
)
def test_in_range_sets_documented_constraint_when_bounds_configured(min_val, max_val, expected):
    assert in_range(min_val=min_val, max_val=max_val).config_constraint == expected
