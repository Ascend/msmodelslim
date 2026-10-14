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
from decimal import Decimal

from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.validation.value import (
    greater_than_zero,
    validate_normalized_value,
    is_string_list,
    non_empty_string,
    int_greater_than_zero,
    allow_empty_list,
    allow_empty_dict,
    at_least_one_key,
    in_range,
)


# ============================== TestGreaterThanZero ==============================


class TestGreaterThanZero:
    """greater_than_zero 的外部接口测试"""

    def test_returns_value_when_positive_float(self):
        result = greater_than_zero(1.5)
        assert result == 1.5

    def test_returns_value_when_positive_int(self):
        result = greater_than_zero(5)
        assert result == 5

    def test_returns_value_when_positive_decimal(self):
        result = greater_than_zero(Decimal("0.1"))
        assert result == Decimal("0.1")

    def test_raises_error_when_zero(self):
        with pytest.raises(SchemaValidateError, match="greater than 0"):
            greater_than_zero(0.0)

    def test_raises_error_when_negative_float(self):
        with pytest.raises(SchemaValidateError, match="greater than 0"):
            greater_than_zero(-0.5)

    def test_raises_error_when_negative_int(self):
        with pytest.raises(SchemaValidateError, match="greater than 0"):
            greater_than_zero(-10)

    def test_raises_error_when_string(self):
        with pytest.raises(SchemaValidateError, match="int/float/Decimal"):
            greater_than_zero("5")

    def test_raises_error_when_none(self):
        with pytest.raises(SchemaValidateError, match="int/float/Decimal"):
            greater_than_zero(None)


# ============================== TestIntGreaterThanZero ==============================


class TestIntGreaterThanZero:
    """int_greater_than_zero 的外部接口测试"""

    def test_returns_value_when_positive_int(self):
        result = int_greater_than_zero(5)
        assert result == 5

    def test_raises_error_when_float(self):
        with pytest.raises(SchemaValidateError, match="int"):
            int_greater_than_zero(1.5)

    def test_raises_error_when_zero(self):
        with pytest.raises(SchemaValidateError, match="greater than 0"):
            int_greater_than_zero(0)

    def test_raises_error_when_negative(self):
        with pytest.raises(SchemaValidateError, match="greater than 0"):
            int_greater_than_zero(-10)


# ============================== TestValidateNormalizedValue ==============================


class TestValidateNormalizedValue:
    """validate_normalized_value 的外部接口测试"""

    def test_returns_none_when_input_none(self):
        result = validate_normalized_value(None)
        assert result is None

    def test_returns_value_when_in_range(self):
        result = validate_normalized_value(0.5)
        assert result == 0.5

    def test_returns_value_when_near_zero(self):
        result = validate_normalized_value(0.001)
        assert result == 0.001

    def test_returns_value_when_near_one(self):
        result = validate_normalized_value(0.999)
        assert result == 0.999

    def test_raises_error_when_string(self):
        with pytest.raises(SchemaValidateError, match="float or None"):
            validate_normalized_value("0.5")

    def test_raises_error_when_int(self):
        with pytest.raises(SchemaValidateError, match="float or None"):
            validate_normalized_value(1)

    def test_raises_error_when_bool(self):
        with pytest.raises(SchemaValidateError, match="float or None"):
            validate_normalized_value(True)

    def test_raises_error_when_zero_boundary(self):
        with pytest.raises(SchemaValidateError, match="range \\(0, 1\\)"):
            validate_normalized_value(0.0)

    def test_raises_error_when_one_boundary(self):
        with pytest.raises(SchemaValidateError, match="range \\(0, 1\\)"):
            validate_normalized_value(1.0)

    def test_raises_error_when_negative(self):
        with pytest.raises(SchemaValidateError, match="range \\(0, 1\\)"):
            validate_normalized_value(-0.1)

    def test_raises_error_when_greater_than_one(self):
        with pytest.raises(SchemaValidateError, match="range \\(0, 1\\)"):
            validate_normalized_value(1.5)


# ============================== TestIsStringList ==============================


class TestIsStringList:
    """is_string_list 的外部接口测试"""

    def test_returns_empty_list_when_input_empty(self):
        result = is_string_list([])
        assert not result

    def test_returns_list_when_all_strings(self):
        result = is_string_list(["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_returns_list_when_contains_empty_string(self):
        result = is_string_list(["123", "test", ""])
        assert result == ["123", "test", ""]

    def test_raises_error_when_string(self):
        with pytest.raises(SchemaValidateError, match="list type"):
            is_string_list("not a list")

    def test_raises_error_when_int(self):
        with pytest.raises(SchemaValidateError, match="list type"):
            is_string_list(123)

    def test_raises_error_when_tuple(self):
        with pytest.raises(SchemaValidateError, match="list type"):
            is_string_list(("a", "b"))

    def test_raises_error_when_contains_int(self):
        with pytest.raises(SchemaValidateError, match="string types"):
            is_string_list([1, "a", "b"])

    def test_raises_error_when_contains_bool(self):
        with pytest.raises(SchemaValidateError, match="string types"):
            is_string_list(["a", True, "b"])

    def test_raises_error_when_contains_float(self):
        with pytest.raises(SchemaValidateError, match="string types"):
            is_string_list(["a", 3.14, "b"])

    def test_raises_error_when_contains_none(self):
        with pytest.raises(SchemaValidateError, match="string types"):
            is_string_list(["a", None, "b"])


# ============================== TestNonEmptyString ==============================


class TestNonEmptyString:
    """non_empty_string 的外部接口测试"""

    def test_returns_string_when_valid(self):
        result = non_empty_string("hello", field_name="value")
        assert result == "hello"

    def test_returns_string_when_has_spaces(self):
        result = non_empty_string("  spaced  ", field_name="prompt")
        assert result == "  spaced  "

    def test_raises_error_when_none(self):
        with pytest.raises(SchemaValidateError, match="prompt must not be null"):
            non_empty_string(None, field_name="prompt")

    def test_raises_error_when_empty(self):
        with pytest.raises(SchemaValidateError, match="value must be a non-empty string"):
            non_empty_string("", field_name="value")

    def test_raises_error_when_whitespace_only(self):
        with pytest.raises(SchemaValidateError, match="name must be a non-empty string"):
            non_empty_string("   ", field_name="name")


# ============================== TestAllowEmptyList ==============================


class TestAllowEmptyList:
    """allow_empty_list 的外部接口测试"""

    def test_returns_empty_list_when_input_empty(self):
        result = allow_empty_list([])
        assert not result

    def test_returns_list_when_non_empty(self):
        result = allow_empty_list([1, 2, 3])
        assert result == [1, 2, 3]

    def test_raises_error_when_string(self):
        with pytest.raises(SchemaValidateError, match="list type"):
            allow_empty_list("not a list")

    def test_raises_error_when_int(self):
        with pytest.raises(SchemaValidateError, match="list type"):
            allow_empty_list(123)

    def test_raises_error_when_none(self):
        with pytest.raises(SchemaValidateError, match="list type"):
            allow_empty_list(None)


# ============================== TestAllowEmptyDict ==============================


class TestAllowEmptyDict:
    """allow_empty_dict 的外部接口测试"""

    def test_returns_empty_dict_when_input_empty(self):
        result = allow_empty_dict({})
        assert not result

    def test_returns_dict_when_non_empty(self):
        result = allow_empty_dict({"a": 1})
        assert result == {"a": 1}

    def test_raises_error_when_string(self):
        with pytest.raises(SchemaValidateError, match="dict type"):
            allow_empty_dict("not a dict")

    def test_raises_error_when_int(self):
        with pytest.raises(SchemaValidateError, match="dict type"):
            allow_empty_dict(123)

    def test_raises_error_when_none(self):
        with pytest.raises(SchemaValidateError, match="dict type"):
            allow_empty_dict(None)

    def test_raises_error_when_list(self):
        with pytest.raises(SchemaValidateError, match="dict type"):
            allow_empty_dict([])


# ============================== TestAtLeastOneKey ==============================


class TestAtLeastOneKey:
    """at_least_one_key 的外部接口测试"""

    def test_returns_dict_when_single_key(self):
        result = at_least_one_key({"a": 1})
        assert result == {"a": 1}

    def test_returns_dict_when_multiple_keys(self):
        result = at_least_one_key({"x": "y", "z": 0})
        assert result == {"x": "y", "z": 0}

    def test_raises_error_when_empty_dict(self):
        with pytest.raises(SchemaValidateError, match="at least one key"):
            at_least_one_key({})

    def test_raises_error_when_string(self):
        with pytest.raises(SchemaValidateError, match="dict type"):
            at_least_one_key("not a dict")

    def test_raises_error_when_none(self):
        with pytest.raises(SchemaValidateError, match="dict type"):
            at_least_one_key(None)


# ============================== TestInRange ==============================


class TestInRange:
    """in_range 的外部接口测试"""

    def test_returns_value_when_in_range(self):
        result = in_range(5, min_val=0, max_val=10)
        assert result == 5

    def test_returns_value_when_at_min(self):
        result = in_range(0, min_val=0, max_val=10)
        assert result == 0

    def test_returns_value_when_at_max(self):
        result = in_range(10, min_val=0, max_val=10)
        assert result == 10

    def test_returns_value_when_no_min(self):
        result = in_range(5, min_val=None, max_val=10)
        assert result == 5

    def test_returns_value_when_no_max(self):
        result = in_range(5, min_val=0, max_val=None)
        assert result == 5

    def test_raises_error_when_below_min(self):
        with pytest.raises(SchemaValidateError, match="greater than or equal to 0"):
            in_range(-1, min_val=0, max_val=10)

    def test_raises_error_when_above_max(self):
        with pytest.raises(SchemaValidateError, match="less than or equal to 10"):
            in_range(11, min_val=0, max_val=10)

    def test_raises_error_when_string(self):
        with pytest.raises(SchemaValidateError, match="must be a number"):
            in_range("5", min_val=0, max_val=10)

    def test_raises_error_when_none(self):
        with pytest.raises(SchemaValidateError, match="must be a number"):
            in_range(None, min_val=0, max_val=10)

    def test_raises_error_when_list(self):
        with pytest.raises(SchemaValidateError, match="must be a number"):
            in_range([1], min_val=0, max_val=10)
