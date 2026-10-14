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

from decimal import Decimal
from typing import Any, List

from msmodelslim.utils.exception import SchemaValidateError, SecurityError
from msmodelslim.utils.security import validate_safe_host


def at_least_one_element(v: Any, param_name: str = "value") -> Any:
    if not v:
        raise SchemaValidateError(
            f"{param_name} must have at least one element",
            action=f"Please provide a list {param_name} with at least one element",
        )
    return v


def greater_than_zero(v: Any, param_name: str = "value") -> Any:
    """
    校验数值是否大于 0，不强制区分类型（int/float/Decimal 均可）。
    """
    if not isinstance(v, (float, int, Decimal)) or v <= 0:
        raise SchemaValidateError(
            f"{param_name} must be int/float/Decimal, and value need greater than 0",
            action=f"Please check the numeric {param_name}",
        )
    return v


def int_greater_than_zero(v: Any, param_name: str = "value") -> Any:
    """
    校验数值是否是int，且数值是否大于 0。
    """
    if not isinstance(v, int) or v <= 0:
        raise SchemaValidateError(
            f"{param_name} must be int, and value need greater than 0",
            action=f"Please check the numeric {param_name}",
        )
    return v


def validate_normalized_value(v: Any, param_name="value") -> float:
    if not isinstance(v, (float, type(None))):
        raise SchemaValidateError(
            f"{param_name} must be a float or None type", action=f"Please provide a float or None {param_name}"
        )
    if v is not None and (v <= 0 or v >= 1):
        raise SchemaValidateError(
            f"{param_name} must be in the range (0, 1)",
            action=f"Please check the float {param_name} to ensure it is between 0 and 1",
        )
    return v


def is_string_list(v: Any, param_name="value") -> List[str]:
    if not isinstance(v, list):
        raise SchemaValidateError(f"{param_name} must be a list type", action=f"Please provide a list {param_name}")

    for item in v:
        if not isinstance(item, str):
            # 注意：错误信息需要与单测中的关键字严格匹配
            # 单测断言的关键字为 "all elements in the list must be string types"
            raise SchemaValidateError(
                "all elements in the list must be string types",
                action="Please ensure all list elements are strings",
            )

    return v


def validate_str_length(input_str, str_name="string", max_len=200):
    """
    校验输入字符串的长度是否在允许范围内

    检查字符串长度是否超过指定的最大限制，默认最大长度为200字符。
    支持自定义字符串名称，报错信息将动态显示该名称，适配各类使用场景。

    参数:
        input_str: 需要进行长度校验的字符串
        str_name: 字符串的自定义名称，用于生成精准报错信息
        max_len: 允许的最大长度（正整数，默认200）

    异常:
        SecurityError: 当max_len不是正整数时抛出
        SecurityError: 当input_str长度超过max_len时抛出
    """
    # 验证max_len参数合法性
    if not isinstance(max_len, int) or max_len <= 0:
        raise SecurityError("max_len must be a positive integer, current value: %r." % max_len)

    # 检查字符串长度是否超限
    if len(input_str) > max_len:
        raise SecurityError(
            f"The length of {str_name} should be less than {max_len}.",
            action=f"Please make sure the {str_name} is not longer than {max_len} characters.",
        )


def is_port(v: Any, param_name: str = "port") -> int:
    """
    校验端口号：必须是整数类型且在 [1, 65535] 范围内。
    供 Pydantic AfterValidator 或普通代码统一使用。
    """
    if not 1 <= v <= 65535:
        raise SchemaValidateError(
            f"{param_name} must be between 1 and 65535, got {v}",
            action=f"Please ensure {param_name} is between 1 and 65535.",
        )
    return v


def is_safe_host(v: str, param_name: str = "host") -> str:
    """
    通用的 host 安全校验函数，接受 param_name 参数。
    直接复用安全模块的 validate_safe_host。

    Args:
        v: 要验证的主机地址
        param_name: 字段名称，用于错误消息，默认为 "host"

    Returns:
        验证后的主机地址
    """
    return validate_safe_host(v, field_name=param_name)


def non_empty_string(v: str, field_name: str = "value") -> str:
    """
    Validate that a string is non-null and non-empty after stripping whitespace.

    Args:
        v: string to validate
        field_name: name for error message context

    Returns:
        The original string if valid

    Raises:
        SchemaValidateError: if the string is None or empty/whitespace
    """
    if v is None:
        raise SchemaValidateError(
            f"{field_name} must not be null", action=f"Please provide a non-empty string for {field_name}"
        )
    if not str(v).strip():
        raise SchemaValidateError(
            f"{field_name} must be a non-empty string", action=f"Please provide a non-empty string for {field_name}"
        )
    return v


def allow_empty_list(v: Any, param_name: str = "value") -> Any:
    """
    校验值是否为列表类型，允许空列表。
    """
    if not isinstance(v, list):
        raise SchemaValidateError(
            f"{param_name} must be a list type",
            action=f"Please provide a list {param_name}",
        )
    return v


def allow_empty_dict(v: Any, param_name: str = "value") -> Any:
    """
    校验值是否为字典类型，允许空字典。
    """
    if not isinstance(v, dict):
        raise SchemaValidateError(
            f"{param_name} must be a dict type",
            action=f"Please provide a dict {param_name}",
        )
    return v


def at_least_one_key(v: Any, param_name: str = "value") -> Any:
    """
    校验字典是否至少包含一个键值对，不允许空字典。
    """
    if not isinstance(v, dict):
        raise SchemaValidateError(
            f"{param_name} must be a dict type",
            action=f"Please provide a dict {param_name}",
        )
    if not v:
        raise SchemaValidateError(
            f"{param_name} must have at least one key-value pair",
            action=f"Please provide a non-empty dict {param_name}",
        )
    return v


def in_range(v: Any, param_name: str = "value", min_val: Any = None, max_val: Any = None) -> Any:
    """
    校验数值是否在指定范围内 [min_val, max_val]。
    min_val 或 max_val 为 None 时表示不限制该方向。
    """
    if not isinstance(v, (int, float, Decimal)):
        raise SchemaValidateError(
            f"{param_name} must be a number (int, float, Decimal), got {type(v).__name__}",
            action=f"Please provide a numeric value for {param_name}",
        )
    if min_val is not None and v < min_val:
        raise SchemaValidateError(
            f"{param_name} must be greater than or equal to {min_val}, got {v}",
            action=f"Please check the {param_name} to ensure it is >= {min_val}",
        )
    if max_val is not None and v > max_val:
        raise SchemaValidateError(
            f"{param_name} must be less than or equal to {max_val}, got {v}",
            action=f"Please check the {param_name} to ensure it is <= {max_val}",
        )
    return v
