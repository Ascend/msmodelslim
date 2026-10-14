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

from msmodelslim.cli.utils import parse_device_string
from msmodelslim.core.const import DeviceType
from msmodelslim.utils.exception import SchemaValidateError


class TestParseDeviceString:
    """Test suite for parse_device_string — 覆盖正常/边界/异常三类情形。"""

    # ---------- 正常情形 ----------

    def test_parse_device_string_returns_type_and_none_when_single_type(self):
        """主路径：仅传设备类型字符串，应返回 (type, None)。"""
        device_type, indices = parse_device_string("npu")

        assert device_type == DeviceType.NPU
        assert indices is None

    def test_parse_device_string_returns_type_and_none_when_input_has_surrounding_whitespace(self):
        """主路径：输入两端含空白，应自动 strip 后解析。"""
        device_type, indices = parse_device_string("  cpu  ")

        assert device_type == DeviceType.CPU
        assert indices is None

    def test_parse_device_string_returns_type_and_indices_when_type_with_multi_indices(self):
        """主路径：多索引逗号分隔，应完整解析。"""
        device_type, indices = parse_device_string("npu:0,1,2,3")

        assert device_type == DeviceType.NPU
        assert indices == [0, 1, 2, 3]

    def test_parse_device_string_returns_type_and_index_when_type_with_single_index(self):
        """主路径：单索引，应解析为单元素列表。"""
        device_type, indices = parse_device_string("npu:2")

        assert device_type == DeviceType.NPU
        assert indices == [2]

    # ---------- 边界情形 ----------

    def test_parse_device_string_returns_type_and_none_when_type_with_empty_indices_section(self):
        """边界：冒号后为空（如 'npu:'），应等同于只传类型。"""
        device_type, indices = parse_device_string("npu:")

        assert device_type == DeviceType.NPU
        assert indices is None

    def test_parse_device_string_returns_type_and_indices_when_indices_have_extra_whitespace(self):
        """边界：索引周围含空白，应被 strip 后正常解析。"""
        device_type, indices = parse_device_string("npu: 0 , 1 , 2 ")

        assert device_type == DeviceType.NPU
        assert indices == [0, 1, 2]

    # ---------- 异常情形 ----------

    def test_parse_device_string_raises_schema_validate_error_when_input_is_empty(self):
        """异常：空字符串应抛 SchemaValidateError。"""
        with pytest.raises(SchemaValidateError):
            parse_device_string("")

    def test_parse_device_string_raises_schema_validate_error_when_input_is_whitespace_only(self):
        """异常：纯空白应被 strip 后视为空，抛 SchemaValidateError。"""
        with pytest.raises(SchemaValidateError):
            parse_device_string("   ")

    def test_parse_device_string_raises_schema_validate_error_when_type_is_unsupported(self):
        """异常：不支持的设备类型应抛 SchemaValidateError。"""
        with pytest.raises(SchemaValidateError):
            parse_device_string("xyz")

    def test_parse_device_string_raises_schema_validate_error_when_indices_are_non_integer(self):
        """异常：索引为非整数应抛 SchemaValidateError。"""
        with pytest.raises(SchemaValidateError):
            parse_device_string("npu:abc")

    def test_parse_device_string_raises_schema_validate_error_when_indices_are_comma_only(self):
        """异常：仅含逗号/空白（如 'npu:,'、'npu: , '）应抛 SchemaValidateError。"""
        with pytest.raises(SchemaValidateError):
            parse_device_string("npu:,")

    def test_parse_device_string_raises_schema_validate_error_when_indices_mix_integer_and_non_integer(self):
        """异常：部分索引非法（如 'npu:0,abc,2'）应抛 SchemaValidateError。"""
        with pytest.raises(SchemaValidateError):
            parse_device_string("npu:0,abc,2")
