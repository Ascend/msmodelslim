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

from msmodelslim.core.context.summary.utils import BASELINE_BITS, format_bit_set, logical_bits
from msmodelslim.core.quantization_schema.qal import QDType


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (QDType.INT8, 8),
        (QDType.FP8_E4M3, 8),
        (QDType.MXFP8, 8),
        (QDType.INT4, 4),
        (QDType.MXFP4, 4),
    ],
)
def test_logical_bits_returns_width_when_dtype_is_supported(dtype, expected):
    assert logical_bits(dtype) == expected


def test_logical_bits_returns_none_when_dtype_is_not_mapped():
    assert logical_bits(QDType.FLOAT) is None


def test_logical_bits_returns_none_when_dtype_is_not_a_qdtype():
    assert logical_bits("not-a-dtype") is None


def test_format_bit_set_returns_baseline_when_no_width_is_given():
    assert format_bit_set([]) == "%dbit" % BASELINE_BITS


def test_format_bit_set_returns_single_width_when_all_values_are_equal():
    assert format_bit_set([8, 8]) == "8bit"


def test_format_bit_set_sorts_and_dedups_when_multiple_widths_are_given():
    assert format_bit_set([8, 4, 8]) == "{4, 8}bit"
