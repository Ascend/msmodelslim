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

from msmodelslim.utils.validation.pydantic import at_least_one_element, in_range


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
