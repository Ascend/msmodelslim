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

Unit tests for msmodelslim.format.common.tensor_utils.
"""

from __future__ import annotations

import pytest

from msmodelslim.format.common.tensor_utils import require_keys, squeeze_last_one, squeeze_shape
from msmodelslim.utils.exception import SchemaValidateError


class _Store:
    def __init__(self, keys):
        self._keys = set(keys)

    def has(self, key: str) -> bool:
        return key in self._keys


class TestRequireKeys:
    """Tests for require_keys."""

    def test_passes_when_all_keys_present(self):
        require_keys(_Store({"fc.weight", "fc.scale"}), "fc", ("weight", "scale"), "W8A8", "action")

    def test_raises_schema_validate_error_with_missing_key(self):
        with pytest.raises(SchemaValidateError, match="fc.scale"):
            require_keys(_Store({"fc.weight"}), "fc", ("weight", "scale"), "W8A8", "do the thing")

    def test_raises_on_first_missing_key(self):
        with pytest.raises(SchemaValidateError, match="fc.weight"):
            require_keys(_Store(set()), "fc", ("weight", "scale"), "FAQuant", "action")


class TestSqueezeShape:
    """Tests for squeeze_shape."""

    def test_removes_all_trailing_singletons(self):
        assert squeeze_shape((2, 1, 1)) == (2,)

    def test_keeps_at_least_one_dimension(self):
        assert squeeze_shape((1, 1)) == (1,)
        assert squeeze_shape((1,)) == (1,)

    def test_keeps_non_trailing_singletons(self):
        assert squeeze_shape((1, 2)) == (1, 2)


class TestSqueezeLastOne:
    """Tests for squeeze_last_one."""

    def test_removes_a_single_trailing_one(self):
        assert squeeze_last_one((2, 3, 1)) == (2, 3)

    def test_leaves_shape_when_no_trailing_one(self):
        assert squeeze_last_one((2, 3)) == (2, 3)

    def test_leaves_single_dimension(self):
        assert squeeze_last_one((1,)) == (1,)
