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

from typing import List

import pytest
from pydantic import BaseModel, Field

from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.pydantic_model_path import (
    resolve_pydantic_model_path,
    set_pydantic_model_path,
    validate_pydantic_model_list_path,
)


class _Leaf(BaseModel):
    tags: List[str] = Field(default_factory=list)


class _Item(BaseModel):
    name: str = "item"
    leaf: _Leaf = Field(default_factory=_Leaf)


class _Root(BaseModel):
    items: List[_Item] = Field(default_factory=list)
    title: str = "root"


class TestPydanticModelPath:
    def test_resolve_pydantic_model_path_returns_value_when_valid_nested_path(self):
        root = _Root(items=[_Item(leaf=_Leaf(tags=["a", "b"]))])
        assert resolve_pydantic_model_path(root, "items.0.leaf.tags") == ["a", "b"]

    def test_set_pydantic_model_path_returns_new_instance_when_value_updated(self):
        root = _Root(items=[_Item(leaf=_Leaf(tags=["a"]))])
        updated = set_pydantic_model_path(root, "items.0.leaf.tags", ["x", "y"])
        assert resolve_pydantic_model_path(updated, "items.0.leaf.tags") == ["x", "y"]
        assert resolve_pydantic_model_path(root, "items.0.leaf.tags") == ["a"]

    def test_validate_pydantic_model_list_path_passes_when_path_targets_list(self):
        root = _Root(items=[_Item()])
        validate_pydantic_model_list_path(root, "items.0.leaf.tags")

    def test_validate_pydantic_model_list_path_raises_SchemaValidateError_when_path_not_list(self):
        root = _Root()
        with pytest.raises(SchemaValidateError) as exc_info:
            validate_pydantic_model_list_path(root, "title")
        assert "list" in str(exc_info.value).lower()

    def test_resolve_pydantic_model_path_raises_SchemaValidateError_when_index_out_of_range(self):
        root = _Root(items=[_Item()])
        with pytest.raises(SchemaValidateError):
            resolve_pydantic_model_path(root, "items.9.leaf.tags")

    def test_resolve_pydantic_model_path_raises_SchemaValidateError_when_segment_missing(self):
        root = _Root()
        with pytest.raises(SchemaValidateError):
            resolve_pydantic_model_path(root, "missing.field")

    def test_resolve_pydantic_model_path_raises_SchemaValidateError_when_path_empty(self):
        root = _Root()
        with pytest.raises(SchemaValidateError):
            resolve_pydantic_model_path(root, "")
