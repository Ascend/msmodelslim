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

import sys

import pydantic
import pytest

from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.patch.pydantic import patch_pydantic


@pytest.fixture
def _restored_pydantic_fixture():
    original_base_model = pydantic.BaseModel
    pydantic_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name.startswith("pydantic") and hasattr(module, "BaseModel")
    }
    original_values = {name: module.BaseModel for name, module in pydantic_modules.items()}
    patch_pydantic()
    try:
        yield
    finally:
        pydantic.BaseModel = original_base_model
        for name, original in original_values.items():
            sys.modules[name].BaseModel = original


def test_patch_pydantic_converts_init_and_extra_field_errors(_restored_pydantic_fixture):
    class StrictModel(pydantic.BaseModel):
        count: int

    assert StrictModel(count="3").count == 3

    with pytest.raises(SchemaValidateError, match="1 validation error\\(s\\) found") as exc_info:
        StrictModel(count="not-an-int")
    assert "count" in str(exc_info.value)

    with pytest.raises(SchemaValidateError, match="Extra inputs are not permitted"):
        StrictModel(count=1, unexpected=True)


def test_patch_pydantic_converts_validate_and_json_errors(_restored_pydantic_fixture):
    class JsonModel(pydantic.BaseModel):
        enabled: bool

    assert JsonModel.model_validate({"enabled": "true"}).enabled is True

    with pytest.raises(SchemaValidateError, match="1 validation error\\(s\\) found"):
        JsonModel.model_validate({"enabled": "not-a-bool"})

    with pytest.raises(SchemaValidateError, match="1 validation error\\(s\\) found"):
        JsonModel.model_validate_json('{"enabled": "not-a-bool"}')
