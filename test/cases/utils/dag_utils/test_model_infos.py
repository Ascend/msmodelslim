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

from msmodelslim.utils.dag_utils.model_infos import ModuleType


def test_module_type_enum_exposes_stable_string_values():
    assert [(member.name, member.value) for member in ModuleType] == [
        ("CONV2D", "Conv2d"),
        ("ADD", "__add__"),
        ("IADD", "__iadd__"),
        ("LINEAR", "Linear"),
        ("BATCHNORM", "BatchNorm2d"),
        ("CONCAT", "cat"),
        ("RELU", "ReLU"),
        ("GETITEM", "__getitem__"),
        ("LAYERNORM", "Layernorm"),
    ]
    assert ModuleType("Linear") is ModuleType.LINEAR
    with pytest.raises(ValueError):
        ModuleType("Unknown")
