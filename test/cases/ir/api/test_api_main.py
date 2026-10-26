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

from msmodelslim.ir.api import api_main
from msmodelslim.ir.api.impl import fp_quantization, int_quantization, mx_quantization


def test_version_is_pinned():
    assert api_main.__version__ == "1.0.0"


def test_reexports_impl_module_handles():
    assert api_main.int_quantization is int_quantization
    assert api_main.fp_quantization is fp_quantization
    assert api_main.mx_quantization is mx_quantization


def test_all_is_list_of_strings_and_matches_impl_exports():
    expected = []
    for module in (int_quantization, fp_quantization, mx_quantization):
        expected.extend(getattr(module, "__all__", []))

    assert isinstance(api_main.__all__, list)
    assert all(isinstance(name, str) for name in api_main.__all__)
    assert api_main.__all__ == expected
