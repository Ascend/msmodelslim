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

from unittest.mock import Mock

from msmodelslim.utils.cache import memory


def test_load_cached_initializes_once_and_reuses_value():
    memory.clear_cache()
    init_func = Mock(return_value={"ready": True})

    first = memory.load_cached("model-a", init_func, args=("checkpoint",), kwargs={"dtype": "float16"})
    second = memory.load_cached("model-a", Mock(side_effect=AssertionError("must not initialize twice")))

    assert first == {"ready": True}
    assert second is first
    init_func.assert_called_once_with("checkpoint", dtype="float16")


def test_clear_cache_removes_one_key_or_all_keys():
    memory.clear_cache()
    first_init = Mock(return_value=1)
    second_init = Mock(return_value=2)
    memory.load_cached("first", first_init)
    memory.load_cached("second", second_init)

    memory.clear_cache("first")
    assert memory.load_cached("first", Mock(return_value=3)) == 3
    assert memory.load_cached("second", Mock(return_value=4)) == 2

    memory.clear_cache()
    assert memory.load_cached("first", Mock(return_value=5)) == 5
    assert memory.load_cached("second", Mock(return_value=6)) == 6
