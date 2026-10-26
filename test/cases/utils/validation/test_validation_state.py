#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# pylint: disable=use-implicit-booleaness-not-comparison

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

import threading

from msmodelslim.utils.validation.validation_state import (
    add_validation_error,
    clear_additional_validation_errors,
    get_additional_validation_errors,
    get_validation_context,
    set_validation_context,
)


def test_validation_context_round_trips_path_prefix():
    set_validation_context()
    assert get_validation_context() == {"path_prefix": ""}

    set_validation_context("quantization.")
    assert get_validation_context() == {"path_prefix": "quantization."}


def test_additional_validation_errors_accumulate_and_clear():
    clear_additional_validation_errors()
    assert get_additional_validation_errors() == []

    add_validation_error("first")
    add_validation_error("second")
    assert get_additional_validation_errors() == ["first", "second"]

    clear_additional_validation_errors()
    assert get_additional_validation_errors() == []


def test_validation_context_is_isolated_between_threads():
    barrier = threading.Barrier(2)
    results = {}

    def worker(name):
        set_validation_context(name)
        barrier.wait(timeout=5)
        results[name] = get_validation_context()

    threads = [threading.Thread(target=worker, args=(name,)) for name in ("first", "second")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert results == {"first": {"path_prefix": "first"}, "second": {"path_prefix": "second"}}
