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

Tests for ``msmodelslim.utils.env.npu``: per-card free/total memory probing.
"""

import sys
from types import SimpleNamespace

import pytest

from msmodelslim.utils.env import get_npu_memory


def test_get_npu_memory_returns_integer_pair(monkeypatch):
    class FakeNpu:
        @staticmethod
        def mem_get_info(index):
            assert index == 2
            return 100, 200

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu=FakeNpu()))

    free, total = get_npu_memory(2)

    assert (free, total) == (100, 200)
    assert isinstance(free, int)
    assert isinstance(total, int)


@pytest.mark.parametrize("memory_info", [(0, 200), (200, 200)])
def test_get_npu_memory_accepts_boundary_values(monkeypatch, memory_info):
    class FakeNpu:
        @staticmethod
        def mem_get_info(index):
            assert index == 0
            return memory_info

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu=FakeNpu()))

    result = get_npu_memory(0)

    assert result == memory_info
    assert all(isinstance(value, int) for value in result)


@pytest.mark.parametrize("invalid_index", [None, -1, True, False, "0", 1.5])
def test_get_npu_memory_returns_minus_one_for_invalid_index(monkeypatch, invalid_index):
    class UnexpectedNpu:
        @staticmethod
        def mem_get_info(_index):
            pytest.fail("memory query should not run for an invalid index")

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu=UnexpectedNpu()))

    result = get_npu_memory(invalid_index)

    assert result == (-1, -1)
    assert all(isinstance(value, int) for value in result)


def test_get_npu_memory_returns_minus_one_when_dependency_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu", None)

    result = get_npu_memory(0)

    assert result == (-1, -1)
    assert all(isinstance(value, int) for value in result)


def test_get_npu_memory_returns_minus_one_when_query_fails(monkeypatch):
    class FakeNpu:
        @staticmethod
        def mem_get_info(_index):
            raise RuntimeError("NPU query failed")

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu=FakeNpu()))

    result = get_npu_memory(0)

    assert result == (-1, -1)
    assert all(isinstance(value, int) for value in result)


def test_get_npu_memory_returns_minus_one_when_conversion_fails(monkeypatch):
    class BadInteger:
        def __int__(self):
            raise ValueError("cannot convert to int")

    class FakeNpu:
        @staticmethod
        def mem_get_info(_index):
            return BadInteger(), BadInteger()

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu=FakeNpu()))

    result = get_npu_memory(0)

    assert result == (-1, -1)
    assert all(isinstance(value, int) for value in result)


@pytest.mark.parametrize(
    "memory_info",
    [
        None,
        100,
        (100,),
        (100, 200, 300),
        (-1, 200),
        (100, 0),
        (100, -1),
        (201, 200),
        (True, 200),
        (100, True),
        (float("nan"), 200),
        (100, float("inf")),
        (100.0, 200.0),
        (None, 200),
        (100, None),
        [],
    ],
)
def test_get_npu_memory_returns_integer_failure_pair_for_invalid_backend_result(monkeypatch, memory_info):
    class FakeNpu:
        @staticmethod
        def mem_get_info(_index):
            return memory_info

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu=FakeNpu()))

    result = get_npu_memory(0)

    assert result == (-1, -1)
    assert all(isinstance(value, int) for value in result)
