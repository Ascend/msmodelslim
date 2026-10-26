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

import torch
from torch import nn

from msmodelslim.utils import memory


def test_format_memory_size_uses_exact_unit_boundaries():
    assert memory.format_memory_size(0) == "0B"
    assert memory.format_memory_size(1023) == "1023B"
    assert memory.format_memory_size(1024) == "1.00KB"
    assert memory.format_memory_size(1536) == "1.50KB"
    assert memory.format_memory_size(1024 * 1024) == "1.00MB"
    assert memory.format_memory_size(1024**3) == "1.00GB"


def test_get_module_param_size_sums_parameter_storage_bytes():
    linear = nn.Linear(2, 3, bias=True)
    assert memory.get_module_param_size(linear) == 9 * 4
    assert memory.get_module_param_size(nn.Identity()) == 0


def test_device_memory_queries_select_cuda_when_npu_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_available=lambda: False), raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 4096 if device == 3 else -1)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 8192 if device == 3 else -1)

    assert memory.get_device_allocated_memory() == 4096
    assert memory.get_device_reserved_memory() == 8192
