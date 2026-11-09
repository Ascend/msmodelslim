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

NPU memory helpers. Detection failures are reported, never raised.
"""

from __future__ import annotations

from numbers import Integral
from typing import Tuple


def get_npu_memory(index: int) -> Tuple[int, int]:
    """Return ``(free, total)`` bytes through torch_npu, or ``(-1, -1)`` on failure."""
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        return -1, -1
    try:
        import torch_npu

        free, total = torch_npu.npu.mem_get_info(index)
        if (
            isinstance(free, bool)
            or isinstance(total, bool)
            or not isinstance(free, Integral)
            or not isinstance(total, Integral)
        ):
            return -1, -1
        free_bytes, total_bytes = int(free), int(total)
        if free_bytes < 0 or total_bytes <= 0 or free_bytes > total_bytes:
            return -1, -1
        return free_bytes, total_bytes
    except Exception:
        return -1, -1
