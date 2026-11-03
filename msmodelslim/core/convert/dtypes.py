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

Checkpoint dtype names used by convert routing.

Safetensors headers (``F8_E4M3``, ``U8``) and torch names (``float8_e4m3fn``)
are treated as the same dtype.
"""

from __future__ import annotations

_UINT8 = frozenset({"u8", "uint8"})
_INT8 = frozenset({"i8", "int8"})
_INT32 = frozenset({"i32", "int32"})
_E4M3 = frozenset({"f8_e4m3", "float8_e4m3fn", "float8_e4m3fnuz"})
_E8M0 = frozenset({"f8_e8m0", "float8_e8m0fnu"})


def canon_dtype(dtype: str | None) -> str:
    """Strip a ``torch.`` prefix and compare case-insensitively."""
    text = (dtype or "").strip()
    lowered = text.lower()
    if lowered.startswith("torch."):
        text = text.split(".", 1)[1]
    return text.lower()


def is_uint8(dtype: str | None) -> bool:
    return canon_dtype(dtype) in _UINT8


def is_int8(dtype: str | None) -> bool:
    return canon_dtype(dtype) in _INT8


def is_int32(dtype: str | None) -> bool:
    return canon_dtype(dtype) in _INT32


def is_e4m3(dtype: str | None) -> bool:
    return canon_dtype(dtype) in _E4M3


def is_e8m0(dtype: str | None) -> bool:
    return canon_dtype(dtype) in _E8M0
