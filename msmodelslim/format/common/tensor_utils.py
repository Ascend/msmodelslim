#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You may use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------

Shared tensor / shape utilities used by format decoders. No quant_type branching.
"""

from typing import Iterable, Tuple


from msmodelslim.utils.exception import SchemaValidateError


def require_keys(
    store,
    prefix: str,
    suffixes: Iterable[str],
    quant_label: str,
    action: str,
) -> None:
    """Verify that every ``{prefix}.{suffix}`` key exists in ``store``."""
    for suffix in suffixes:
        key = f"{prefix}.{suffix}"
        if not store.has(key):
            raise SchemaValidateError(
                f"Missing {quant_label} tensor '{key}'",
                action=action,
            )


def squeeze_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Remove trailing singleton dimensions (e.g. ``(N, 1)`` -> ``(N,)``)."""
    dims = list(shape)
    while len(dims) > 1 and dims[-1] == 1:
        dims.pop()
    return tuple(dims)


def squeeze_last_one(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Remove a single trailing 1 if present."""
    if len(shape) > 1 and shape[-1] == 1:
        return tuple(shape[:-1])
    return shape


__all__ = ["require_keys", "squeeze_shape", "squeeze_last_one"]
