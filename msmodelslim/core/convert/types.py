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

Shared value types for offline weight conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class IRKind(str, Enum):
    """Canonical intermediate-representation node names for routing."""

    FLOAT = "FLOAT"
    FP8_BLOCK = "FP8_BLOCK"
    W8A8_MXFP8 = "W8A8_MXFP8"
    W4A4_MXFP4 = "W4A4_MXFP4"
    W4A8_MXFP8 = "W4A8_MXFP8"
    INT4_PACKED = "INT4_PACKED"
    NVFP4_MODELOPT = "NVFP4_MODELOPT"
    HIFP4 = "HIFP4"
    UNKNOWN = "UNKNOWN"


class TensorRole(str, Enum):
    """Role of a tensor within a logical weight group."""

    MAIN_WEIGHT = "main_weight"
    BIAS = "bias"
    SCALE = "scale"
    OFFSET = "offset"
    SHAPE_META = "shape_meta"
    PACKED = "packed"
    GLOBAL_SCALE = "global_scale"
    OTHER = "other"


class LossLevel(str, Enum):
    """Whether a transform edge is lossless for weights."""

    LOSSLESS = "lossless"
    LOSSY = "lossy"


@dataclass(frozen=True)
class TensorRef:
    """
    Reference to a checkpoint tensor without loading payload.

    Used by virtual modules' ``tensor_bindings`` and preprocess planners.
    """

    logical_name: str
    key: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    role: TensorRole | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceIR:
    """
    Resolved source IR for a virtual module, with optional inference evidence.
    """

    kind: IRKind
    source_format: str | None = None
    confidence: float = 1.0
    evidence: list[str] = field(default_factory=list)
