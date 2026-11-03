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

Checkpoint 格式常量，以及 FSDP/ckpt 包装路径规范化工具。
"""

from __future__ import annotations

STRUCTURE_FILE = "qat_checkpoint.json"
WEIGHT_PREFIX = "model"
WEIGHT_FILE = f"{WEIGHT_PREFIX}.safetensors"
WEIGHT_INDEX_FILE = f"{WEIGHT_PREFIX}.safetensors.index.json"
FORMAT_NAME = "msmodelslim_qat_checkpoint"
FORMAT_VERSION = "1"

# Backward-compatible private aliases used by package exports.
_STRUCTURE_FILE = STRUCTURE_FILE
_WEIGHT_FILE = WEIGHT_FILE

_WRAPPER_NAME_MARKERS = (
    "._checkpoint_wrapped_module.",
    "._fsdp_wrapped_module.",
    "._orig_mod.",
)


def canonical_module_name(name: str) -> str:
    """Map wrapped FSDP/ckpt module paths to HF ``named_modules`` names."""
    out = name
    changed = True
    while changed:
        changed = False
        for marker in _WRAPPER_NAME_MARKERS:
            if marker in out:
                out = out.replace(marker, ".")
                changed = True
    return out


__all__ = [
    "STRUCTURE_FILE",
    "WEIGHT_PREFIX",
    "WEIGHT_FILE",
    "WEIGHT_INDEX_FILE",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "_STRUCTURE_FILE",
    "_WEIGHT_FILE",
    "canonical_module_name",
]
