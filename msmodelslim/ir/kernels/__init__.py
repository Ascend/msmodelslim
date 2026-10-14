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

Offline weight-transform kernels (convert, model scripts, data-free processors).

These are **stateless tensor ops** on checkpoint layouts. They are intentionally
**not** placed under ``core/convert`` (orchestration only) nor ``ir.api`` (QFuncRegistry
quantize/dequantize dispatch for ``QStorage`` / ``QDType``).

Add new families as separate modules, e.g. ``fp8_block.py``, ``int4_packed.py``.
"""

from msmodelslim.ir.kernels.fp8_block import WEIGHT_SCALE_INV_SUFFIX, weight_dequant

__all__ = [
    "WEIGHT_SCALE_INV_SUFFIX",
    "weight_dequant",
]
