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

FP8_BLOCK -> W8A8_MXFP8 直转的前置判断。

``can_direct_mxfp8`` 只看 catalog 里的 dtype 与 shape。
是否调用由 ``DefaultIRTaskBuilder`` 决定，且仅在源为 FP8_BLOCK、目标为 W8A8_MXFP8 时调用。
"""

from __future__ import annotations

from msmodelslim.core.convert.dtypes import is_e4m3, is_e8m0
from msmodelslim.core.quant_service.modelslim_convert.virtual_module import ModelFreeLinear


def can_direct_mxfp8(module: ModelFreeLinear) -> bool:
    """Catalog dtype/shape only. ``0xFF`` is checked after the payload is read."""
    weight = module.tensor_bindings.get("weight")
    scale = module.tensor_bindings.get("weight_scale_inv")
    if weight is None or scale is None:
        return False
    if not is_e4m3(weight.dtype) or not is_e8m0(scale.dtype):
        return False
    if len(weight.shape) != 2 or len(scale.shape) != 2:
        return False
    rows, cols = weight.shape
    s0, s1 = scale.shape
    if s0 <= 0 or s1 <= 0 or rows % s0 != 0 or cols % s1 != 0:
        return False
    return (cols // s1) % 32 == 0
