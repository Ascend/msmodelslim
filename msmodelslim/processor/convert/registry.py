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

Register data-free convert processors on the global IR router.
"""

from __future__ import annotations

from msmodelslim.core.convert.router import IRRouter
from msmodelslim.processor.convert.dequant_to_float import DequantToFloatProcessor
from msmodelslim.processor.convert.int4_to_float import Int4PackedToFloatProcessor
from msmodelslim.processor.convert.mxfp8_quant import MxFp8QuantProcessor

_REGISTERED = False


def register_convert_processors(router: IRRouter | None = None) -> IRRouter:
    global _REGISTERED
    r = router or IRRouter.default()
    if _REGISTERED and router is None:
        return r
    for proc in (DequantToFloatProcessor(), Int4PackedToFloatProcessor(), MxFp8QuantProcessor()):
        r.register_processor(proc)
    _REGISTERED = True
    return r
