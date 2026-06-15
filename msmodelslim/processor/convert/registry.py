#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Register data-free convert processors on the global IR router."""

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
