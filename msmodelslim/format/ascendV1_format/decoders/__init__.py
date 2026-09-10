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

Per-IR format decoders for AscendV1. Each file handles one IR class (or a small
group sharing the same quantization shape).
"""

from .base import Fa3IrDecoder, IrDecoder, KvCacheIrDecoder, LinearIrDecoder
from .fa3_per_block import Fa3PerBlockDecoder
from .fa3_per_head import Fa3PerHeadDecoder
from .fa3_per_token import Fa3PerTokenDecoder
from .kv_cache_c8 import KvCacheC8Decoder
from .w8a8_dynamic import W8A8DynamicDecoder
from .w8a8_mx import W8A8MxDecoder
from .w8a8_static import W8A8StaticDecoder

__all__ = [
    "IrDecoder",
    "LinearIrDecoder",
    "Fa3IrDecoder",
    "KvCacheIrDecoder",
    "W8A8StaticDecoder",
    "W8A8DynamicDecoder",
    "W8A8MxDecoder",
    "Fa3PerHeadDecoder",
    "Fa3PerTokenDecoder",
    "Fa3PerBlockDecoder",
    "KvCacheC8Decoder",
]
