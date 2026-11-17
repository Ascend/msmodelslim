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

AscendV1 format: save-side config (AscendV1QuantFormatConfig) and load-side
loader (AscendV1Format) + decoder (AscendV1FormatDecoder).
"""

from .ascendV1 import AscendV1QuantFormatConfig
from .format import ASCENDV1_DESC_JSON_NAME, AscendV1Format

__all__ = [
    "AscendV1QuantFormatConfig",
    "AscendV1Format",
    "ASCENDV1_DESC_JSON_NAME",
]
