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

modelslim_convert quant_service：离线权重格式转换（data-free，不经 runner 校准）。
"""

__all__ = [
    "ModelslimConvertQuantService",
    "ModelslimConvertQuantServiceConfig",
    "ModelslimConvertQuantConfig",
]

from .quant_config import ModelslimConvertQuantConfig
from .quant_service import ModelslimConvertQuantService, ModelslimConvertQuantServiceConfig
