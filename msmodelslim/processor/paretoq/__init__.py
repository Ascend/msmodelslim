#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

from msmodelslim.processor.paretoq.core import (
    WEIGHT_CLIP_INIT_AMAX,
    WEIGHT_CLIP_INIT_CHOICES,
    WEIGHT_CLIP_INIT_MEAN2_SQRT_QP,
    LsqBinaryTernaryExtension,
    StretchedElasticQuant,
    fake_quant_weight,
    init_weight_clip_from_weight,
    paretoq_weight_qp,
)
from msmodelslim.processor.paretoq.impl import ParetoQFakeQuantLinear

__all__ = [
    "ParetoQFakeQuantLinear",
    "fake_quant_weight",
    "init_weight_clip_from_weight",
    "paretoq_weight_qp",
    "LsqBinaryTernaryExtension",
    "StretchedElasticQuant",
    "WEIGHT_CLIP_INIT_AMAX",
    "WEIGHT_CLIP_INIT_MEAN2_SQRT_QP",
    "WEIGHT_CLIP_INIT_CHOICES",
]
