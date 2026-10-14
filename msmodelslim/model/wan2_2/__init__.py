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
"""

from .model_adapter import Wan2Point2Adapter
from .base_model_adapter import Wan2_2BaseModelAdapter
from .t2v.model_adapter import Wan2_2T2VModelAdapter
from .i2v.model_adapter import Wan2_2I2VModelAdapter
from .ti2v.model_adapter import Wan2_2TI2VModelAdapter
from .expert_sub_adapter import (
    Wan2_2ExpertSubAdapter,
    Wan2_2HighNoiseSubAdapter,
    Wan2_2LowNoiseSubAdapter,
)

__all__ = [
    "Wan2Point2Adapter",
    "Wan2_2BaseModelAdapter",
    "Wan2_2T2VModelAdapter",
    "Wan2_2I2VModelAdapter",
    "Wan2_2TI2VModelAdapter",
    "Wan2_2ExpertSubAdapter",
    "Wan2_2LowNoiseSubAdapter",
    "Wan2_2HighNoiseSubAdapter",
]
