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
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""


__all__ = [
    'BaseConfig',
    'WeightActivationQuantConfig',
    'WeightQuantConfig',
    'SparseQuantConfig',
    'KVQuantConfig',
    'SimulateTPConfig',
    'FAQuantConfig',
    'TimestepQuantConfig',
]

from .base_config import BaseConfig
from .weight_activation_quant_config import WeightActivationQuantConfig
from .weight_quant_config import WeightQuantConfig
from .sparse_quant_config import SparseQuantConfig
from .kv_quant_config import KVQuantConfig
from .fa_quant_config import FAQuantConfig
from .simulate_tp_config import SimulateTPConfig
from .timestep_quant_config import TimestepQuantConfig