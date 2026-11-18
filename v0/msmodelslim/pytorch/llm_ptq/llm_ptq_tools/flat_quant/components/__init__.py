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
    'FakeQuantizedLinear',
    'FlatQuantizedLinear',
    'FlatNormWrapper',
    'ForwardMode',
    'FakeQuantizedLinearConfig',
    'WeightQuantizer',
    'ActivationQuantizer',
    'asym_quant',
    'asym_dequant',
    'GeneralMatrixTrans',
    'SVDSingleTransMatrix',
    'InvSingleTransMatrix',
    'DiagonalTransMatrix'
]


from .flat_linear import (
    FakeQuantizedLinear,
    FlatQuantizedLinear,
    FlatNormWrapper,
    ForwardMode,
    FakeQuantizedLinearConfig
)
from .quantizer import (
    WeightQuantizer,
    ActivationQuantizer,
    asym_quant,
    asym_dequant
)
from .trans import (
    GeneralMatrixTrans,
    SVDSingleTransMatrix,
    InvSingleTransMatrix,
    DiagonalTransMatrix
)

