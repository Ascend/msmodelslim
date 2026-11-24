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

from .per_head import FakeQuantActivationPerHead
from .per_tensor import FakeQuantActivationPerTensor
from .per_token import FakeQuantActivationPerToken
from .per_block import FakeQuantActivationPerBlock
from .hif4_per_block import HIF4FakeQuantActivationPerBlock
from .fp8_per_head import FP8FakeQuantActivationPerHead
from .fp8_per_tensor import FP8FakeQuantActivationPerTensor
from .hif8_per_head import HIF8FakeQuantActivationPerHead
from .hif8_per_tensor import HIF8FakeQuantActivationPerTensor
from .int8_per_head import INT8FakeQuantActivationPerHead
from .mxfp8_per_channel import MXFP8FakeQuantActivationPerChannel

__all__ = [
    "FakeQuantActivationPerHead",
    "FakeQuantActivationPerTensor",
    "FakeQuantActivationPerToken",
    "FakeQuantActivationPerBlock",
    "HIF4FakeQuantActivationPerBlock",
    "FP8FakeQuantActivationPerHead",
    "FP8FakeQuantActivationPerTensor",
    "HIF8FakeQuantActivationPerHead",
    "HIF8FakeQuantActivationPerTensor",
    "INT8FakeQuantActivationPerHead",
    "MXFP8FakeQuantActivationPerChannel",
]
