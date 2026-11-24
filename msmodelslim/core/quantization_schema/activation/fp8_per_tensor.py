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

from msmodelslim.utils.logging import logger_setter

from ..auto import AutoFakeQuantActivation
from ..const import fp8_e4m3_per_tensor_sym
from ..qal import QABCRegistry
from .per_tensor import FakeQuantActivationPerTensor


@QABCRegistry.multi_register(dispatch_key=[fp8_e4m3_per_tensor_sym], abc_type=AutoFakeQuantActivation)
@logger_setter()
class FP8FakeQuantActivationPerTensor(FakeQuantActivationPerTensor):
    """FP8 E4M3 对称 per-tensor 伪量化/反量化。"""

    pass
