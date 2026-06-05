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

from msmodelslim.ir.qal import QABCRegistry
from msmodelslim.utils.logging import logger_setter
from .auto import AutoFakeQuantActivation
from .const import fp8_e4m3_per_head_sym
from .activation_static import FakeQuantActivationPerHead


@QABCRegistry.multi_register(dispatch_key=[fp8_e4m3_per_head_sym], abc_type=AutoFakeQuantActivation)
@logger_setter()
class FP8FakeQuantActivationPerHead(FakeQuantActivationPerHead):
    """对称 per-head 伪量化/反量化。

    输入形状: (batch_size, num_head, seq_len, head_dim)，按第 1 维做 per-head。
    仅需 ext['scale']，忽略 offset。
    """

    # 所有实现已在基类中提供
    pass
