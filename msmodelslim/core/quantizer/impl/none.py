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

import torch

import msmodelslim.ir as qir
from msmodelslim.ir.qal import QABCRegistry, QScheme, QParam
from msmodelslim.utils.logging import logger_setter
from ..base import AutoActQuantizer, QConfig


@QABCRegistry.multi_register(dispatch_key=[(qir.float_per_tensor_sym, "none")], abc_type=AutoActQuantizer)
@logger_setter()
class ActPerTensorNone(AutoActQuantizer):
    def __init__(self, config: QConfig):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def is_data_free(self) -> bool:
        """
        空实现，data_free设置为True
        """
        return True

    def get_q_param(self) -> QParam:
        return QParam(
            scheme=QScheme(
                dtype=self.config.dtype,
                scope=self.config.scope,
                symmetric=self.config.symmetric,
            )
        )
