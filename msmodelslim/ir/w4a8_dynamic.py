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
# pylint: disable=consider-using-from-import

import torch
import torch.nn as nn
import torch.nn.functional as F

from msmodelslim.ir.qal import QABCRegistry, QScheme, QScope, QDType, QParam, QStorage
from msmodelslim.ir.api import dequantize, fake_quantize, calculate_qparam
from msmodelslim.ir import AutoFakeQuantLinear
from msmodelslim.utils.logging import logger_setter
from .const import int8_per_token_sym, int4_per_channel_sym


@QABCRegistry.multi_register(
    dispatch_key=[
        (int8_per_token_sym, int4_per_channel_sym),
    ],
    abc_type=AutoFakeQuantLinear,
)
@logger_setter()
class W4A8DynamicFakeQuantLinear(AutoFakeQuantLinear):
    def __init__(self, x_q_param: QParam, w_q_param: QParam, w_q: QStorage, bias: torch.Tensor):
        super().__init__()
        self.weight_scale = nn.Parameter(w_q_param.ext["scale"].detach(), requires_grad=False)
        self.weight = nn.Parameter(w_q.value.detach(), requires_grad=False)
        self.bias = nn.Parameter(bias.detach(), requires_grad=False) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x_reshape = x.reshape(-1, x_shape[-1])
        x_token_min = torch.amin(x_reshape, dim=0)
        x_token_max = torch.amax(x_reshape, dim=0)
        x_q_param = calculate_qparam(x_token_min, x_token_max, QDType.INT8, QScope.PER_TOKEN, True)
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x_reshape), x_q_param)
        w_q_param = QParam(
            scheme=QScheme(scope=QScope.PER_CHANNEL, dtype=QDType.INT4, symmetric=True),
            ext={"scale": self.weight_scale.data},
        )
        weight_q_dq = dequantize(QStorage(dtype=QDType.INT4, value=self.weight.data).T, w_q_param).T
        return F.linear(x_q_dq.value.reshape(x_shape), weight_q_dq.value, self.bias)
