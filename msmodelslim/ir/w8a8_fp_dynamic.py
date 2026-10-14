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
# pylint: disable=useless-import-alias

import torch
from torch import nn as nn
from torch.nn import functional as F

from msmodelslim.ir.api import calculate_qparam, fake_quantize, dequantize
from msmodelslim.ir.qal import QABCRegistry, QScope, QScheme, QParam, QStorage, QDType
from msmodelslim.ir import AutoFakeQuantLinear
from msmodelslim.ir.const import fp8_e4m3_per_token_sym, fp8_e4m3_per_channel_sym
from msmodelslim.utils.logging import logger_setter
from msmodelslim.utils.exception import SchemaValidateError


@QABCRegistry.multi_register(
    dispatch_key=[
        (fp8_e4m3_per_token_sym, fp8_e4m3_per_channel_sym),
    ],
    abc_type=AutoFakeQuantLinear,
)
@logger_setter()
class WFP8AFP8DynamicPerChannelFakeQuantLinear(AutoFakeQuantLinear):
    """
    W8A8FP8 per-channel/per-token sym/sym 动态对称量化方式的伪量化IR。
    """

    def __init__(self, x_q_param: QParam, w_q_param: QParam, w_q: QStorage, bias: torch.Tensor):
        super().__init__()
        if "scale" not in w_q_param.ext:
            raise SchemaValidateError("scale is not in w_q_param.ext", action="Please check the w_q_param")
        self.weight_scale = nn.Parameter(w_q_param.ext["scale"].detach(), requires_grad=False)
        self.weight = nn.Parameter(w_q.value.detach(), requires_grad=False)
        self.bias = nn.Parameter(bias.detach(), requires_grad=False) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x_reshape = x.reshape(-1, x_shape[-1])
        x_token_min = torch.amin(x_reshape, dim=0)
        x_token_max = torch.amax(x_reshape, dim=0)
        x_q_param = calculate_qparam(x_token_min, x_token_max, QDType.FP8_E4M3, QScope.PER_TOKEN, True)
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x_reshape), x_q_param)
        w_q_param = QParam(
            scheme=QScheme(scope=QScope.PER_CHANNEL, dtype=QDType.FP8_E4M3, symmetric=True),
            ext={"scale": self.weight_scale.data},
        )
        weight_q_dq = dequantize(QStorage(dtype=QDType.FP8_E4M3, value=self.weight.data).T, w_q_param).T
        return F.linear(x_q_dq.value.reshape(x_shape), weight_q_dq.value, self.bias)
