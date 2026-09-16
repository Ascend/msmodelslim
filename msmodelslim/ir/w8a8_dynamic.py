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

import torch
from torch import nn
from torch.nn import functional as F

from msmodelslim.ir import int8_per_token_sym, int8_per_channel_sym, AutoFakeQuantLinear, int8_per_group_sym
from msmodelslim.ir.api import calculate_qparam, fake_quantize, dequantize
from msmodelslim.ir.const import int8_per_group_asym
from msmodelslim.ir.qal import QABCRegistry, QScope, QScheme, QParam, QStorage, QDType
from msmodelslim.utils.logging import logger_setter


@QABCRegistry.multi_register(
    dispatch_key=[
        (int8_per_token_sym, int8_per_channel_sym),
    ],
    abc_type=AutoFakeQuantLinear,
)
class W8A8DynamicPerChannelFakeQuantLinear(AutoFakeQuantLinear):
    """
    W8A8 per-channel/per-token sym/sym 动态对称量化方式的伪量化IR。

    W8A8 静态非对称量化方式可以用以下参数描述：
        weight_scale: 权重张量的量化参数，类型为torch.Tensor, dtype为torch.float32
        weight: 权重张量，类型为torch.Tensor, dtype为torch.int8
        bias: 偏置张量，类型为torch.Tensor, dtype为torch.float32
    """

    def __init__(self, x_q_param: QParam, w_q_param: QParam, w_q: QStorage, bias: torch.Tensor):
        super().__init__()
        self.weight_scale = nn.Parameter(w_q_param.ext["scale"].detach(), requires_grad=False)
        self.weight = nn.Parameter(w_q.value.detach().to(torch.int8), requires_grad=False)
        self.bias = nn.Parameter(bias.detach(), requires_grad=False) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x_reshape = x.reshape(-1, x_shape[-1])
        x_token_min = torch.amin(x_reshape, dim=1, keepdim=True)
        x_token_max = torch.amax(x_reshape, dim=1, keepdim=True)
        x_q_param = calculate_qparam(x_token_min, x_token_max, QDType.INT8, QScope.PER_TOKEN, True)
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x_reshape), x_q_param)
        w_q_param = QParam(
            scheme=QScheme(scope=QScope.PER_CHANNEL, dtype=QDType.INT8, symmetric=True),
            ext={"scale": self.weight_scale.data},
        )
        weight_q_dq = dequantize(QStorage(dtype=QDType.INT8, value=self.weight.data).T, w_q_param).T
        return F.linear(x_q_dq.value.reshape(x_shape), weight_q_dq.value, self.bias)


@QABCRegistry.multi_register(
    dispatch_key=[
        (int8_per_token_sym, int8_per_group_sym),
        (int8_per_token_sym, int8_per_group_asym),
    ],
    abc_type=AutoFakeQuantLinear,
)
@logger_setter()
class W8A8DynamicPerGroupFakeQuantLinear(AutoFakeQuantLinear):
    def __init__(self, x_q_param: QParam, w_q_param: QParam, w_q: QStorage, bias: torch.Tensor):
        super().__init__()
        self.group_size = w_q_param.ext.pop("group_size")
        self.w_scheme = w_q_param.scheme
        self.weight_scale = nn.Parameter(w_q_param.ext.pop("scale"), requires_grad=False)
        self.weight_offset = nn.Parameter(w_q_param.ext.pop("offset"), requires_grad=False)
        self.weight = nn.Parameter(w_q.value.to(torch.int8), requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

    def __repr__(self) -> str:
        return f"W8A8DynamicPerGroupFakeQuantLinear(symmetric={self.w_scheme.symmetric}, group_size={self.group_size})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x_reshape = x.reshape(-1, x_shape[-1])
        x_token_min = torch.amin(x_reshape, dim=1, keepdim=True)
        x_token_max = torch.amax(x_reshape, dim=1, keepdim=True)
        x_q_param = calculate_qparam(x_token_min, x_token_max, QDType.INT8, QScope.PER_TOKEN, True)
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x_reshape), x_q_param)
        w_q_param = QParam(
            scheme=self.w_scheme,
            ext={"scale": self.weight_scale, "offset": self.weight_offset, "group_size": self.group_size},
        )
        w_q_storage = QStorage(dtype=QDType.INT8, value=self.weight)
        weight_q_dq = dequantize(w_q_storage, w_q_param)
        return F.linear(x_q_dq.value.reshape(x_shape), weight_q_dq.value, self.bias)
