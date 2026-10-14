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

from msmodelslim.ir.qal import QABCRegistry, QScope, QDType, QParam, QStorage
from msmodelslim.ir.api import dequantize, fake_quantize, calculate_qparam
from msmodelslim.ir import AutoFakeQuantLinear
from msmodelslim.utils.logging import logger_setter
from .const import (
    int4_per_token_sym,
    int4_per_group_asym,
    int4_per_group_sym,
    int4_per_channel_asym,
    int4_per_channel_sym,
)


@QABCRegistry.multi_register(
    dispatch_key=[
        (int4_per_token_sym, int4_per_channel_sym),
        (int4_per_token_sym, int4_per_channel_asym),
    ],
    abc_type=AutoFakeQuantLinear,
)
@logger_setter()
class W4A4DynamicPerChannelFakeQuantLinear(AutoFakeQuantLinear):
    """
    W4A4 动态量化方式的伪量化IR。

    W4A4 动态量化方式可以用以下参数描述：
        weight_scale: 权重张量的量化参数，类型为torch.Tensor, dtype为torch.float32
        weight_offset: 权重张量的量化参数，类型为torch.Tensor, dtype为torch.int32
        weight: 权重张量，类型为torch.Tensor, dtype为torch.int4
        bias: 偏置张量，类型为torch.Tensor, dtype为torch.float32
    """

    def __init__(self, x_q_param: QParam, w_q_param: QParam, w_q: QStorage, bias: torch.Tensor):
        super().__init__()
        self.w_scheme = w_q_param.scheme
        self.weight_scale = nn.Parameter(w_q_param.ext.pop("scale"), requires_grad=False)
        self.weight_offset = nn.Parameter(w_q_param.ext.pop("offset"), requires_grad=False)
        self.weight = nn.Parameter(w_q.value.to(torch.int8), requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

    def __repr__(self) -> str:
        return f"W4A4DynamicFakeQuantLinear(w_scheme={self.w_scheme})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x_reshape = x.reshape(-1, x_shape[-1])
        x_token_min = torch.amin(x_reshape, dim=1, keepdim=True)
        x_token_max = torch.amax(x_reshape, dim=1, keepdim=True)
        x_q_param = calculate_qparam(x_token_min, x_token_max, QDType.INT4, QScope.PER_TOKEN, True)
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x_reshape), x_q_param)
        w_q_param = QParam(
            scheme=self.w_scheme,
            ext={
                "scale": self.weight_scale,
                "offset": self.weight_offset,
            },
        )
        w_q_storage = QStorage(dtype=QDType.INT4, value=self.weight)
        weight_q_dq = dequantize(w_q_storage.T, w_q_param).T
        return F.linear(x_q_dq.value.reshape(x_shape), weight_q_dq.value, self.bias)


@QABCRegistry.multi_register(
    dispatch_key=[
        (int4_per_token_sym, int4_per_group_sym),
        (int4_per_token_sym, int4_per_group_asym),
    ],
    abc_type=AutoFakeQuantLinear,
)
@logger_setter()  # use module path as logger name (msmodelslim.ir.w4a4_dynamic)
class W4A4DynamicPerGroupFakeQuantLinear(AutoFakeQuantLinear):
    """
    W4A4 动态量化方式的伪量化IR。

    W4A4 动态量化方式可以用以下参数描述：
        weight_scale: 权重张量的量化参数，类型为torch.Tensor, dtype为torch.float32
        weight_offset: 权重张量的量化参数，类型为torch.Tensor, dtype为torch.int32
        weight: 权重张量，类型为torch.Tensor, dtype为torch.int4
        bias: 偏置张量，类型为torch.Tensor, dtype为torch.float32
    """

    def __init__(self, x_q_param: QParam, w_q_param: QParam, w_q: QStorage, bias: torch.Tensor):
        super().__init__()
        self.group_size = w_q_param.ext.pop("group_size")
        self.w_scheme = w_q_param.scheme
        self.weight_scale = nn.Parameter(w_q_param.ext.pop("scale"), requires_grad=False)
        self.weight_offset = nn.Parameter(w_q_param.ext.pop("offset"), requires_grad=False)
        self.weight = nn.Parameter(w_q.value.to(torch.int8), requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

    def __repr__(self) -> str:
        return f"W4A4DynamicPerGroupFakeQuantLinear(symmetric={self.w_scheme.symmetric}, group_size={self.group_size})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x_reshape = x.reshape(-1, x_shape[-1])
        x_token_min = torch.amin(x_reshape, dim=1, keepdim=True)
        x_token_max = torch.amax(x_reshape, dim=1, keepdim=True)
        x_q_param = calculate_qparam(x_token_min, x_token_max, QDType.INT4, QScope.PER_TOKEN, True)
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x_reshape), x_q_param)
        w_q_param = QParam(
            scheme=self.w_scheme,
            ext={"scale": self.weight_scale, "offset": self.weight_offset, "group_size": self.group_size},
        )
        w_q_storage = QStorage(dtype=QDType.INT4, value=self.weight)
        weight_q_dq = dequantize(w_q_storage, w_q_param)
        return F.linear(x_q_dq.value.reshape(x_shape), weight_q_dq.value, self.bias)
