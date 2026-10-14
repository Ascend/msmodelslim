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

from msmodelslim.ir.api import dequantize
from msmodelslim.ir.qal import QABCRegistry, QDType, QParam, QStorage
from msmodelslim.ir.qal import QScheme, QScope  # noqa: F401
from msmodelslim.ir import AutoFakeQuantLinear
from msmodelslim.utils.logging import logger_setter
from .const import (
    float_per_tensor_sym,
    int8_per_channel_sym,
    int8_per_channel_asym,
    int8_per_group_sym,
    int8_per_group_asym,
)


@QABCRegistry.multi_register(
    dispatch_key=[(float_per_tensor_sym, int8_per_channel_sym), (float_per_tensor_sym, int8_per_channel_asym)],
    abc_type=AutoFakeQuantLinear,
)
@logger_setter()
class W8A16StaticPerChannelFakeQuantLinear(AutoFakeQuantLinear):
    """
    W8A16 per-channel 静态对称/非对称量化方式的伪量化IR。

    W8A16 静态对称/非对称量化方式可以用以下参数描述：
        weight_scale: 权重张量的量化参数，类型为torch.Tensor, dtype为torch.float32
        weight_offset: 权重张量的量化参数，类型为torch.Tensor, dtype为torch.float32
        weight: 权重张量，类型为torch.Tensor, dtype为torch.int8
        bias: 偏置张量，类型为torch.Tensor, dtype为torch.float32
    """

    def __init__(self, x_q_param: QParam, w_q_param: QParam, w_q: QStorage, bias: torch.Tensor):
        super().__init__()
        self.w_scheme = w_q_param.scheme
        self.weight_scale = nn.Parameter(w_q_param.ext["scale"], requires_grad=False)
        self.weight_offset = nn.Parameter(w_q_param.ext["offset"], requires_grad=False)
        self.weight = nn.Parameter(w_q.value, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q_param = QParam(
            scheme=self.w_scheme, ext={"scale": self.weight_scale.data, "offset": self.weight_offset.data}
        )
        weight_q_dq = dequantize(QStorage(dtype=QDType.INT8, value=self.weight.data).T, w_q_param).T
        return F.linear(x, weight_q_dq.value, self.bias)


@QABCRegistry.multi_register(
    dispatch_key=[(float_per_tensor_sym, int8_per_group_sym), (float_per_tensor_sym, int8_per_group_asym)],
    abc_type=AutoFakeQuantLinear,
)
@logger_setter()
class W8A16StaticPerGroupFakeQuantLinear(AutoFakeQuantLinear):
    """
    W8A16 per_group 静态对称/非对称量化方式的伪量化IR。

    W8A16 静态对称/非对称量化方式可以用以下参数描述：
        weight_scale: 权重张量的量化参数，类型为torch.Tensor, dtype为torch.float32
        weight_offset: 权重张量的量化参数，类型为torch.Tensor, dtype为torch.float32
        weight: 权重张量，类型为torch.Tensor, dtype为torch.int8
        bias: 偏置张量，类型为torch.Tensor, dtype为torch.float32
    """

    def __init__(self, x_q_param: QParam, w_q_param: QParam, w_q: QStorage, bias: torch.Tensor):
        super().__init__()
        self.group_size = w_q_param.ext["group_size"]
        self.w_scheme = w_q_param.scheme
        self.weight_scale = nn.Parameter(w_q_param.ext["scale"], requires_grad=False)
        self.weight_offset = nn.Parameter(w_q_param.ext["offset"], requires_grad=False)
        self.weight = nn.Parameter(w_q.value, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q_param = QParam(
            scheme=self.w_scheme,
            ext={"scale": self.weight_scale.data, "offset": self.weight_offset.data, "group_size": self.group_size},
        )
        weight_q_dq = dequantize(QStorage(dtype=QDType.INT8, value=self.weight.data).T, w_q_param).T
        return F.linear(x, weight_q_dq.value, self.bias)
