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

from msmodelslim.ir.qal import QABCRegistry, QScheme, QDType, QParam, QStorage, QScope
from msmodelslim.ir.api import dequantize, fake_quantize
from msmodelslim.ir import AutoFakeQuantLinear
from msmodelslim.utils.logging import logger_setter
from .const import int8_per_tensor_asym, int8_per_tensor_sym, int8_per_channel_sym


@QABCRegistry.multi_register(
    dispatch_key=[
        (int8_per_tensor_asym, int8_per_channel_sym),
        (int8_per_tensor_sym, int8_per_channel_sym),  # 也以非对称量化表示
    ],
    abc_type=AutoFakeQuantLinear,
)
@logger_setter()
class W8A8StaticFakeQuantLinear(AutoFakeQuantLinear):
    """
    W8A8 per-channel/per-tensor sym/asym 静态非对称量化方式的伪量化IR。

    W8A8 静态非对称量化方式可以用以下参数描述：
        input_scale: 输入张量的量化参数，类型为torch.Tensor, dtype为torch.float32
        input_offset: 输入张量的量化参数，类型为torch.Tensor, dtype为torch.int32
        weight_scale: 权重张量的量化参数，类型为torch.Tensor, dtype为torch.float32
        weight: 权重张量，类型为torch.Tensor, dtype为torch.int8
        bias: 偏置张量，类型为torch.Tensor, dtype为torch.float32
    """

    def __init__(self, x_q_param: QParam, w_q_param: QParam, w_q: QStorage, bias: torch.Tensor):
        super().__init__()

        self.input_scale = nn.Parameter(x_q_param.ext["scale"], requires_grad=False)
        self.input_offset = nn.Parameter(x_q_param.ext["offset"], requires_grad=False)
        self.weight_scale = nn.Parameter(w_q_param.ext["scale"], requires_grad=False)
        self.weight = nn.Parameter(w_q.value, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q_param = QParam(
            scheme=QScheme(scope=QScope.PER_TENSOR, dtype=QDType.INT8, symmetric=False),
            ext={"scale": self.input_scale.data, "offset": self.input_offset.data},
        )
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x), x_q_param)

        w_q_param = QParam(
            scheme=QScheme(scope=QScope.PER_CHANNEL, dtype=QDType.INT8, symmetric=True),
            ext={"scale": self.weight_scale.data},
        )
        weight_q_dq = dequantize(QStorage(dtype=QDType.INT8, value=self.weight.data).T, w_q_param).T
        return F.linear(x_q_dq.value, weight_q_dq.value, self.bias)
