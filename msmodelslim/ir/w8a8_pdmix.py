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

from enum import Enum

import torch
from torch.nn import functional as F

from msmodelslim.ir.api import calculate_qparam, fake_quantize, dequantize
from msmodelslim.ir.qal import QABCRegistry, QScope, QScheme, QParam, QStorage, QDType
from msmodelslim.utils.logging import logger_setter
from .auto import AutoFakeQuantLinear
from .const import int8_pd_mix_asym, int8_per_channel_sym
from .w8a8_static import W8A8StaticFakeQuantLinear


class PDMixState(str, Enum):
    PREFILLING = "prefilling"
    DECODING = "decoding"


@QABCRegistry.multi_register(
    dispatch_key=[
        (int8_pd_mix_asym, int8_per_channel_sym),
    ],
    abc_type=AutoFakeQuantLinear,
)
@logger_setter()
class W8A8PDMixFakeQuantLinear(W8A8StaticFakeQuantLinear):
    """
    W8A8 pdmix sym/asym PDMIX混合量化方式的伪量化IR。

    W8A8 PDMIX混合量化方式可以用以下参数描述：
        input_scale: 输入张量的量化参数，类型为torch.Tensor, dtype为torch.float32
        input_offset: 输入张量的量化参数，类型为torch.Tensor, dtype为torch.int32
        weight_scale: 权重张量的量化参数，类型为torch.Tensor, dtype为torch.float32
        weight: 权重张量，类型为torch.Tensor, dtype为torch.int8
        bias: 偏置张量，类型为torch.Tensor, dtype为torch.float32
        state: 状态，类型为PDMixState, 可选值为PDMixState.PREFILLING和PDMixState.DECODING
    """

    def __init__(self, x_q_param: QParam, w_q_param: QParam, w_q: QStorage, bias: torch.Tensor):
        super().__init__(x_q_param, w_q_param, w_q, bias)
        self.state = PDMixState.PREFILLING

    def set_state(self, state: PDMixState):
        self.state = state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # decoding as static
        if self.state == PDMixState.DECODING:
            return super().forward(x)

        # prefilling as dynamic
        x_shape = x.shape
        x_reshape = x.reshape(-1, x_shape[-1])
        x_token_min = torch.amin(x_reshape, dim=0)
        x_token_max = torch.amax(x_reshape, dim=0)
        x_q_param = calculate_qparam(x_token_min, x_token_max, QDType.INT8, QScope.PER_TOKEN, True)
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x_reshape), x_q_param)
        w_q_param = QParam(
            scheme=QScheme(scope=QScope.PER_CHANNEL, dtype=QDType.INT8, symmetric=True),
            ext={"scale": self.weight_scale.data},
        )
        weight_q_dq = dequantize(QStorage(dtype=QDType.INT8, value=self.weight.data).T, w_q_param).T
        return F.linear(x_q_dq.value.reshape(x_shape), weight_q_dq.value, self.bias)
