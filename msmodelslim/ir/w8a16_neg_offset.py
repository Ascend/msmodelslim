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
import torch.nn.functional as F

from msmodelslim.ir.api import dequantize
from msmodelslim.ir.auto import AutoFakeQuantLinear
from msmodelslim.ir.const import float_per_tensor_sym, int8_per_channel_asym, int8_per_channel_asym_neg_offset
from msmodelslim.ir.qal import QABCRegistry, QDType, QParam, QStorage
from msmodelslim.ir.w8a16_static import W8A16StaticPerChannelFakeQuantLinear
from msmodelslim.utils.logging import logger_setter


@QABCRegistry.multi_register(
    dispatch_key=[
        (float_per_tensor_sym, int8_per_channel_asym_neg_offset),
    ],
    abc_type=AutoFakeQuantLinear,
)
@logger_setter()
class W8A16PerChannelNegOffsetFakeQuantLinear(W8A16StaticPerChannelFakeQuantLinear):
    """
    W8A16 per-channel 非对称量化、且 offset 取反语义的伪量化 IR。

    可以用以下参数描述：
        weight_scale: 权重张量的量化参数，类型为torch.Tensor, dtype为torch.float32
        weight_offset: 权重张量的量化参数，类型为torch.Tensor, dtype为torch.float32
        weight: 权重张量，类型为torch.Tensor, dtype为torch.int8
        bias: 偏置张量，类型为torch.Tensor, dtype为torch.float32
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q_param = QParam(
            scheme=int8_per_channel_asym,
            ext={
                "scale": self.weight_scale.data,
                "offset": -self.weight_offset.data,
            },
        )
        weight_q_dq = dequantize(QStorage(dtype=QDType.INT8, value=self.weight.data).T, w_q_param).T
        return F.linear(x, weight_q_dq.value, self.bias)
