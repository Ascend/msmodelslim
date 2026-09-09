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
from torch import nn

from msmodelslim.ir.api import fake_quantize
from msmodelslim.ir.qal import QABCRegistry, QDType, QParam, QStorage
from msmodelslim.utils.logging import logger_setter
from .auto import AutoFakeQuantActivation
from .const import mxfp8_per_channel_sym


@QABCRegistry.multi_register(dispatch_key=[mxfp8_per_channel_sym], abc_type=AutoFakeQuantActivation)
@logger_setter()
class MXFP8FakeQuantActivationPerChannel(AutoFakeQuantActivation):
    """FA V 分支：MXFP8 per-channel 静态伪量化。

    输入形状: (batch_size, num_head, seq_len, head_dim)。
    与参考分支 kvcache_E8M0_Scale 的 DynamicCacheQuantizer 对齐：将 head 维并入通道，
    每 (head, dim) 一个 E8M0 shared_exp，共 num_head * head_dim 个 scale。
    """

    def __init__(self, x_q_param: QParam):
        super().__init__()
        self.x_q_scheme = x_q_param.scheme

        scale = x_q_param.ext.get("scale")
        if scale is None:
            raise ValueError(f"`scale` is needed in ext but is missing for {self.__class__.__name__}")

        self.input_scale = nn.Parameter(scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, S, D) → (B, S, H, D) → (B*S, H*D)，per-channel 沿 H*D
        x_t = x.transpose(-2, -3)
        x_shape_t = x_t.shape
        x_2d = x_t.reshape(-1, x_shape_t[-1] * x_shape_t[-2])
        x_q_param = QParam(scheme=self.x_q_scheme, ext={"scale": self.input_scale.data})
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x_2d), x_q_param)
        x_out = x_q_dq.value.reshape(x_shape_t)
        return x_out.transpose(-2, -3)
