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

from msmodelslim.utils.logging import logger_setter

from ..api import calculate_qparam, fake_quantize
from ..auto import AutoFakeQuantActivation
from ..const import fp4_e2m1_per_token_sym, fp8_e4m3_per_token_sym, int8_per_token_sym
from ..qal import QABCRegistry, QDType, QParam, QScope, QStorage


@QABCRegistry.multi_register(
    dispatch_key=[fp8_e4m3_per_token_sym, int8_per_token_sym, fp4_e2m1_per_token_sym],
    abc_type=AutoFakeQuantActivation,
)
@logger_setter()
class FakeQuantActivationPerToken(AutoFakeQuantActivation):
    """对称 per-token 动态伪量化/反量化。

    输入形状: (batch_size, num_head, seq_len, head_dim)，按 token 维度（seq_len）动态计算量化参数。
    不需要静态scale，在forward中动态计算每个token的min/max并计算量化参数。
    """

    def __init__(self, x_q_param: QParam):
        super().__init__()
        self.x_q_scheme = x_q_param.scheme

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入形状: (B, H, S, D)
        # 按token维度计算min/max
        original_shape = x.shape
        # 将x reshape为 (B*H*S, D)，每个token是一行
        x_reshaped = x.reshape(-1, original_shape[-1])
        # 计算每个token的min和max，dim=1表示按行（每个token）计算
        x_token_min = torch.amin(x_reshaped, dim=1, keepdim=True)
        x_token_max = torch.amax(x_reshaped, dim=1, keepdim=True)

        # 动态计算量化参数, scale/offset为(B*H*S, 1)
        x_q_param = calculate_qparam(
            min_val=x_token_min,  # (B*H*S, 1)
            max_val=x_token_max,  # (B*H*S, 1)
            q_dtype=self.x_q_scheme.dtype,
            q_scope=QScope.PER_TOKEN,
            symmetric=True,
        )

        # 执行量化反量化
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x_reshaped), x_q_param)

        # 恢复原始形状和数据类型
        x_out = x_q_dq.value.reshape(original_shape).to(x)
        return x_out
