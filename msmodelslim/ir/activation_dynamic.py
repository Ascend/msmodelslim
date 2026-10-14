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

from msmodelslim.ir.api import fake_quantize
from msmodelslim.ir.qal import QABCRegistry, QScope, QDType, QParam, QStorage
from msmodelslim.ir.api import calculate_qparam
from msmodelslim.utils.logging import logger_setter
from .auto import AutoFakeQuantActivation
from .const import fp8_e4m3_per_token_sym, int8_per_token_sym, mxfp4_per_block_sym, mxfp8_per_block_sym
from msmodelslim.core.observer import MsMinMaxBlockObserver, MinMaxBlockObserverConfig
from msmodelslim.ir.utils import reshape_to_blocks, undo_reshape_to_blocks


@QABCRegistry.multi_register(
    dispatch_key=[
        fp8_e4m3_per_token_sym,
        int8_per_token_sym,
    ],
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


@QABCRegistry.multi_register(
    dispatch_key=[
        mxfp4_per_block_sym,
        mxfp8_per_block_sym,
    ],
    abc_type=AutoFakeQuantActivation,
)
@logger_setter()
class FakeQuantActivationPerBlock(AutoFakeQuantActivation):
    """MXFP4 per-block 动态伪量化/反量化。

    输入形状: (batch_size, num_head, seq_len, head_dim)，按 block_size=32 分块，
    每 block 计算共享指数（E8M0 shared exponent），进行 MXFP4 量化反量化。
    """

    def __init__(self, x_q_param: QParam):
        super().__init__()
        self.x_q_scheme = x_q_param.scheme
        self.mx_finfo = x_q_param.scheme.dtype.mx_finfo
        self.x_minmax_block_observer = MsMinMaxBlockObserver(MinMaxBlockObserverConfig(axes=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        # 沿最后一维 (head_dim) 分块, block_size=32
        x_reshaped, axes, _, padded_shape = reshape_to_blocks(x, [-1], self.mx_finfo.block_size)
        # 计算 per-block min/max
        shared_exp_axes = [a + 1 for a in axes]
        self.x_minmax_block_observer.update(x_reshaped, shared_exp_axes=shared_exp_axes)
        x_min_val, x_max_val = self.x_minmax_block_observer.get_min_max()

        x_q_param = calculate_qparam(
            min_val=x_min_val,
            max_val=x_max_val,
            q_dtype=self.x_q_scheme.dtype,
            q_scope=QScope.PER_BLOCK,
            symmetric=self.x_q_scheme.symmetric,
        )

        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x_reshaped), x_q_param)
        x_q_dq.value = undo_reshape_to_blocks(x_q_dq.value, padded_shape, original_shape, axes)
        return x_q_dq.value.to(x)
