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

from msmodelslim.core.observer import MinMaxBlockObserverConfig, MsMinMaxBlockObserver
from msmodelslim.utils.logging import logger_setter

from ..api import calculate_qparam, fake_quantize
from ..auto import AutoFakeQuantActivation
from ..const import fp4_e2m1_per_block_sym, mxfp4_per_block_sym, mxfp8_per_block_sym
from ..qal import QABCRegistry, QDType, QParam, QScope, QStorage
from ..utils import reshape_to_blocks, undo_reshape_to_blocks


@QABCRegistry.multi_register(
    dispatch_key=[
        mxfp4_per_block_sym,
        mxfp8_per_block_sym,
        fp4_e2m1_per_block_sym,
    ],
    abc_type=AutoFakeQuantActivation,
)
@logger_setter()
class FakeQuantActivationPerBlock(AutoFakeQuantActivation):
    """Dynamic per-block activation quantization along the last dimension."""

    BLOCK_SIZE = 32

    def __init__(self, x_q_param: QParam):
        super().__init__()
        self.x_q_scheme = x_q_param.scheme
        dtype = x_q_param.scheme.dtype
        self.mx_finfo = dtype.mx_finfo if dtype in (QDType.MXFP4, QDType.MXFP8) else None
        self.block_size = getattr(self.mx_finfo, "block_size", self.BLOCK_SIZE)
        self.x_axes = -1
        self.x_minmax_block_observer = MsMinMaxBlockObserver(MinMaxBlockObserverConfig(axes=self.x_axes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_reshaped, axes, _, padded_shape = reshape_to_blocks(x, [self.x_axes], self.block_size)
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
