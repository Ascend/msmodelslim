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
from torch import nn as nn
from torch.nn import functional as F

from msmodelslim.ir.api import calculate_qparam, fake_quantize, dequantize
from msmodelslim.ir.qal import QABCRegistry, QScope, QScheme,  QParam, QStorage, QDType
from msmodelslim.ir import AutoFakeQuantLinear
from msmodelslim.utils.logging import logger_setter

from msmodelslim.ir import mxfp4_per_block_sym
from msmodelslim.core.observer import MsMinMaxBlockObserver, MinMaxBlockObserverConfig

from msmodelslim.ir.utils import reshape_to_blocks, undo_reshape_to_blocks


@QABCRegistry.multi_register(
    dispatch_key=[
        (mxfp4_per_block_sym, mxfp4_per_block_sym)
    ],
    abc_type=AutoFakeQuantLinear
)
@logger_setter()
class W4A4MXDynamicPerBlockFakeQuantLinear(AutoFakeQuantLinear):

    def __init__(
            self,
            x_q_param: QParam,
            w_q_param: QParam,
            w_q: QStorage,
            bias: torch.Tensor
    ):
        super().__init__()
        self.w_scheme = w_q_param.scheme
        self.w_mx_finfo = w_q_param.scheme.dtype.mx_finfo
        self.w_axes = w_q_param.ext.get("axes")

        self.x_scheme = x_q_param.scheme
        self.x_mx_finfo = x_q_param.scheme.dtype.mx_finfo
        self.x_axes = x_q_param.ext.get("axes")

        self.weight_scale = nn.Parameter(w_q_param.ext.get("scale"), requires_grad=False)
        self.weight_offset = nn.Parameter(w_q_param.ext.get("offset"), requires_grad=False)
        self.weight = nn.Parameter(w_q.value, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

        minmax_config = MinMaxBlockObserverConfig(axes=self.x_axes)
        self.x_minmax_block_observer = MsMinMaxBlockObserver(minmax_config)

    def __repr__(self) -> str:
        return f"W4A4MXDynamicPerBlockFakeQuantLinear(symmetric={self.w_scheme.symmetric})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        axes = self.x_axes
        axes = [axes] if isinstance(axes, int) else axes
        axes = [a + x.ndim if a < 0 else a for a in axes]
        x, axes_, orig_shape, padded_shape = reshape_to_blocks(x, axes, self.x_mx_finfo.block_size)
        shared_exp_axes = [x + 1 for x in axes_] if self.x_mx_finfo.block_size > 0 else axes_
        self.x_minmax_block_observer.update(x, shared_exp_axes=shared_exp_axes)
        x_min_val, x_max_val = self.x_minmax_block_observer.get_min_max()
        x_q_param = calculate_qparam(
            x_min_val, x_max_val,
            q_dtype=self.x_scheme.dtype,
            q_scope=self.x_scheme.scope,
            symmetric=self.x_scheme.symmetric
        )
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x), x_q_param)
        x_q_dq.value = undo_reshape_to_blocks(x_q_dq.value, padded_shape, orig_shape, axes)

        w_q_storage = QStorage(dtype=self.w_scheme.dtype, value=self.weight)
        axes = self.w_axes
        axes = [axes] if isinstance(axes, int) else axes
        axes = [a + w_q_storage.value.ndim if a < 0 else a for a in axes]
        w_q_storage.value, _, w_orig_shape, w_padded_shape = reshape_to_blocks(
            w_q_storage.value, axes, self.w_mx_finfo.block_size
        )
        w_q_param = QParam(scheme=QScheme(scope=QScope.PER_BLOCK, dtype=QDType.MXFP4, symmetric=True),
                           ext={"scale": self.weight_scale.data})
        weight_q_dq = dequantize(w_q_storage, w_q_param)
        weight_q_dq.value = undo_reshape_to_blocks(weight_q_dq.value, w_padded_shape, w_orig_shape, axes)

        return F.linear(x_q_dq.value, weight_q_dq.value, self.bias)