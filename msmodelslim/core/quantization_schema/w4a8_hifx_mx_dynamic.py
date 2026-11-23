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
from torch.nn import functional as F

from msmodelslim.core.quantization_schema.api import calculate_qparam, fake_quantize, dequantize
from msmodelslim.core.quantization_schema.qal import QABCRegistry, QScope, QScheme, QParam, QStorage, QDType
from msmodelslim.core.quantization_schema import AutoFakeQuantLinear
from msmodelslim.utils.logging import logger_setter
from msmodelslim.core.quantization_schema.utils import (
    pad_and_reshape,
    undo_pad_and_reshape,
    reshape_to_blocks,
    undo_reshape_to_blocks,
)
from msmodelslim.core.quantization_schema import hif4_per_block_sym, mxfp8_per_block_sym
from msmodelslim.core.observer import MsMinMaxBlockObserver, MinMaxBlockObserverConfig


@QABCRegistry.multi_register(dispatch_key=[(mxfp8_per_block_sym, hif4_per_block_sym)], abc_type=AutoFakeQuantLinear)
@logger_setter()
class W4A8HiFxMXDynamicPerBlockFakeQuantLinear(AutoFakeQuantLinear):
    def __init__(self, x_q_param: QParam, w_q_param: QParam, w_q: QStorage, bias: torch.Tensor):
        super().__init__()

        self.w_scheme = w_q_param.scheme
        self.w_mx_finfo = w_q_param.scheme.dtype.hifx_finfo
        self.w_axes = self.w_mx_finfo.axes

        self.x_scheme = x_q_param.scheme
        self.x_mx_finfo = x_q_param.scheme.dtype.mx_finfo
        self.x_axes = x_q_param.ext.get("axes")

        self.weight_scale_lv3 = nn.Parameter(w_q_param.ext.get("weight_scale_lv3"), requires_grad=False)
        self.weight_scale_lv2 = nn.Parameter(w_q_param.ext.get("weight_scale_lv2"), requires_grad=False)
        self.weight_scale_factor = nn.Parameter(w_q_param.ext.get("weight_scale_factor"), requires_grad=False)
        self.weight = nn.Parameter(w_q.value, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

        minmax_config = MinMaxBlockObserverConfig(axes=self.x_axes)
        self.x_minmax_block_observer = MsMinMaxBlockObserver(minmax_config)

    def __repr__(self) -> str:
        return f"W4A8HiFxMXDynamicPerBlockFakeQuantLinear(symmetric={self.w_scheme.symmetric})"

    @classmethod
    def from_hif4_export_state_dict(
        cls, state_dict: dict[str, torch.Tensor]
    ) -> "W4A8HiFxMXDynamicPerBlockFakeQuantLinear":
        """
        从 convert 导出态重建模块，仅供 AscendV1 保存阶段使用。
        不改变原量化链路 __init__ 入参语义。
        """
        inst = object.__new__(cls)
        nn.Module.__init__(inst)  # pylint: disable=unnecessary-dunder-call

        inst.w_scheme = None
        inst.w_mx_finfo = None
        inst.w_axes = None
        inst.x_scheme = None
        inst.x_mx_finfo = None
        inst.x_axes = None
        inst.x_minmax_block_observer = None

        inst.weight_scale_lv3 = nn.Parameter(state_dict["weight_scale_lv3"], requires_grad=False)
        inst.weight_scale_lv2 = nn.Parameter(state_dict["weight_scale_lv2"], requires_grad=False)
        inst.weight_scale_factor = nn.Parameter(state_dict["weight_scale_factor"], requires_grad=False)
        inst.weight = nn.Parameter(state_dict["weight"], requires_grad=False)
        bias = state_dict.get("bias")
        inst.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
        return inst

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        axes = self.x_axes
        axes = [axes] if isinstance(axes, int) else axes
        axes = [a + x.ndim if a < 0 else a for a in axes]
        x, axes_, orig_shape, padded_shape = reshape_to_blocks(x, axes, self.x_mx_finfo.block_size)
        shared_exp_axes = [x + 1 for x in axes_] if self.x_mx_finfo.block_size > 0 else axes_
        self.x_minmax_block_observer.update(x, shared_exp_axes=shared_exp_axes)
        x_min_val, x_max_val = self.x_minmax_block_observer.get_min_max()
        x_q_param = calculate_qparam(
            x_min_val,
            x_max_val,
            q_dtype=self.x_scheme.dtype,
            q_scope=self.x_scheme.scope,
            symmetric=self.x_scheme.symmetric,
        )
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x), x_q_param)
        x_q_dq.value = undo_reshape_to_blocks(x_q_dq.value, padded_shape, orig_shape)

        w_q_storage = QStorage(dtype=self.w_scheme.dtype, value=self.weight)
        w_orig_shape, w_orig_dtype = w_q_storage.value.shape, w_q_storage.value.dtype
        w_q_storage.value, axes, w_padc, w_dim0 = pad_and_reshape(
            w_q_storage.value.to(torch.float32), w_orig_shape, self.w_axes, self.w_mx_finfo
        )
        w_q_param = QParam(
            scheme=QScheme(scope=QScope.PER_BLOCK, dtype=QDType.HiF4, symmetric=True),
            ext={
                "weight_scale_lv3": self.weight_scale_lv3.data,
                "weight_scale_lv2": self.weight_scale_lv2.data,
                "weight_scale_factor": self.weight_scale_factor.data,
            },
        )
        weight_q_dq = dequantize(w_q_storage, w_q_param)
        weight_kwargs = {
            "padc": w_padc,
            "padded_shape": weight_q_dq.value.shape,
            "orig_shape": w_orig_shape,
            "dim0": w_dim0,
        }
        weight_q_dq.value = undo_pad_and_reshape(weight_q_dq.value.to(w_orig_dtype), self.w_axes, **weight_kwargs)

        return F.linear(x_q_dq.value, weight_q_dq.value, self.bias)  # pylint: disable=not-callable
