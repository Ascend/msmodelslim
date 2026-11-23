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
from msmodelslim.core.quantization_schema.utils import pad_and_reshape, undo_pad_and_reshape
from msmodelslim.core.quantization_schema import hif4_per_block_sym
from msmodelslim.core.observer import MsMinMaxBlockObserver, MinMaxBlockObserverConfig


@QABCRegistry.multi_register(dispatch_key=[(hif4_per_block_sym, hif4_per_block_sym)], abc_type=AutoFakeQuantLinear)
@logger_setter()
class W4A4Hif4DynamicPerBlockFakeQuantLinear(AutoFakeQuantLinear):
    def __init__(self, x_q_param: QParam, w_q_param: QParam, w_q: QStorage, bias: torch.Tensor):
        super().__init__()

        self.w_scheme = w_q_param.scheme
        self.w_mx_finfo = w_q_param.scheme.dtype.hifx_finfo
        self.w_axes = self.w_mx_finfo.axes

        self.x_scheme = x_q_param.scheme
        self.x_mx_finfo = x_q_param.scheme.dtype.hifx_finfo
        self.x_axes = self.x_mx_finfo.axes
        self.x_scale_ratio = x_q_param.ext.get("scale_ratio", 1)
        self.x_clip2 = x_q_param.ext.get("clip2", 4)
        self.x_clip3 = x_q_param.ext.get("clip3", 2)

        self.weight_scale_lv3 = nn.Parameter(w_q_param.ext.get("weight_scale_lv3"), requires_grad=False)
        self.weight_scale_lv2 = nn.Parameter(w_q_param.ext.get("weight_scale_lv2"), requires_grad=False)
        self.weight_scale_factor = nn.Parameter(w_q_param.ext.get("weight_scale_factor"), requires_grad=False)
        self.weight = nn.Parameter(w_q.value, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

        minmax_config = MinMaxBlockObserverConfig(axes=self.x_axes)
        self.x_minmax_block_observer = MsMinMaxBlockObserver(minmax_config)

    def __repr__(self) -> str:
        return f"W4A4Hif4DynamicPerBlockFakeQuantLinear(symmetric={self.w_scheme.symmetric})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        axes = self.x_axes
        ori_shape, ori_dtype = x.shape, x.dtype
        x, axes, padc, dim0 = pad_and_reshape(x.to(torch.float32), ori_shape, axes, self.x_mx_finfo)
        x_level = torch.abs(x)
        max_levels = {}
        quant_level = 3
        for ax in range(axes, axes - 3, -1):
            self.x_minmax_block_observer.update(x_level, shared_exp_axes=[ax])
            _, max_lv = self.x_minmax_block_observer.get_min_max()
            max_levels[quant_level] = max_lv
            quant_level -= 1
            x_level = max_lv
        x_q_param = calculate_qparam(
            min_val=None,
            max_val=None,
            q_dtype=self.x_scheme.dtype,
            q_scope=self.x_scheme.scope,
            symmetric=self.x_scheme.symmetric,
            max_levels=max_levels,
            axes=axes,
            scale_ratio=self.x_scale_ratio,
            clip2=self.x_clip2,
            clip3=self.x_clip3,
        )
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x), x_q_param)
        kwargs = {"padc": padc, "padded_shape": x.shape, "orig_shape": ori_shape, "dim0": dim0}
        x_q_dq.value = undo_pad_and_reshape(x_q_dq.value.to(ori_dtype), axes, **kwargs)

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

        return F.linear(x_q_dq.value, weight_q_dq.value, self.bias)
