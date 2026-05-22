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
from torch import nn as nn
from torch.nn import functional as F

from msmodelslim.ir.api import calculate_qparam, fake_quantize, dequantize
from msmodelslim.ir.qal import QABCRegistry, QScope, QScheme,  QParam, QStorage, QDType
from msmodelslim.ir import AutoFakeQuantLinear
from msmodelslim.utils.logging import logger_setter

from msmodelslim.ir import mxfp4_per_block_sym, mxfp4_dual_scale_sym
from msmodelslim.core.observer import MsMinMaxBlockObserver, MinMaxBlockObserverConfig

from msmodelslim.ir.utils import reshape_to_blocks, undo_reshape_to_blocks
from msmodelslim.core.quantizer.impl.dualscale import MXFP4_MAX_NORMAL


"""
========================================================================================
二级块量化 (Dual-Scale Block Quantization) 计算公式与流程说明：
========================================================================================

本算法实现了基于 MXFP4 (Microscaling Formats) 的 W4A4 两级缩放动态量化线性层。
其核心思想是通过“外层大块 (Dual Block) 配合高精度 scale”与“内层小块 (Inner Block) 
配合 E8M0 类型 scale”两级缩放因子共同控制精度，以降低低比特（4-bit）量化带来的精度损失。

计算逻辑与核心公式如下：

1. 激活值 (Activation) 的动态量化与反量化 (Fake Quantization):
   ---------------------------------------------------------------------------------
   a) 外层缩放 (Dual Scale):
      将输入 X 按照 x_dual_block_size 划分，计算每个大块的最大绝对值，得到外层尺度 S_dual_x:
      
      S_dual_x = max(abs(X_block)) / MXFP4_MAX_NORMAL
      X_dualscaled = X / S_dual_x
      
   b) 内层量化与反量化 (Inner Quant-Dequant):
      将 X_dualscaled 进一步按 x_inner_block_size 划分，计算内层尺度并转换为目标低比特格式：
      
      X_q_dq_inner = fake_quantize(X_dualscaled, S_inner_x)
      
   c) 外层恢复 (Dual Scale Dequantization):
      X_q_dq = X_q_dq_inner * S_dual_x

2. 权重 (Weight) 的静态反量化 (Dequantization):
   ---------------------------------------------------------------------------------
   权重在初始化时已完成了量化存储，前向传播时仅进行两级反量化恢复至高精度：
   
   a) 内层反量化 (Inner Dequantization):
      根据内层参数 inner_w_q_param 恢复基础缩放：
      
      W_dualscaled_q_dq = dequantize(W_quantized, S_inner_w)
      
   b) 外层反量化 (Dual Scale Dequantization):
      乘以模型初始化时固化的外层权重尺度 weight_dual_scale (S_dual_w):
      
      W_q_dq = W_dualscaled_q_dq * S_dual_w

3. 矩阵乘法计算 (Linear Matrix Multiply):
   ---------------------------------------------------------------------------------
   Output = X_q_dq @ (W_q_dq.T) + bias
   (注: @ 表示矩阵乘法，.T 表示转置)
========================================================================================
"""

@QABCRegistry.multi_register(
    dispatch_key=[
        (mxfp4_dual_scale_sym, mxfp4_dual_scale_sym)
    ],
    abc_type=AutoFakeQuantLinear
)
@logger_setter()
class W4A4MXDynamicDualScaleFakeQuantLinear(AutoFakeQuantLinear):
    def __init__(
            self,
            x_q_param: QParam,
            w_q_param: QParam,
            w_q: QStorage,
            bias: torch.Tensor
    ):
        super().__init__()
        self.w_scheme = w_q_param.scheme
        self.w_inner_block_size = w_q_param.scheme.dtype.mx_finfo.block_size
        self.w_axes = w_q_param.ext.get("axes")
        self.w_dual_block_size = w_q_param.ext.get("dual_block_size")

        self.x_scheme = x_q_param.scheme
        self.x_inner_block_size = x_q_param.scheme.dtype.mx_finfo.block_size
        self.x_axes = x_q_param.ext.get("axes")
        self.x_dual_block_size = x_q_param.ext.get("dual_block_size")

        self.inner_w_q_param = QParam(
            scheme=QScheme(
                dtype=self.w_scheme.dtype,
                scope=QScope.PER_BLOCK,
                symmetric=self.w_scheme.symmetric,
            ),
            ext={
                "scale": w_q_param.ext.get("scale"),
                "offset": w_q_param.ext.get("offset"),
                "keep_mask": w_q_param.ext.get("keep_mask"),
                "axes": w_q_param.ext.get("axes")
            }
        )

        minmax_config = MinMaxBlockObserverConfig(axes=self.x_axes)
        self.x_minmax_block_observer = MsMinMaxBlockObserver(minmax_config)
        self.weight_scale = nn.Parameter(w_q_param.ext.get("scale"), requires_grad=False)
        self.weight_dual_scale = nn.Parameter(w_q_param.ext.get("dual_scale"), requires_grad=False)
        self.weight_offset = nn.Parameter(w_q_param.ext.get("offset"), requires_grad=False)
        self.weight = nn.Parameter(w_q.value, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

    def __repr__(self) -> str:
        return f"W4A4MXDynamicDualScaleFakeQuantLinear(symmetric={self.w_scheme.symmetric})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_target_axes = self.w_axes
        w_target_axes = [w_target_axes] if isinstance(w_target_axes, int) else w_target_axes
        self.w_target_axes = [a + self.weight.ndim if a < 0 else a for a in w_target_axes]

        x_target_axes = self.x_axes
        x_target_axes = [x_target_axes] if isinstance(x_target_axes, int) else x_target_axes
        self.x_target_axes = [a + x.ndim if a < 0 else a for a in x_target_axes]

        # act quant - dual-scale
        x, axes_, orig_shape, padded_shape = reshape_to_blocks(x, self.x_target_axes, self.x_dual_block_size)
        dual_scale_axes = [a + 1 for a in axes_] if self.x_dual_block_size > 0 else axes_
        self.x_minmax_block_observer.update(x, shared_exp_axes=dual_scale_axes)
        dual_block_min_val, dual_block_max_val = self.x_minmax_block_observer.get_min_max()
        dual_scale_x = dual_block_max_val / MXFP4_MAX_NORMAL
        x_dualscaled = x.to(torch.float32) / dual_scale_x
        x_dualscaled = undo_reshape_to_blocks(x_dualscaled, padded_shape, orig_shape, self.x_target_axes)

        # act inner quant-dequant - mxfp
        x_dualscaled, axes_, orig_shape, padded_shape = reshape_to_blocks(x_dualscaled, self.x_target_axes, self.x_inner_block_size)
        shared_exp_axes = [a + 1 for a in axes_] if self.x_inner_block_size > 0 else axes_
        self.x_minmax_block_observer.update(x_dualscaled, shared_exp_axes=shared_exp_axes)
        inner_x_min_val, inner_x_max_val = self.x_minmax_block_observer.get_min_max()
        x_dualscaled_q_param = calculate_qparam(
            inner_x_min_val, inner_x_max_val,
            q_dtype=self.x_scheme.dtype,
            q_scope=QScope.PER_BLOCK,
            symmetric=self.x_scheme.symmetric
        )
        x_dualscaled_q_dq = fake_quantize(QStorage(QDType.FLOAT, x_dualscaled), x_dualscaled_q_param).value
        x_dualscaled_q_dq = undo_reshape_to_blocks(x_dualscaled_q_dq, padded_shape, orig_shape, self.x_target_axes)

        # act de_quant - dual-scale
        x_dualscaled_q_dq, axes_, orig_shape, padded_shape = reshape_to_blocks(x_dualscaled_q_dq, self.x_target_axes, self.x_dual_block_size)
        x_q_dq = x_dualscaled_q_dq.to(torch.float32) * dual_scale_x
        x_q_dq = undo_reshape_to_blocks(x_q_dq, padded_shape, orig_shape, self.x_target_axes)

        # wei inner de_quant - mxfp
        weight_dualscaled = self.weight
        weight_dualscaled, axes_, orig_shape, padded_shape = reshape_to_blocks(weight_dualscaled, self.w_target_axes, self.w_inner_block_size)
        weight_dualscaled_q_dq = dequantize(QStorage(self.w_scheme.dtype, weight_dualscaled), self.inner_w_q_param).value
        weight_dualscaled_q_dq = undo_reshape_to_blocks(weight_dualscaled_q_dq, padded_shape, orig_shape, self.w_target_axes)

        # wei de_quant - dual_scale
        weight_dualscaled_q_dq, axes_, orig_shape, padded_shape = reshape_to_blocks(weight_dualscaled_q_dq, self.w_target_axes, self.w_dual_block_size)
        weight_q_dq = weight_dualscaled_q_dq.to(torch.float32) * self.weight_dual_scale
        weight_q_dq = undo_reshape_to_blocks(weight_q_dq, padded_shape, orig_shape, self.w_target_axes)

        return F.linear(x_q_dq, weight_q_dq, self.bias)
