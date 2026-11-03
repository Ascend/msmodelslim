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

IR edge: FLOAT -> W4A4_MXFP4.

Data-free per-block quantization through ``dtype_ops``. No calibration tensors
are read. ``mxfp4_calc_qparam_sym`` returns uint8 E8M0; the deployed module
stores the unbiased shared exponent, which export shifts back by +127.
"""

from __future__ import annotations

import torch
from torch import nn

from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.convert.types import IRKind, LossLevel
from msmodelslim.core.dtype_ops import mxfp4_calc_qparam_sym, mxfp4_quantize_sym
from msmodelslim.ir.w4a4_mx_dynamic import W4A4MXDynamicPerBlockFakeQuantLinear
from msmodelslim.processor.convert.base import BaseConvertProcessor
from msmodelslim.processor.convert.mxfp8_quant import _materialize_linear

_E8M0_BIAS = 127


class MxFp4QuantProcessor(BaseConvertProcessor):
    name = "MxFp4QuantProcessor"
    src_ir = IRKind.FLOAT
    dst_ir = IRKind.W4A4_MXFP4
    loss_level = LossLevel.LOSSY.value

    def transform(self, module: nn.Module, context: ConvertContext) -> nn.Module:
        linear = _materialize_linear(module, context)
        if linear is None:
            return module

        weight = linear.weight.detach()
        shared_exp = mxfp4_calc_qparam_sym(weight, axis=-1)
        q, shared_exp = mxfp4_quantize_sym(weight, shared_exp, axis=-1)
        exponent = shared_exp.to(torch.float32) - _E8M0_BIAS
        state = {
            "weight": q.detach(),
            "weight_scale": exponent,
            "weight_offset": torch.zeros_like(exponent),
        }
        if linear.bias is not None:
            state["bias"] = linear.bias.detach()
        return W4A4MXDynamicPerBlockFakeQuantLinear.from_deploy_state_dict(state)
