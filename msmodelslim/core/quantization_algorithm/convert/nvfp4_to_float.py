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

IR edge: NVFP4_MODELOPT -> FLOAT (bf16 nn.Linear).
"""

from __future__ import annotations

from torch import nn

from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.convert.types import IRKind, LossLevel
from msmodelslim.core.quant_service.modelslim_convert.virtual_module import ModelFreeLinear
from msmodelslim.core.quantization_schema.kernels.nvfp4_modelopt import weight_dequant
from msmodelslim.core.quantization_algorithm.convert.base import BaseConvertProcessor


class Nvfp4ModelOptToFloatProcessor(BaseConvertProcessor):
    name = "Nvfp4ModelOptToFloatProcessor"
    src_ir = IRKind.NVFP4_MODELOPT
    dst_ir = IRKind.FLOAT
    loss_level = LossLevel.LOSSLESS.value

    def transform(self, module: nn.Module, context: ConvertContext) -> nn.Module:
        if not isinstance(module, ModelFreeLinear):
            return module
        if not module.lazy_initialized:
            module.lazy_init(context.reader, device="cpu")

        packed = getattr(module, "weight", None)
        scale = getattr(module, "weight_scale", None)
        scale2 = getattr(module, "weight_scale_2", None)
        if packed is None or scale is None or scale2 is None:
            return module

        weight_bf16 = weight_dequant(
            weight_packed=packed,
            weight_scale=scale,
            weight_scale_2=scale2,
            group_size=16,
        )
        bias = getattr(module, "bias", None)
        out = nn.Linear(weight_bf16.shape[1], weight_bf16.shape[0], bias=bias is not None)
        out.weight = nn.Parameter(weight_bf16, requires_grad=False)
        if bias is not None:
            out.bias = nn.Parameter(bias.detach().to(weight_bf16.dtype), requires_grad=False)
        return out
