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

IR edge: MXFP4_PACKED -> FLOAT (bf16 ``nn.Linear``).

Reads ``weight_packed`` + ``weight_scale``. Checkpoint ``.weight`` / ``.scale``
must be renamed to those names before this edge runs. Nibbles are unpacked
first; the block multiply is ``mxfp4_dequantize_sym``.
"""

from __future__ import annotations

import torch
from torch import nn

from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.convert.types import IRKind, LossLevel
from msmodelslim.core.quant_service.modelslim_convert.virtual_module import ModelFreeLinear
from msmodelslim.ir.kernels.mxfp4 import prepare_mxfp4_dequant
from msmodelslim.core.dtype_ops import mxfp4_dequantize_sym
from msmodelslim.processor.convert.base import BaseConvertProcessor
from msmodelslim.utils.exception import SchemaValidateError


class MxFp4PackedToFloatProcessor(BaseConvertProcessor):
    name = "MxFp4PackedToFloatProcessor"
    src_ir = IRKind.MXFP4_PACKED
    dst_ir = IRKind.FLOAT
    loss_level = LossLevel.LOSSLESS.value

    def transform(self, module: nn.Module, context: ConvertContext) -> nn.Module:
        if not isinstance(module, ModelFreeLinear):
            return module
        if not module.lazy_initialized:
            module.lazy_init(context.reader, device=context.resolved_worker_device)

        packed, scale = _packed_and_scale(module)
        values, shared_exp = prepare_mxfp4_dequant(packed, scale)
        restored = mxfp4_dequantize_sym(values, shared_exp, axis=-1)
        weight_bf16 = restored.to(torch.bfloat16)

        bias = getattr(module, "bias", None)
        out = nn.Linear(weight_bf16.shape[1], weight_bf16.shape[0], bias=bias is not None)
        out.weight = nn.Parameter(weight_bf16, requires_grad=False)
        if bias is not None:
            out.bias = nn.Parameter(bias.detach().to(torch.bfloat16), requires_grad=False)
        return out


def _packed_and_scale(module: ModelFreeLinear) -> tuple[torch.Tensor, torch.Tensor]:
    packed = module._buffers.get("weight_packed")
    scale = module._buffers.get("weight_scale")
    if packed is None or scale is None:
        raise SchemaValidateError(
            f"{module.full_name}: MXFP4 layer is missing weight_packed and weight_scale",
            action="Rename checkpoint .weight/.scale to weight_packed/weight_scale before convert.",
        )
    return packed, scale
