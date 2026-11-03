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

IR edge: FP8_BLOCK -> W8A8_MXFP8, lossless when the source block already
matches MX groups.
"""

from __future__ import annotations

import torch
from torch import nn

from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.convert.types import IRKind, LossLevel
from msmodelslim.core.quant_service.modelslim_convert.virtual_module import ModelFreeLinear
from msmodelslim.ir.kernels.fp8_block import fp8_block_to_mx
from msmodelslim.ir.w8a8_mx_dynamic import W8A8MXDynamicPerBlockFakeQuantLinear
from msmodelslim.processor.convert.base import BaseConvertProcessor
from msmodelslim.utils.exception import SchemaValidateError


class Fp8BlockToMxFp8Processor(BaseConvertProcessor):
    name = "Fp8BlockToMxFp8Processor"
    src_ir = IRKind.FP8_BLOCK
    dst_ir = IRKind.W8A8_MXFP8
    loss_level = LossLevel.LOSSLESS.value

    def transform(self, module: nn.Module, context: ConvertContext) -> nn.Module:
        if not isinstance(module, ModelFreeLinear):
            return module
        if not module.lazy_initialized:
            module.lazy_init(context.reader, device=context.resolved_worker_device)

        weight = getattr(module, "weight", None)
        scale = module._buffers.get("weight_scale_inv")
        if weight is None or scale is None:
            raise SchemaValidateError(
                f"{module.full_name}: direct MXFP8 conversion requires E4M3 weight and E8M0 weight_scale_inv",
            )
        mx_weight, exponent = fp8_block_to_mx(weight, scale)
        state = {
            "weight": mx_weight,
            "weight_scale": exponent,
            "weight_offset": torch.zeros_like(exponent),
        }
        bias = getattr(module, "bias", None)
        if bias is not None:
            state["bias"] = bias.detach()
        return W8A8MXDynamicPerBlockFakeQuantLinear.from_deploy_state_dict(state)
