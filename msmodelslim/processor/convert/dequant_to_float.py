#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
IR edge: FP8_BLOCK -> FLOAT (bf16 ``nn.Linear``).

Reuses block dequant kernel from ``ir.kernels.fp8_block``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from msmodelslim.core.quant_service.modelslim_convert.virtual_module import ModelFreeLinear
from msmodelslim.ir.kernels import WEIGHT_SCALE_INV_SUFFIX, weight_dequant
from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.convert.types import IRKind, LossLevel
from msmodelslim.processor.convert.base import BaseConvertProcessor


class DequantToFloatProcessor(BaseConvertProcessor):
    name = "DequantToFloatProcessor"
    src_ir = IRKind.FP8_BLOCK
    dst_ir = IRKind.FLOAT
    loss_level = LossLevel.LOSSLESS.value

    def transform(self, module: nn.Module, context: ConvertContext) -> nn.Module:
        if not isinstance(module, ModelFreeLinear):
            return module
        weight = getattr(module, "weight", None)
        if weight is None:
            return module

        scale = module._buffers.get("weight_scale_inv")
        if scale is None:
            for logical, ref in module.tensor_bindings.items():
                if logical in ("weight_scale_inv", "weight_scale") and ref.key.endswith(WEIGHT_SCALE_INV_SUFFIX):
                    if not module.lazy_initialized:
                        module.lazy_init(context.reader, device=context.resolved_worker_device)
                    scale = module._buffers.get(logical)
                    break

        block_size = _fp8_block_size(context)
        if scale is not None:
            weight_bf16 = weight_dequant(weight, scale, block_size=block_size)
        else:
            if str(weight.device).startswith("npu") and weight.dtype == torch.float8_e4m3fn:
                dev = weight.device
                weight_bf16 = weight.to("cpu").to(torch.bfloat16).to(dev)
            else:
                weight_bf16 = weight.to(torch.bfloat16)

        bias = getattr(module, "bias", None)
        out = nn.Linear(weight_bf16.shape[1], weight_bf16.shape[0], bias=bias is not None)
        out.weight = nn.Parameter(weight_bf16, requires_grad=False)
        if bias is not None:
            out.bias = nn.Parameter(bias.to(torch.bfloat16), requires_grad=False)
        return out


def _fp8_block_size(context: ConvertContext) -> int:
    cfg_path = Path(context.config.model_path) / "config.json"
    if cfg_path.is_file():
        qc = json.loads(cfg_path.read_text(encoding="utf-8")).get("quantization_config") or {}
        bs = qc.get("weight_block_size")
        if bs:
            return int(bs[0])
    return 128
