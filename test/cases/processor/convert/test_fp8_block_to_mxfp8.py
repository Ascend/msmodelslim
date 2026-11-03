#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""FP8_BLOCK -> W8A8_MXFP8 direct processor."""

import pytest
import torch

from msmodelslim.core.convert.config import ConvertConfig
from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.convert.types import IRKind, LossLevel, SourceIR, TensorRef
from msmodelslim.core.quant_service.modelslim_convert.virtual_module import ModelFreeLinear
from msmodelslim.ir.w8a8_mx_dynamic import W8A8MXDynamicPerBlockFakeQuantLinear
from msmodelslim.processor.convert.fp8_block_to_mxfp8 import Fp8BlockToMxFp8Processor

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "float8_e8m0fnu"),
    reason="torch build has no float8 e4m3/e8m0",
)


def test_transform_emits_lossless_mxfp8_module_with_source_e8m0_bytes():
    rows, cols = 32, 64
    weight_bits = torch.arange(rows * cols, dtype=torch.uint8).reshape(rows, cols) % 128
    weight = weight_bits.view(torch.float8_e4m3fn)
    scale_bits = torch.tensor([[120, 140]], dtype=torch.uint8)
    scale = scale_bits.view(torch.float8_e8m0fnu)
    mod = ModelFreeLinear(
        full_name="layers.0.attn.wq_a",
        tensor_bindings={
            "weight": TensorRef("weight", "layers.0.attn.wq_a.weight", "s0", "F8_E4M3", (rows, cols)),
            "weight_scale_inv": TensorRef(
                "weight_scale_inv",
                "layers.0.attn.wq_a.weight_scale_inv",
                "s0",
                "F8_E8M0",
                tuple(scale.shape),
                meta={"renamed_from": "layers.0.attn.wq_a.scale"},
            ),
        },
        source_ir=SourceIR(kind=IRKind.FP8_BLOCK),
    )
    mod.register_parameter("weight", torch.nn.Parameter(weight, requires_grad=False))
    mod.register_buffer("weight_scale_inv", scale)
    mod.lazy_initialized = True
    context = ConvertContext(config=ConvertConfig(model_path="/m", save_path="/o"))

    proc = Fp8BlockToMxFp8Processor()
    out = proc.transform(mod, context)

    assert proc.loss_level == LossLevel.LOSSLESS.value
    assert isinstance(out, W8A8MXDynamicPerBlockFakeQuantLinear)
    assert torch.equal(out.weight.view(torch.uint8).cpu(), weight_bits)
    saved = (out.weight_scale.squeeze(-1) + 127).to(torch.uint8)
    assert torch.equal(saved, scale_bits.expand(rows, cols // 32))
