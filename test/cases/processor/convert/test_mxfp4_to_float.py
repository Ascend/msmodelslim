#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""MXFP4_PACKED -> FLOAT. weight_packed + weight_scale matches a decode_tensor-style reference."""

import pytest
import torch
from torch import nn

from msmodelslim.core.convert.config import ConvertConfig
from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.convert.types import IRKind, LossLevel, SourceIR, TensorRef
from msmodelslim.core.quant_service.modelslim_convert.virtual_module import ModelFreeLinear
from msmodelslim.processor.convert.mxfp4_to_float import MxFp4PackedToFloatProcessor
from msmodelslim.utils.exception import SchemaValidateError

_LUT = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])


def _reference_bf16(packed: torch.Tensor, scale_bits: torch.Tensor) -> torch.Tensor:
    """Same nibble order and E8M0 decode as DeepSeek ``decode_tensor``, then bf16."""
    data = packed.reshape(packed.shape[0], -1).contiguous().view(torch.uint8)
    low = _LUT[(data & 0x0F).long()].to(torch.bfloat16)
    high = _LUT[(data >> 4).long()].to(torch.bfloat16)
    values = torch.stack((low, high), dim=-1).reshape(data.shape[0], -1)
    scale = torch.exp2((scale_bits.to(torch.int32) - 127).to(torch.float32))
    restored = (values.reshape(*scale.shape, 32) * scale.unsqueeze(-1)).reshape_as(values)
    return restored.to(torch.bfloat16)


def _module(bindings, **tensors) -> ModelFreeLinear:
    mod = ModelFreeLinear(
        full_name="layers.0.ffn.experts.0.w1",
        tensor_bindings=bindings,
        source_ir=SourceIR(kind=IRKind.MXFP4_PACKED),
    )
    for name, tensor in tensors.items():
        if name in ("weight", "bias"):
            mod.register_parameter(name, nn.Parameter(tensor, requires_grad=False))
        else:
            mod.register_buffer(name, tensor)
    mod.lazy_initialized = True
    return mod


def test_packed_layout_matches_reference_including_bias_and_negative_nibbles():
    packed = torch.tensor(
        [[0x10, 0x2F, 0x83, 0x06] * 4, [0xF0, 0x01, 0x6A, 0xC4] * 4],
        dtype=torch.uint8,
    )
    scale_bits = torch.tensor([[120], [140]], dtype=torch.uint8)
    bias = torch.tensor([1.5, -2.0], dtype=torch.float32)
    expected = _reference_bf16(packed, scale_bits)
    context = ConvertContext(config=ConvertConfig(model_path="/m", save_path="/o"))
    proc = MxFp4PackedToFloatProcessor()

    compressed = _module(
        {
            "weight_packed": TensorRef("weight_packed", "w", "s0", "U8", tuple(packed.shape)),
            "weight_scale": TensorRef("weight_scale", "s", "s0", "U8", tuple(scale_bits.shape)),
            "bias": TensorRef("bias", "b", "s0", "BF16", tuple(bias.shape)),
        },
        weight_packed=packed,
        weight_scale=scale_bits,
        bias=bias,
    )

    out = proc.transform(compressed, context)
    assert proc.loss_level == LossLevel.LOSSLESS.value
    assert torch.equal(out.weight, expected)
    assert torch.equal(out.bias, bias.to(torch.bfloat16))


def test_transform_rejects_scale_shape_before_dequant():
    packed = torch.zeros(2, 16, dtype=torch.uint8)
    scale = torch.zeros(2, 2, dtype=torch.uint8)
    mod = _module(
        {
            "weight_packed": TensorRef("weight_packed", "w", "s0", "U8", tuple(packed.shape)),
            "weight_scale": TensorRef("weight_scale", "s", "s0", "U8", tuple(scale.shape)),
        },
        weight_packed=packed,
        weight_scale=scale,
    )
    context = ConvertContext(config=ConvertConfig(model_path="/m", save_path="/o"))
    with pytest.raises(SchemaValidateError, match="scale shape"):
        MxFp4PackedToFloatProcessor().transform(mod, context)
