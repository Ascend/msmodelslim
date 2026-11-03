#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""FLOAT -> W4A4_MXFP4. A second quantize of the dequantized weight stays on the grid."""

import torch
from torch import nn

import msmodelslim.core.quant_service.modelslim_convert.virtual_module  # noqa: F401
from msmodelslim.core.convert.config import ConvertConfig
from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.convert.types import LossLevel
from msmodelslim.ir.api import dequantize
from msmodelslim.ir.qal import QParam, QScheme, QScope, QStorage, QDType
from msmodelslim.core.dtype_ops import mxfp4_dequantize_sym
from msmodelslim.ir.utils import reshape_to_blocks, undo_reshape_to_blocks
from msmodelslim.ir.w4a4_mx_dynamic import W4A4MXDynamicPerBlockFakeQuantLinear
from msmodelslim.processor.convert.mxfp4_quant import MxFp4QuantProcessor


def _dequant_weight(mod: W4A4MXDynamicPerBlockFakeQuantLinear) -> torch.Tensor:
    axes = mod.w_axes
    axes = [axes] if isinstance(axes, int) else list(axes)
    axes = [axis + mod.weight.ndim if axis < 0 else axis for axis in axes]
    reshaped, _, orig_shape, padded_shape = reshape_to_blocks(
        mod.weight.detach(),
        axes,
        mod.w_mx_finfo.block_size,
    )
    restored = dequantize(
        QStorage(QDType.MXFP4, reshaped),
        QParam(
            scheme=QScheme(scope=QScope.PER_BLOCK, dtype=QDType.MXFP4, symmetric=True),
            ext={"scale": mod.weight_scale.detach()},
        ),
    )
    return undo_reshape_to_blocks(restored.value, padded_shape, orig_shape, axes)


def test_transform_quantizes_bf16_linear_and_requant_stays_close():
    torch.manual_seed(0)
    linear = nn.Linear(64, 8, bias=True)
    linear.weight = nn.Parameter(torch.randn(8, 64), requires_grad=False)
    linear.bias = nn.Parameter(torch.randn(8), requires_grad=False)
    context = ConvertContext(config=ConvertConfig(model_path="/m", save_path="/o", dst_format="ascendv1"))
    proc = MxFp4QuantProcessor()

    first = proc.transform(linear, context)
    assert proc.loss_level == LossLevel.LOSSY.value
    assert isinstance(first, W4A4MXDynamicPerBlockFakeQuantLinear)
    restored = _dequant_weight(first)
    assert torch.isfinite(restored).all()
    e8m0 = (first.weight_scale.detach() + 127).to(torch.uint8)
    via_dtype_ops = mxfp4_dequantize_sym(first.weight.detach(), e8m0, axis=-1)
    assert torch.equal(restored, via_dtype_ops)

    again = nn.Linear(restored.shape[1], restored.shape[0], bias=False)
    again.weight = nn.Parameter(restored.detach().to(torch.bfloat16), requires_grad=False)
    second = proc.transform(again, context)
    restored_again = _dequant_weight(second)
    assert torch.allclose(restored_again, restored.to(restored_again.dtype), rtol=1e-2, atol=1e-2)
