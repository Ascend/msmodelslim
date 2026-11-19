#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Integer convert processors, and the AscendV1 tensors they produce."""

import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import msmodelslim.core.quant_service.modelslim_convert.virtual_module  # noqa: F401
from msmodelslim.core.convert.config import ConvertConfig
from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.convert.types import IRKind, SourceIR, TensorRef
from msmodelslim.core.dtype_ops import (
    int4_quantize_dynamic_per_channel_sym,
    int8_quantize_dynamic_per_channel_asym,
    int8_quantize_dynamic_per_channel_sym,
)
from msmodelslim.core.quant_service.modelslim_convert.virtual_module import ModelFreeLinear
from msmodelslim.core.quant_service.modelslim_v1.save.ascendv1 import AscendV1Config, AscendV1Saver
from msmodelslim.core.quantization_schema.api import calculate_qparam, dequantize, fake_quantize
from msmodelslim.core.quantization_schema.const import int4_per_channel_sym
from msmodelslim.core.quantization_schema.qal import QDType, QParam, QScope, QStorage
from msmodelslim.core.quantization_schema.w4a4_dynamic import W4A4DynamicPerChannelFakeQuantLinear
from msmodelslim.core.quantization_schema.w4a8_dynamic import W4A8DynamicFakeQuantLinear
from msmodelslim.core.quantization_schema.w8a16_static import W8A16StaticPerChannelFakeQuantLinear
from msmodelslim.core.quantization_schema.w8a16_neg_offset import W8A16PerChannelNegOffsetFakeQuantLinear
from msmodelslim.core.quantization_schema.w8a8_dynamic import W8A8DynamicPerChannelFakeQuantLinear
from msmodelslim.core.quantization_algorithm.convert.int_quant import (
    INT_QUANT_PROCESSORS,
    quantize_w4a4,
    quantize_w8a16_asym,
    quantize_w8a16_sym,
    quantize_w8a8,
)

_OUT = 8
_IN = 16


def _context() -> ConvertContext:
    return ConvertContext(config=ConvertConfig(model_path="/m", save_path="/o", dst_format="ascendv1"))


def _processor(kind: IRKind):
    return next(proc for proc in INT_QUANT_PROCESSORS if proc.dst_ir == kind)


def _linear(bias: bool) -> nn.Linear:
    torch.manual_seed(0)
    linear = nn.Linear(_IN, _OUT, bias=bias)
    linear.weight = nn.Parameter(torch.randn(_OUT, _IN), requires_grad=False)
    if bias:
        linear.bias = nn.Parameter(torch.randn(_OUT), requires_grad=False)
    return linear


class _Adapter:
    def __init__(self, model_path: str):
        self.model_path = model_path


def _capture_saver():
    tmp = tempfile.mkdtemp()
    with patch("msmodelslim.core.quant_service.modelslim_v1.save.ascendv1.dist.is_initialized", return_value=False):
        saver = AscendV1Saver(nn.Linear(_IN, _OUT), AscendV1Config(save_directory=tmp, part_file_size=0), _Adapter(tmp))
    saver.json_writer = MagicMock()
    saver.safetensors_writer = MagicMock()
    saved = {}

    def _write(prefix, desc, tensor):
        saved[prefix] = (desc, tensor.detach().cpu().clone())

    saver.write_tensor = _write
    return saver, saved


class TestIntQuantProcessor:
    @pytest.mark.parametrize("bias", [False, True])
    def test_w8a16_sym_offset_is_zero_and_matches_dtype_ops(self, bias):
        linear = _linear(bias)
        mod = quantize_w8a16_sym(linear)
        q, scale = int8_quantize_dynamic_per_channel_sym(linear.weight.detach(), axis=0, dtype=torch.int8)

        assert isinstance(mod, W8A16StaticPerChannelFakeQuantLinear)
        assert mod.weight.shape == (_OUT, _IN)
        assert torch.equal(mod.weight, q)
        assert torch.equal(mod.weight_scale, scale.squeeze(-1))
        assert torch.equal(mod.weight_offset, torch.zeros_like(mod.weight_scale))
        assert (mod.bias is None) == (not bias)

    @pytest.mark.parametrize("bias", [False, True])
    def test_w8a16_asym_offset_matches_dtype_ops(self, bias):
        linear = _linear(bias)
        # Strictly positive weights make the zero-point non-zero, so this is not the symmetric path.
        linear.weight = nn.Parameter(torch.rand(_OUT, _IN) + 0.1, requires_grad=False)
        mod = quantize_w8a16_asym(linear)
        q, scale, offset = int8_quantize_dynamic_per_channel_asym(linear.weight.detach(), axis=0, dtype=torch.int8)

        zp = offset.squeeze(-1)

        assert isinstance(mod, W8A16PerChannelNegOffsetFakeQuantLinear)
        assert torch.equal(mod.weight, q)
        assert torch.equal(mod.weight_scale, scale.squeeze(-1))
        # vllm dequant is (q + offset) * scale, so the checkpoint stores -zp.
        assert torch.equal(mod.weight_offset, -zp)
        assert torch.any(mod.weight_offset != 0)

        saver, saved = _capture_saver()
        prefix = "model.layers.0.linear"
        saver.process_map[type(mod)](prefix, mod)
        _, exported = saved[prefix + ".weight_offset"]
        assert torch.equal(exported.view(-1), -zp)

    def test_w8a8_has_no_offset_and_keeps_int8_weight(self):
        linear = _linear(False)
        mod = quantize_w8a8(linear)
        q, scale = int8_quantize_dynamic_per_channel_sym(linear.weight.detach(), axis=0, dtype=torch.int8)

        assert isinstance(mod, W8A8DynamicPerChannelFakeQuantLinear)
        assert not hasattr(mod, "weight_offset")
        assert torch.equal(mod.weight, q)
        assert torch.equal(mod.weight_scale, scale.squeeze(-1))
        assert mod.weight.shape == (_OUT, _IN)

    @pytest.mark.parametrize(
        ("kind", "cls"),
        [
            (IRKind.W4A4_INT4, W4A4DynamicPerChannelFakeQuantLinear),
        ],
    )
    def test_int4_codes_stay_in_range_and_keep_shape(self, kind, cls):
        linear = _linear(False)
        mod = _processor(kind).transform(linear, _context())
        q, scale = int4_quantize_dynamic_per_channel_sym(linear.weight.detach(), axis=0, dtype=torch.int8)

        assert isinstance(mod, cls)
        assert mod.weight.dtype == torch.int8
        assert mod.weight.shape == (_OUT, _IN)
        assert int(mod.weight.min()) >= -8
        assert int(mod.weight.max()) <= 7
        assert torch.equal(mod.weight, q)
        assert torch.equal(mod.weight_scale, scale.squeeze(-1))

    def test_w4a4_offset_is_zero(self):
        mod = _processor(IRKind.W4A4_INT4).transform(_linear(True), _context())
        assert isinstance(mod, W4A4DynamicPerChannelFakeQuantLinear)
        assert torch.equal(mod.weight_offset, torch.zeros_like(mod.weight_scale))

    def test_non_2d_weight_is_returned_unchanged(self):
        module = ModelFreeLinear(full_name="layers.0.norm", source_ir=SourceIR(kind=IRKind.FLOAT))
        module.lazy_initialized = True
        module.weight = nn.Parameter(torch.randn(4), requires_grad=False)
        out = _processor(IRKind.W4A4_INT4).transform(module, _context())
        assert out is module


class TestIntQuantAscendV1Export:
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.parametrize(
        ("kind", "desc"),
        [
            (IRKind.W8A16_INT8, "W8A16"),
            (IRKind.W8A16_INT8_ASYM, "W8A16"),
            (IRKind.W8A8_INT8, "W8A8_DYNAMIC"),
            (IRKind.W4A4_INT4, "W4A4_DYNAMIC"),
        ],
    )
    def test_unpacked_targets_keep_nk_weight(self, kind, desc, bias):
        mod = {
            IRKind.W8A16_INT8: quantize_w8a16_sym,
            IRKind.W8A16_INT8_ASYM: quantize_w8a16_asym,
            IRKind.W8A8_INT8: quantize_w8a8,
            IRKind.W4A4_INT4: quantize_w4a4,
        }[kind](_linear(bias))
        saver, saved = _capture_saver()
        prefix = "model.layers.0.linear"
        saver.process_map[type(mod)](prefix, mod)

        weight_desc, weight = saved[prefix + ".weight"]
        scale_desc, scale = saved[prefix + ".weight_scale"]
        assert weight_desc == desc
        assert scale_desc == desc
        assert weight.shape == (_OUT, _IN)
        assert weight.dtype == torch.int8
        assert scale.shape == (_OUT, 1)
        assert (prefix + ".bias" in saved) == bias
        if kind in (IRKind.W8A16_INT8, IRKind.W4A4_INT4):
            _, offset = saved[prefix + ".weight_offset"]
            assert torch.equal(offset, torch.zeros_like(offset))


def _fp8_module(weight: torch.Tensor, bias: torch.Tensor | None = None) -> ModelFreeLinear:
    bindings = {"weight": TensorRef("weight", "w", "s0", "bf16", tuple(weight.shape))}
    tensors = {"weight": weight}
    if bias is not None:
        bindings["bias"] = TensorRef("bias", "b", "s0", "bf16", tuple(bias.shape))
        tensors["bias"] = bias
    mod = ModelFreeLinear(
        full_name="layers.0.q_proj",
        tensor_bindings=bindings,
        source_ir=SourceIR(kind=IRKind.FP8_BLOCK),
    )
    for name, tensor in tensors.items():
        mod.register_parameter(name, nn.Parameter(tensor, requires_grad=False))
    mod.lazy_initialized = True
    return mod


def _mxfp4_module() -> tuple[ModelFreeLinear, torch.Tensor]:
    packed = torch.tensor(
        [[0x10, 0x2F, 0x83, 0x06] * 4, [0xF0, 0x01, 0x6A, 0xC4] * 4],
        dtype=torch.uint8,
    )
    scale_bits = torch.tensor([[120], [140]], dtype=torch.uint8)
    mod = ModelFreeLinear(
        full_name="layers.0.ffn.experts.0.w1",
        tensor_bindings={
            "weight_packed": TensorRef("weight_packed", "w", "s0", "U8", tuple(packed.shape)),
            "weight_scale": TensorRef("weight_scale", "s", "s0", "U8", tuple(scale_bits.shape)),
        },
        source_ir=SourceIR(kind=IRKind.MXFP4_PACKED),
    )
    mod.register_buffer("weight_packed", packed)
    mod.register_buffer("weight_scale", scale_bits)
    mod.lazy_initialized = True
    from msmodelslim.core.quantization_algorithm.convert.mxfp4_to_float import MxFp4PackedToFloatProcessor

    restored = MxFp4PackedToFloatProcessor().transform(mod, _context()).weight.detach()
    return mod, restored


class TestDirectIntRoutes:
    def test_float_integer_edges(self):
        float_edges = [proc.dst_ir for proc in INT_QUANT_PROCESSORS if proc.src_ir == IRKind.FLOAT]
        assert float_edges == [
            IRKind.W8A16_INT8,
            IRKind.W8A16_INT8_ASYM,
            IRKind.W8A8_INT8,
            IRKind.W4A8_INT4,
            IRKind.W4A4_INT4,
        ]

    @pytest.mark.parametrize(
        ("dst", "quantize", "cls"),
        [
            (IRKind.W8A16_INT8, quantize_w8a16_sym, W8A16StaticPerChannelFakeQuantLinear),
            (IRKind.W8A16_INT8_ASYM, quantize_w8a16_asym, W8A16PerChannelNegOffsetFakeQuantLinear),
            (IRKind.W8A8_INT8, quantize_w8a8, W8A8DynamicPerChannelFakeQuantLinear),
        ],
    )
    def test_fp8_block_quantizes_the_dequantized_weight(self, dst, quantize, cls):
        from msmodelslim.core.quantization_algorithm.convert.fp8_block_to_float import Fp8BlockToFloatProcessor

        weight = torch.rand(_OUT, _IN).add(0.1).to(torch.bfloat16)
        source = _fp8_module(weight)
        restored = Fp8BlockToFloatProcessor().transform(source, _context())
        mod = _processor(dst).transform(restored, _context())
        reference = nn.Linear(_IN, _OUT, bias=False)
        reference.weight = nn.Parameter(weight, requires_grad=False)

        assert _processor(dst).src_ir == IRKind.FLOAT
        assert isinstance(mod, cls)
        assert torch.equal(mod.weight, quantize(reference).weight)
        assert torch.equal(mod.weight_scale, quantize(reference).weight_scale)

    def test_mxfp4_to_w4a8_fake_quant_matches_dequantized_weight(self):
        source, restored = _mxfp4_module()
        restored_linear = nn.Linear(restored.shape[1], restored.shape[0], bias=False)
        restored_linear.weight = nn.Parameter(restored, requires_grad=False)
        mod = _processor(IRKind.W4A8_INT4).transform(restored_linear, _context())
        q, scale = int4_quantize_dynamic_per_channel_sym(restored, axis=0, dtype=torch.int8)

        assert isinstance(mod, W4A8DynamicFakeQuantLinear)
        assert not hasattr(mod, "weight_offset")
        assert mod.weight.dtype == torch.int8
        assert int(mod.weight.min()) >= -8
        assert int(mod.weight.max()) <= 7
        assert torch.equal(mod.weight, q)
        assert torch.equal(mod.weight_scale, scale.squeeze(-1))

        w_q_param = QParam(scheme=int4_per_channel_sym, ext={"scale": mod.weight_scale.data})
        weight_q_dq = dequantize(QStorage(dtype=QDType.INT4, value=mod.weight.data).T, w_q_param).T.value
        per_channel_error = (weight_q_dq.float() - restored.float()).abs()
        assert torch.all(per_channel_error <= mod.weight_scale.float().unsqueeze(-1) + 1e-4)

        x = torch.randn(3, restored.shape[1])
        x_reshape = x.reshape(-1, x.shape[-1])
        x_q_param = calculate_qparam(
            torch.amin(x_reshape, dim=0),
            torch.amax(x_reshape, dim=0),
            QDType.INT8,
            QScope.PER_TOKEN,
            True,
        )
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x_reshape), x_q_param).value.reshape(x.shape)
        expected = F.linear(x_q_dq, weight_q_dq, None)
        assert torch.allclose(mod(x), expected, atol=1e-5, rtol=1e-4)

    def test_w4a8_export_packs_int4_weight(self):
        _, restored = _mxfp4_module()
        restored_linear = nn.Linear(restored.shape[1], restored.shape[0], bias=False)
        restored_linear.weight = nn.Parameter(restored, requires_grad=False)
        mod = _processor(IRKind.W4A8_INT4).transform(restored_linear, _context())
        saver, saved = _capture_saver()
        prefix = "model.layers.0.ffn.experts.0.w1"
        saver.process_map[type(mod)](prefix, mod)

        weight_desc, weight = saved[prefix + ".weight"]
        _, offset = saved[prefix + ".weight_offset"]
        assert weight_desc == "W4A8_DYNAMIC"
        assert weight.dtype == torch.int8
        assert weight.shape == (mod.weight.shape[0] // 2, mod.weight.shape[1])
        assert torch.equal(offset, torch.zeros_like(offset))
        assert prefix + ".scale_bias" in saved
