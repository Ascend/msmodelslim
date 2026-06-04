#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# pylint: disable=redefined-outer-name

import inspect
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from msmodelslim.core.quant_service.modelslim_v1.save.ascendv1 import AscendV1Config, AscendV1Saver
from msmodelslim.ir.qal import QScope, QDType
from msmodelslim.model.interface import IModel
from msmodelslim.utils.exception import SchemaValidateError


def _copy_file_stub(src_path, dest_path):
    shutil.copy(src_path, dest_path)


class _Adapter:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)


class _IModelAdapter(IModel):
    def __init__(self, model_path: str):
        self._model_path = Path(model_path)

    @property
    def model_type(self) -> str:
        return "dummy"

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def trust_remote_code(self) -> bool:
        return False


@pytest.fixture
def saver():
    tmp = tempfile.mkdtemp()
    with patch("msmodelslim.core.quant_service.modelslim_v1.save.ascendv1.dist.is_initialized", return_value=False):
        saver_ins = AscendV1Saver(nn.Linear(4, 4), AscendV1Config(save_directory=tmp, part_file_size=0), _Adapter(tmp))
    saver_ins.json_writer = MagicMock()
    saver_ins.safetensors_writer = MagicMock()
    return saver_ins


def _capture_write_tensor(saver_ins: AscendV1Saver):
    calls = []

    def _write(prefix, desc, tensor):
        calls.append((prefix, desc, tensor.dtype))

    saver_ins.write_tensor = _write
    return calls


class TestAscendV1ExportFormatMethods:
    def test_AscendV1Saver_on_methods_should_be_covered_when_reflecting_class(self):
        expected = {
            "on_w8a8_static",
            "on_w8a16_static_per_channel",
            "on_w8a16_static_per_group",
            "on_w8a8_dynamic_per_channel",
            "on_w8a8_pd_mix",
            "on_w8a8_dynamic_per_group",
            "on_wfp8afp8_dynamic_per_channel",
            "on_w8a8_mx_dynamic_per_block",
            "on_w4a8_dynamic",
            "on_w4a4_dynamic_per_channel",
            "on_w4a4_dynamic_per_group",
            "on_w4a4_mx_dynamic_per_block",
            "on_w4a8_mx_dynamic_per_block",
            "on_float_linear",
            "on_float_module",
            "on_dynamic_cache",
            "on_w16a16s",
            "on_activation_per_head",
            "on_activation_per_token",
            "on_online_rotation_wrapper",
            "on_rotation_wrapper",
            "on_kronecker_rotation_wrapper",
            "on_quarot_extra_info_wrapper",
            "on_flat_clip_wrapper",
            "on_non_fusion_smooth_quant_wrapper",
        }
        actual = {n for n, _ in inspect.getmembers(AscendV1Saver, predicate=callable) if n.startswith("on_")}
        assert expected.issubset(actual)

    def test_w8a8_static_should_export_expected_artifacts_when_invoked(self, saver):
        calls = _capture_write_tensor(saver)
        module = SimpleNamespace(
            weight=torch.randint(-8, 7, (4, 4), dtype=torch.int8),
            input_scale=torch.tensor([0.5], dtype=torch.float32),
            input_offset=torch.tensor([0.0], dtype=torch.float32),
            weight_scale=torch.ones(4, dtype=torch.float32) * 0.1,
            bias=torch.zeros(4, dtype=torch.float32),
        )
        saver.on_w8a8_static("m.q", module)

        assert ("m.q.weight", "W8A8", torch.int8) in calls
        assert ("m.q.quant_bias", "W8A8", torch.int32) in calls
        expected_deq_scale_dtype = torch.float32 if saver._global_torch_dtype_is_bf16 else torch.int64
        assert ("m.q.deq_scale", "W8A8", expected_deq_scale_dtype) in calls

    def test_w8a16_per_channel_should_export_weight_when_invoked(self, saver):
        calls = _capture_write_tensor(saver)
        module = SimpleNamespace(
            weight=torch.randint(-8, 7, (4, 4), dtype=torch.int8),
            weight_scale=torch.ones(4, dtype=torch.float32),
            weight_offset=torch.zeros(4, dtype=torch.float32),
            bias=torch.zeros(4, dtype=torch.float32),
        )
        saver.on_w8a16_static_per_channel("m.pc", module)
        assert ("m.pc.weight", "W8A16", torch.int8) in calls

    def test_w8a16_per_group_should_set_group_size_when_invoked(self, saver):
        calls = _capture_write_tensor(saver)
        module = SimpleNamespace(
            weight=torch.randint(-8, 7, (4, 4), dtype=torch.int8),
            weight_scale=torch.ones(4, dtype=torch.float32),
            weight_offset=torch.zeros(4, dtype=torch.float32),
            bias=torch.zeros(4, dtype=torch.float32),
            group_size=64,
        )
        saver.on_w8a16_static_per_group("m.pg", module)
        assert saver.group_size == 64
        assert ("m.pg.weight_scale", "W8A16", torch.float32) in calls

    def test_w8a8_dynamic_per_channel_should_export_weight_when_invoked(self, saver):
        calls = _capture_write_tensor(saver)
        module = SimpleNamespace(
            weight=torch.randint(-8, 7, (4, 4), dtype=torch.int8), weight_scale=torch.ones(4), bias=torch.zeros(4)
        )
        saver.on_w8a8_dynamic_per_channel("m.dpc", module)
        assert ("m.dpc.weight", "W8A8_DYNAMIC", torch.int8) in calls

    def test_w8a8_pd_mix_should_export_weight_when_invoked(self, saver):
        calls = _capture_write_tensor(saver)
        module = SimpleNamespace(
            weight=torch.randint(-8, 7, (4, 4), dtype=torch.int8),
            input_scale=torch.tensor([0.5]),
            input_offset=torch.tensor([0.0]),
            weight_scale=torch.ones(4) * 0.1,
            bias=torch.zeros(4),
        )
        saver.on_w8a8_pd_mix("m.mix", module)
        assert ("m.mix.weight", "W8A8_MIX", torch.int8) in calls

    def test_w8a8_dynamic_per_group_should_set_group_size_when_invoked(self, saver):
        calls = _capture_write_tensor(saver)
        module = SimpleNamespace(
            weight=torch.randint(-8, 7, (4, 4), dtype=torch.int8),
            weight_scale=torch.ones((4, 2)),
            group_size=128,
            bias=torch.zeros(4),
        )
        saver.on_w8a8_dynamic_per_group("m.dpg", module)
        assert saver.group_size == 128
        assert ("m.dpg.weight", "W8A8_DYNAMIC", torch.int8) in calls

    def test_wfp8afp8_dynamic_per_channel_should_export_fp8_weight_when_invoked(self, saver):
        calls = _capture_write_tensor(saver)
        module = SimpleNamespace(weight=torch.randn(4, 4), weight_scale=torch.ones(4), bias=torch.zeros(4))
        saver.on_wfp8afp8_dynamic_per_channel("m.fp8", module)
        assert ("m.fp8.weight", "WFP8AFP8_DYNAMIC", torch.float8_e4m3fn) in calls

    def test_w8a8_mx_per_block_should_export_uint8_weight_scale_when_invoked(self, saver):
        calls = _capture_write_tensor(saver)
        module = SimpleNamespace(
            weight=torch.randn(4, 4), weight_scale=torch.zeros((4, 1), dtype=torch.int32), w_axes=1, bias=torch.zeros(4)
        )
        saver.on_w8a8_mx_dynamic_per_block("m.mx", module)
        assert ("m.mx.weight_scale", "W8A8_MXFP8", torch.uint8) in calls

    def test_w8a8_mx_per_block_should_raise_error_when_w_axes_invalid(self, saver):
        module = SimpleNamespace(
            weight=torch.randn(4, 4), weight_scale=torch.zeros((4, 1), dtype=torch.int32), w_axes=object(), bias=None
        )
        with pytest.raises(SchemaValidateError):
            saver.on_w8a8_mx_dynamic_per_block("m.mx", module)

    def test_w4a8_dynamic_should_export_weight_when_invoked(self, saver):
        calls = _capture_write_tensor(saver)
        module = SimpleNamespace(
            weight=torch.randint(-8, 7, (4, 4), dtype=torch.int8), weight_scale=torch.ones(4), bias=torch.zeros(4)
        )
        saver.on_w4a8_dynamic("m.w4a8", module)
        assert ("m.w4a8.weight", "W4A8_DYNAMIC", torch.int8) in calls

    def test_w4a4_per_channel_should_export_weight_when_invoked(self, saver):
        calls = _capture_write_tensor(saver)
        module = SimpleNamespace(
            weight=torch.randint(-8, 7, (4, 4), dtype=torch.int8),
            weight_scale=torch.ones(4),
            weight_offset=torch.zeros(4),
            bias=torch.zeros(4),
        )
        saver.on_w4a4_dynamic_per_channel("m.w4c", module)
        assert ("m.w4c.weight", "W4A4_DYNAMIC", torch.int8) in calls

    def test_w4a4_per_group_should_set_group_size_when_invoked(self, saver):
        calls = _capture_write_tensor(saver)
        module = SimpleNamespace(
            weight=torch.randint(-8, 7, (4, 4), dtype=torch.int8),
            weight_scale=torch.ones(4),
            weight_offset=torch.zeros(4),
            bias=torch.zeros(4),
            group_size=32,
        )
        saver.on_w4a4_dynamic_per_group("m.w4g", module)
        assert saver.group_size == 32
        assert ("m.w4g.weight_scale", "W4A4_DYNAMIC", torch.float32) in calls

    def test_w4a4_mx_per_block_should_export_weight_when_invoked(self, saver):
        calls = _capture_write_tensor(saver)
        module = SimpleNamespace(
            weight=torch.randn(4, 4), weight_scale=torch.zeros((4, 1), dtype=torch.int32), w_axes=1, bias=torch.zeros(4)
        )
        saver.on_w4a4_mx_dynamic_per_block("m.w4mx4", module)
        valid_dtypes = {torch.float8_e4m3fn, torch.uint8}
        assert any(
            prefix == "m.w4mx4.weight" and desc == "W4A4_MXFP4" and dtype in valid_dtypes
            for prefix, desc, dtype in calls
        )

    def test_w4a8_mx_per_block_should_export_weight_when_invoked(self, saver):
        calls = _capture_write_tensor(saver)
        module = SimpleNamespace(
            weight=torch.randn(4, 4), weight_scale=torch.zeros((4, 1), dtype=torch.int32), w_axes=1, bias=torch.zeros(4)
        )
        saver.on_w4a8_mx_dynamic_per_block("m.w4mx8", module)
        valid_dtypes = {torch.float8_e4m3fn, torch.uint8}
        assert any(
            prefix == "m.w4mx8.weight" and desc == "W4A8_MXFP" and dtype in valid_dtypes
            for prefix, desc, dtype in calls
        )

    def test_w4a4_mx_per_block_should_raise_error_when_w_axes_invalid(self, saver):
        module = SimpleNamespace(
            weight=torch.randn(4, 4), weight_scale=torch.zeros((4, 1), dtype=torch.int32), w_axes=object(), bias=None
        )
        with pytest.raises(SchemaValidateError):
            saver.on_w4a4_mx_dynamic_per_block("m.bad", module)

    def test_w4a8_mx_per_block_should_raise_error_when_w_axes_invalid(self, saver):
        module = SimpleNamespace(
            weight=torch.randn(4, 4), weight_scale=torch.zeros((4, 1), dtype=torch.int32), w_axes=object(), bias=None
        )
        with pytest.raises(SchemaValidateError):
            saver.on_w4a8_mx_dynamic_per_block("m.bad", module)

    def test_float_linear_should_export_weight_when_invoked(self, saver):
        calls = _capture_write_tensor(saver)
        linear = nn.Linear(4, 3, bias=True)
        saver.on_float_linear("m.float", linear)
        assert ("m.float.weight", "FLOAT", linear.weight.dtype) in calls

    def test_dynamic_cache_should_export_k_proj_scale_when_key_states_input(self, saver):
        calls = _capture_write_tensor(saver)
        cache = SimpleNamespace(kv_cache_scale=torch.ones(2), kv_cache_offset=torch.zeros(2))
        saver.on_dynamic_cache("model.layers.0.self_attn.key_states", cache)
        assert ("model.layers.0.self_attn.k_proj.kv_cache_scale", "C8", torch.float32) in calls

    def test_w16a16s_should_export_weight_when_invoked(self, saver):
        calls = _capture_write_tensor(saver)
        module = SimpleNamespace(named_parameters=lambda recurse=False, prefix="": [("m.w16.weight", torch.ones(2, 2))])
        saver.on_w16a16s("m.w16", module)
        assert ("m.w16.weight", "W16A16S", torch.float32) in calls

    def test_activation_per_head_should_export_scale_when_invoked(self, saver):
        calls = _capture_write_tensor(saver)
        head = SimpleNamespace(input_scale=torch.ones(2))
        saver.on_activation_per_head("m.head", head)
        assert ("m.head.scale", "FAQuant", torch.float32) in calls

    def test_activation_per_token_should_write_fp8_quant_type_when_dtype_fp8(self, saver):
        token = SimpleNamespace(x_q_scheme=SimpleNamespace(dtype=QDType.FP8_E4M3, scope=QScope.PER_TOKEN))
        saver.on_activation_per_token("model.layers.0.self_attn.fa3_q", token)
        saver.json_writer.write.assert_called_with("model.layers.0.self_attn.quant_type", "FP8_DYNAMIC")

    def test_activation_per_token_should_write_int8_quant_type_when_dtype_int8(self, saver):
        token = SimpleNamespace(x_q_scheme=SimpleNamespace(dtype=QDType.INT8, scope=QScope.PER_TOKEN))
        saver.on_activation_per_token("model.layers.0.self_attn.fa3_k", token)
        saver.json_writer.write.assert_called_with("model.layers.0.self_attn.quant_type", "INT8_DYNAMIC")

    def test_activation_per_token_should_raise_error_when_dtype_invalid(self, saver):
        token = SimpleNamespace(x_q_scheme=SimpleNamespace(dtype=QDType.INT4, scope=QScope.PER_TOKEN))
        with pytest.raises(SchemaValidateError):
            saver.on_activation_per_token("model.layers.0.self_attn.fa3_x", token)

    def test_online_rotation_wrapper_should_export_rotation_matrix_when_invoked(self, saver):
        calls = _capture_write_tensor(saver)
        module = SimpleNamespace(rotation_info=SimpleNamespace(rotation_matrix=torch.eye(4)))
        saver.on_online_rotation_wrapper("model.rotation", module)
        assert ("model.rotation", "FLOAT", torch.float32) in calls

    def test_rotation_wrapper_should_export_heads_rotation_when_invoked(self, saver):
        saver.safetensors_writer = MagicMock()
        rot = SimpleNamespace(rotation_info=SimpleNamespace(heads_rotation=torch.eye(2)))
        saver.on_rotation_wrapper("model.layers.0.self_attn", rot)
        args, _ = saver.safetensors_writer.write.call_args
        assert args[0] == "model.layers.0.self_attn.heads_rotation"

    def test_kronecker_rotation_wrapper_should_export_two_tensors_when_invoked(self, saver):
        saver.safetensors_writer = MagicMock()
        module = SimpleNamespace(
            rotation_info=SimpleNamespace(kronecker_rotation_m=torch.eye(2), kronecker_rotation_n=torch.eye(2))
        )
        saver.on_kronecker_rotation_wrapper("model.layers.0.self_attn", module)
        assert saver.safetensors_writer.write.call_count == 2

    def test_quarot_extra_info_wrapper_should_store_optional_info_when_invoked(self, saver):
        with patch("msmodelslim.core.quant_service.modelslim_v1.save.ascendv1.SafetensorsWriter"):
            module = SimpleNamespace(rotation_info=SimpleNamespace(global_rotation=torch.eye(2)))
            saver.on_quarot_extra_info_wrapper("model", module)
        assert "quarot" in saver.json_optional_infos

    def test_flat_clip_wrapper_should_export_flatquant_desc_when_invoked(self, saver):
        calls = _capture_write_tensor(saver)
        with patch.object(saver, "_process_module") as mock_process:
            mock_process.side_effect = lambda prefix, module: saver.write_tensor(
                f"{prefix}.weight", "W8A8_DYNAMIC", torch.ones(2, 2)
            )
            wrapper = SimpleNamespace(
                wrapped_module=SimpleNamespace(), save_trans={"left_trans": torch.ones(2, 2)}, clip_factor=torch.ones(1)
            )
            saver.on_flat_clip_wrapper("model.layers.0.ffn", wrapper)
        assert any(desc.endswith("_FLATQUANT_DYNAMIC") for _, desc, _ in calls)

    def test_non_fusion_smooth_quant_wrapper_should_export_mul_scale_when_invoked(self, saver):
        calls = _capture_write_tensor(saver)
        with patch.object(saver, "_process_module") as mock_process:
            mock_process.side_effect = lambda prefix, module: saver.write_tensor(
                f"{prefix}.weight", "FLOAT", torch.ones(2, 2)
            )
            wrapper = SimpleNamespace(wrapped_module=SimpleNamespace(), scales=torch.ones(4))
            saver.on_non_fusion_smooth_quant_wrapper("model.layers.1.mlp", wrapper)
        assert ("model.layers.1.mlp.div.mul_scale", "FLOAT", torch.float32) in calls


def _build_post_run_case_module(method_name: str):
    if method_name in {"on_w8a8_static", "on_w8a8_pd_mix"}:
        return (
            "m.q",
            SimpleNamespace(
                weight=torch.randint(-8, 7, (4, 4), dtype=torch.int8),
                input_scale=torch.tensor([0.5], dtype=torch.float32),
                input_offset=torch.tensor([0.0], dtype=torch.float32),
                weight_scale=torch.ones(4, dtype=torch.float32) * 0.1,
                bias=torch.zeros(4, dtype=torch.float32),
            ),
            "m.q.weight",
        )
    if method_name in {"on_w8a16_static_per_channel", "on_w8a16_static_per_group"}:
        mod = SimpleNamespace(
            weight=torch.randint(-8, 7, (4, 4), dtype=torch.int8),
            weight_scale=torch.ones(4, dtype=torch.float32),
            weight_offset=torch.zeros(4, dtype=torch.float32),
            bias=torch.zeros(4, dtype=torch.float32),
        )
        if method_name == "on_w8a16_static_per_group":
            mod.group_size = 64
        return "m.w8a16", mod, "m.w8a16.weight"
    if method_name in {"on_w8a8_dynamic_per_channel", "on_w8a8_dynamic_per_group"}:
        mod = SimpleNamespace(
            weight=torch.randint(-8, 7, (4, 4), dtype=torch.int8),
            weight_scale=torch.ones((4, 2), dtype=torch.float32)
            if method_name.endswith("group")
            else torch.ones(4, dtype=torch.float32),
            bias=torch.zeros(4, dtype=torch.float32),
        )
        if method_name.endswith("group"):
            mod.group_size = 128
        return "m.w8dyn", mod, "m.w8dyn.weight"
    if method_name == "on_wfp8afp8_dynamic_per_channel":
        return (
            "m.fp8",
            SimpleNamespace(
                weight=torch.randn(4, 4),
                weight_scale=torch.ones(4),
                bias=torch.zeros(4),
            ),
            "m.fp8.weight",
        )
    if method_name in {"on_w8a8_mx_dynamic_per_block", "on_w4a4_mx_dynamic_per_block", "on_w4a8_mx_dynamic_per_block"}:
        return (
            "m.mx",
            SimpleNamespace(
                weight=torch.randn(4, 4),
                weight_scale=torch.zeros((4, 1), dtype=torch.int32),
                w_axes=1,
                bias=torch.zeros(4),
            ),
            "m.mx.weight",
        )
    if method_name == "on_w4a8_dynamic":
        return (
            "m.w4a8",
            SimpleNamespace(
                weight=torch.randint(-8, 7, (4, 4), dtype=torch.int8),
                weight_scale=torch.ones(4),
                bias=torch.zeros(4),
            ),
            "m.w4a8.weight",
        )
    if method_name in {"on_w4a4_dynamic_per_channel", "on_w4a4_dynamic_per_group"}:
        mod = SimpleNamespace(
            weight=torch.randint(-8, 7, (4, 4), dtype=torch.int8),
            weight_scale=torch.ones(4),
            weight_offset=torch.zeros(4),
            bias=torch.zeros(4),
        )
        if method_name.endswith("group"):
            mod.group_size = 32
        return "m.w4a4", mod, "m.w4a4.weight"
    if method_name == "on_float_linear":
        return "m.float", nn.Linear(4, 3, bias=True), "m.float.weight"
    if method_name == "on_float_module":
        mod = nn.Module()
        mod.register_parameter("weight", nn.Parameter(torch.ones(2, 2)))
        return "m.module", mod, "m.module.weight"
    if method_name == "on_dynamic_cache":
        return (
            "model.layers.0.self_attn.key_states",
            SimpleNamespace(
                kv_cache_scale=torch.ones(2),
                kv_cache_offset=torch.zeros(2),
            ),
            "model.layers.0.self_attn.k_proj.kv_cache_scale",
        )
    if method_name == "on_w16a16s":
        return (
            "m.w16",
            SimpleNamespace(named_parameters=lambda recurse=False, prefix="": [("m.w16.weight", torch.ones(2, 2))]),
            "m.w16.weight",
        )
    if method_name == "on_activation_per_head":
        return "m.head", SimpleNamespace(input_scale=torch.ones(2)), "m.head.scale"
    if method_name == "on_activation_per_token":
        return (
            "model.layers.0.self_attn.fa3_q",
            SimpleNamespace(x_q_scheme=SimpleNamespace(dtype=QDType.FP8_E4M3, scope=QScope.PER_TOKEN)),
            "model.layers.0.self_attn.quant_type",
        )
    if method_name == "on_online_rotation_wrapper":
        return (
            "model.rotation",
            SimpleNamespace(rotation_info=SimpleNamespace(rotation_matrix=torch.eye(4))),
            "model.rotation",
        )
    if method_name == "on_rotation_wrapper":
        rot_info = SimpleNamespace(heads_rotation=torch.eye(2))
        rot_info.get_quarot_save_info = lambda: {"heads_rotation": {"layers": []}}
        return (
            "model.layers.0.self_attn",
            SimpleNamespace(rotation_info=rot_info),
            "model.layers.0.self_attn.heads_rotation",
        )
    if method_name == "on_kronecker_rotation_wrapper":
        rot_info = SimpleNamespace(kronecker_rotation_m=torch.eye(2), kronecker_rotation_n=torch.eye(2))
        rot_info.get_quarot_save_info = lambda: {"kronecker_rotation": {"layers": []}}
        return (
            "model.layers.0.self_attn",
            SimpleNamespace(rotation_info=rot_info),
            "model.layers.0.self_attn.kronecker_rotation_m",
        )
    if method_name == "on_quarot_extra_info_wrapper":
        return "model", SimpleNamespace(rotation_info=SimpleNamespace(global_rotation=torch.eye(2))), "optional"
    if method_name == "on_flat_clip_wrapper":
        return (
            "model.layers.0.ffn",
            SimpleNamespace(
                wrapped_module=SimpleNamespace(),
                save_trans={"left_trans": torch.ones(2, 2)},
                clip_factor=torch.ones(1),
            ),
            "model.layers.0.ffn.left_trans",
        )
    if method_name == "on_non_fusion_smooth_quant_wrapper":
        return (
            "model.layers.1.mlp",
            SimpleNamespace(
                wrapped_module=SimpleNamespace(),
                scales=torch.ones(4),
            ),
            "model.layers.1.mlp.div.mul_scale",
        )
    raise AssertionError(f"unsupported on_xxx method: {method_name}")


def _expected_desc_value_and_group_size(method_name: str):
    expected_desc_value = {
        "on_w8a8_static": "W8A8",
        "on_w8a16_static_per_channel": "W8A16",
        "on_w8a16_static_per_group": "W8A16",
        "on_w8a8_dynamic_per_channel": "W8A8_DYNAMIC",
        "on_w8a8_pd_mix": "W8A8_MIX",
        "on_w8a8_dynamic_per_group": "W8A8_DYNAMIC",
        "on_wfp8afp8_dynamic_per_channel": "WFP8AFP8_DYNAMIC",
        "on_w8a8_mx_dynamic_per_block": "W8A8_MXFP8",
        "on_w4a8_dynamic": "W4A8_DYNAMIC",
        "on_w4a4_dynamic_per_channel": "W4A4_DYNAMIC",
        "on_w4a4_dynamic_per_group": "W4A4_DYNAMIC",
        "on_w4a4_mx_dynamic_per_block": "W4A4_MXFP4",
        "on_w4a8_mx_dynamic_per_block": "W4A8_MXFP",
        "on_float_linear": "FLOAT",
        "on_float_module": "FLOAT",
        "on_dynamic_cache": "C8",
        "on_w16a16s": "W16A16S",
        "on_activation_per_head": "FAQuant",
        "on_activation_per_token": "FP8_DYNAMIC",
        "on_online_rotation_wrapper": "FLOAT",
        "on_flat_clip_wrapper": "W8A8_FLATQUANT_DYNAMIC",
        "on_non_fusion_smooth_quant_wrapper": "FLOAT",
    }.get(method_name, None)

    expected_group_size = {
        "on_w8a16_static_per_group": 64,
        "on_w8a8_dynamic_per_group": 128,
        "on_w4a4_dynamic_per_group": 32,
        "on_w8a8_mx_dynamic_per_block": 32,
        "on_w4a4_mx_dynamic_per_block": 32,
        "on_w4a8_mx_dynamic_per_block": 32,
    }.get(method_name, 0)

    return expected_desc_value, expected_group_size


class TestAscendV1PostRunArtifacts:
    @pytest.mark.parametrize(
        "method_name",
        [
            "on_w8a8_static",
            "on_w8a16_static_per_channel",
            "on_w8a16_static_per_group",
            "on_w8a8_dynamic_per_channel",
            "on_w8a8_pd_mix",
            "on_w8a8_dynamic_per_group",
            "on_wfp8afp8_dynamic_per_channel",
            "on_w8a8_mx_dynamic_per_block",
            "on_w4a8_dynamic",
            "on_w4a4_dynamic_per_channel",
            "on_w4a4_dynamic_per_group",
            "on_w4a4_mx_dynamic_per_block",
            "on_w4a8_mx_dynamic_per_block",
            "on_float_linear",
            "on_float_module",
            "on_dynamic_cache",
            "on_w16a16s",
            "on_activation_per_head",
            "on_activation_per_token",
            "on_online_rotation_wrapper",
            "on_rotation_wrapper",
            "on_kronecker_rotation_wrapper",
            "on_quarot_extra_info_wrapper",
            "on_flat_clip_wrapper",
            "on_non_fusion_smooth_quant_wrapper",
        ],
    )
    def test_each_on_method_should_generate_post_run_artifacts_when_exported(self, tmp_path, method_name):
        model_src = tmp_path / "src_model"
        model_src.mkdir()
        (model_src / "config.json").write_text(
            json.dumps({"model_type": "llama", "quantization_config": {"bits": 8}}),
            encoding="utf-8",
        )
        (model_src / "tokenizer_config.json").write_text('{"tokenizer_class":"LlamaTokenizer"}', encoding="utf-8")
        (model_src / "modeling_dummy.py").write_text("DUMMY = True\n", encoding="utf-8")

        save_dir = tmp_path / f"exported_{method_name}"
        save_dir.mkdir()

        with patch("msmodelslim.core.quant_service.modelslim_v1.save.ascendv1.dist.is_initialized", return_value=False):
            saver_ins = AscendV1Saver(
                nn.Linear(4, 4),
                AscendV1Config(save_directory=str(save_dir), part_file_size=1),
                _IModelAdapter(str(model_src)),
            )

        prefix, module, expected_key = _build_post_run_case_module(method_name)
        expected_desc_value, expected_group_size = _expected_desc_value_and_group_size(method_name)

        if method_name in {"on_flat_clip_wrapper", "on_non_fusion_smooth_quant_wrapper"}:
            saver_ins._process_module = lambda p, m: saver_ins.write_tensor(
                f"{p}.weight", "W8A8_DYNAMIC", torch.ones(2, 2)
            )

        getattr(saver_ins, method_name)(prefix, module)
        with patch(
            "msmodelslim.core.quant_service.modelslim_v1.save.ascendv1.safe_copy_file",
            side_effect=_copy_file_stub,
        ):
            saver_ins.post_run()

        desc_path = save_dir / "quant_model_description.json"
        assert desc_path.exists()
        desc = json.loads(desc_path.read_text(encoding="utf-8"))

        index_path = save_dir / "quant_model_weights.safetensors.index.json"
        assert index_path.exists()
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        assert "metadata" in index_data and "weight_map" in index_data

        assert (save_dir / "tokenizer_config.json").exists()
        assert (save_dir / "modeling_dummy.py").exists()
        copied_config = json.loads((save_dir / "config.json").read_text(encoding="utf-8"))
        assert "quantization_config" not in copied_config
        assert desc.get("group_size") == expected_group_size
        if method_name == "on_activation_per_head":
            assert desc.get("fa_quant_type") == "FAKQuant"

        if expected_key == "optional":
            assert "optional" in desc
        elif expected_key.endswith(".quant_type"):
            assert expected_key in desc
            assert desc[expected_key] == expected_desc_value
        elif expected_key.endswith("heads_rotation") or expected_key.endswith("kronecker_rotation_m"):
            assert expected_key in index_data["weight_map"]
        else:
            assert expected_key in desc
            assert desc[expected_key] == expected_desc_value
            assert expected_key in index_data["weight_map"]
