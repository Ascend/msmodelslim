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

from msmodelslim.core.quant_service.modelslim_v1.save.mindie_format import MindIEFormatConfig, MindIEFormatSaver
from msmodelslim.ir.qal import QScope, QDType
from msmodelslim.utils.exception import SchemaValidateError


def _copy_file_stub(src_path, dest_path):
    shutil.copy(src_path, dest_path)


class _Adapter:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)


@pytest.fixture
def saver():
    tmp = tempfile.mkdtemp()
    with patch(
        "msmodelslim.core.quant_service.modelslim_v1.save.mindie_format.dist.is_initialized", return_value=False
    ):
        saver_ins = MindIEFormatSaver(
            nn.Linear(4, 4), MindIEFormatConfig(save_directory=tmp, part_file_size=0), _Adapter(tmp)
        )
    saver_ins.json_writer = MagicMock()
    saver_ins.safetensors_writer = MagicMock()
    return saver_ins


def _capture_write_tensor(saver_ins: MindIEFormatSaver):
    calls = []

    def _write(prefix, desc, tensor):
        calls.append((prefix, desc, tensor.dtype))

    saver_ins.write_tensor = _write
    return calls


def _mindie_post_run_case(method_name: str):
    if method_name == "on_w8a8_static":
        return (
            "m.q",
            SimpleNamespace(
                weight=torch.randint(-8, 7, (4, 4), dtype=torch.int8),
                input_scale=torch.tensor([0.5], dtype=torch.float32),
                input_offset=torch.tensor([0.0], dtype=torch.float32),
                weight_scale=torch.ones(4, dtype=torch.float32),
                bias=torch.zeros(4, dtype=torch.float32),
            ),
            "m.q.weight",
            "w8a8",
        )
    if method_name == "on_w8a8_dynamic_per_channel":
        return (
            "m.dq",
            SimpleNamespace(
                weight=torch.randint(-8, 7, (4, 4), dtype=torch.int8),
                weight_scale=torch.ones(4, dtype=torch.float32),
                bias=torch.zeros(4, dtype=torch.float32),
            ),
            "m.dq.weight",
            "w8a8_dynamic",
        )
    if method_name == "on_w8a8_mx_dynamic_per_block":
        return (
            "m.mx",
            SimpleNamespace(
                weight=torch.randn(4, 4, dtype=torch.float32),
                weight_scale=torch.zeros((4, 1), dtype=torch.int32),
                w_axes=1,
                bias=torch.zeros(4, dtype=torch.float32),
            ),
            "m.mx.weight",
            "w8a8_mxfp8",
        )
    if method_name == "on_online_rotation_wrapper":
        return (
            "model.rotation",
            SimpleNamespace(rotation_info=SimpleNamespace(rotation_matrix=torch.eye(4, dtype=torch.float32))),
            "model.rotation",
            None,
        )
    if method_name == "on_float_linear":
        return "m.float", nn.Linear(4, 3, bias=True), "m.float.weight", None
    if method_name == "on_float_module":
        mod = nn.Module()
        mod.register_parameter("weight", nn.Parameter(torch.ones(2, 2)))
        return "m.module", mod, "m.module.weight", None
    if method_name == "on_activation_per_token":
        return (
            "model.layers.0.self_attn.fa3_q",
            SimpleNamespace(x_q_scheme=SimpleNamespace(dtype=QDType.FP8_E4M3, scope=QScope.PER_TOKEN)),
            "model.layers.0.self_attn.quant_type",
            None,
        )
    if method_name == "on_activation_per_block":
        return (
            "model.layers.0.self_attn.fa3_q",
            SimpleNamespace(x_q_scheme=SimpleNamespace(dtype=QDType.MXFP4, scope=QScope.PER_BLOCK)),
            "model.layers.0.self_attn.quant_type",
            None,
        )
    raise AssertionError(method_name)


def _mindie_expected_desc_value(method_name: str):
    return {
        "on_w8a8_static": "W8A8",
        "on_w8a8_dynamic_per_channel": "W8A8_DYNAMIC",
        "on_w8a8_mx_dynamic_per_block": "W8A8_MXFP8",
        "on_online_rotation_wrapper": "FLOAT",
        "on_float_linear": "FLOAT",
        "on_float_module": "FLOAT",
        "on_activation_per_token": "Q_FP8_DYNAMIC",
        "on_activation_per_block": "Q_MXFP4_DYNAMIC",
    }[method_name]


class TestMindIEFormatSaver:
    def test_MindIEFormatSaver_defines_expected_on_handlers_when_class_inspected(self):
        expected = {
            "on_w8a8_static",
            "on_w8a8_dynamic_per_channel",
            "on_w8a8_mx_dynamic_per_block",
            "on_online_rotation_wrapper",
            "on_float_linear",
            "on_float_module",
            "on_activation_per_token",
            "on_activation_per_block",
        }
        actual = {n for n, _ in inspect.getmembers(MindIEFormatSaver, predicate=callable) if n.startswith("on_")}
        assert expected.issubset(actual)

    def test_MindIEFormatSaver_writes_w8a8_tensors_and_json_when_on_w8a8_static(self, saver):
        calls = _capture_write_tensor(saver)
        module = SimpleNamespace(
            weight=torch.randint(-8, 7, (4, 4), dtype=torch.int8),
            input_scale=torch.tensor([0.5], dtype=torch.float32),
            input_offset=torch.tensor([0.0], dtype=torch.float32),
            weight_scale=torch.ones(4, dtype=torch.float32),
            bias=torch.zeros(4, dtype=torch.float32),
        )
        saver.on_w8a8_static("m.q", module)
        assert [(k, d) for k, d, _ in calls] == [
            ("m.q.weight", "W8A8"),
            ("m.q.quant_bias", "W8A8"),
            ("m.q.input_scale", "W8A8"),
            ("m.q.input_offset", "W8A8"),
            ("m.q.deq_scale", "W8A8"),
            ("m.q.bias", "FLOAT"),
        ]
        assert saver.json_append["json_append"]["model_quant_type"] == "W8A8"

    def test_MindIEFormatSaver_writes_w8a8_dynamic_tensors_and_json_when_on_w8a8_dynamic_per_channel(self, saver):
        calls = _capture_write_tensor(saver)
        module = SimpleNamespace(
            weight=torch.randint(-8, 7, (4, 4), dtype=torch.int8),
            weight_scale=torch.ones(4, dtype=torch.float32),
            bias=torch.zeros(4, dtype=torch.float32),
        )
        saver.on_w8a8_dynamic_per_channel("m.dq", module)
        assert ("m.dq.weight", "W8A8_DYNAMIC", torch.int8) in calls
        assert ("m.dq.weight_scale", "W8A8_DYNAMIC", torch.float32) in calls
        assert ("m.dq.weight_offset", "W8A8_DYNAMIC", torch.float32) in calls
        assert saver.json_append["json_append"]["model_quant_type"] == "W8A8_DYNAMIC"

    def test_MindIEFormatSaver_writes_w8a8_mxfp8_tensors_and_json_when_on_w8a8_mx_dynamic_per_block(self, saver):
        calls = _capture_write_tensor(saver)
        module = SimpleNamespace(
            weight=torch.randn(4, 4, dtype=torch.float32),
            weight_scale=torch.zeros((4, 1), dtype=torch.int32),
            w_axes=1,
            bias=torch.zeros(4, dtype=torch.float32),
        )
        saver.on_w8a8_mx_dynamic_per_block("m.mx", module)
        assert ("m.mx.weight", "W8A8_MXFP8", torch.float8_e4m3fn) in calls
        assert ("m.mx.weight_scale", "W8A8_MXFP8", torch.uint8) in calls
        assert saver.json_append["json_append"]["model_quant_type"] == "W8A8_MXFP8"

    def test_MindIEFormatSaver_raises_schema_validate_error_when_w8a8_mx_w_axes_invalid(self, saver):
        module = SimpleNamespace(
            weight=torch.randn(4, 4, dtype=torch.float32),
            weight_scale=torch.zeros((4, 1), dtype=torch.int32),
            w_axes=object(),
            bias=None,
        )
        with pytest.raises(SchemaValidateError):
            saver.on_w8a8_mx_dynamic_per_block("m.mx", module)

    def test_MindIEFormatSaver_writes_float_rotation_tensor_only_when_on_online_rotation_wrapper(self, saver):
        calls = _capture_write_tensor(saver)
        module = SimpleNamespace(rotation_info=SimpleNamespace(rotation_matrix=torch.eye(4, dtype=torch.float32)))
        saver.on_online_rotation_wrapper("model.rotation", module)
        assert calls == [("model.rotation", "FLOAT", torch.float32)]

    def test_MindIEFormatSaver_writes_float_weight_and_bias_when_on_float_linear(self, saver):
        calls = _capture_write_tensor(saver)
        linear = nn.Linear(4, 3, bias=True)
        saver.on_float_linear("m.float", linear)
        assert ("m.float.weight", "FLOAT", linear.weight.dtype) in calls
        assert ("m.float.bias", "FLOAT", linear.bias.dtype) in calls

    def test_MindIEFormatSaver_writes_float_submodule_param_when_on_float_module(self, saver):
        calls = _capture_write_tensor(saver)
        mod = nn.Module()
        mod.register_parameter("weight", nn.Parameter(torch.ones(2, 2)))
        saver.on_float_module("m.module", mod)
        assert ("m.module.weight", "FLOAT", torch.float32) in calls

    def test_MindIEFormatSaver_json_writes_fp8_dynamic_quant_type_when_on_activation_per_token(self, saver):
        module = SimpleNamespace(x_q_scheme=SimpleNamespace(dtype=QDType.FP8_E4M3, scope=QScope.PER_TOKEN))
        saver.on_activation_per_token("model.layers.0.self_attn.fa3_q", module)
        saver.json_writer.write.assert_called_once_with("model.layers.0.self_attn.quant_type", "Q_FP8_DYNAMIC")

    def test_MindIEFormatSaver_json_writes_mxfp4_dynamic_quant_type_when_on_activation_per_block(self, saver):
        module = SimpleNamespace(x_q_scheme=SimpleNamespace(dtype=QDType.MXFP4, scope=QScope.PER_BLOCK))
        saver.on_activation_per_block("model.layers.0.self_attn.fa3_q", module)
        saver.json_writer.write.assert_called_once_with("model.layers.0.self_attn.quant_type", "Q_MXFP4_DYNAMIC")

    def test_MindIEFormatSaver_raises_schema_validate_error_when_activation_scheme_invalid(self, saver):
        module = SimpleNamespace(x_q_scheme=SimpleNamespace(dtype=QDType.INT4, scope=QScope.PER_TOKEN))
        with pytest.raises(SchemaValidateError):
            saver.on_activation_per_token("model.layers.0.self_attn.fa3_q", module)

    @patch("msmodelslim.core.quant_service.modelslim_v1.save.mindie_format.dist.is_initialized", return_value=False)
    @pytest.mark.parametrize(
        "method_name",
        [
            "on_w8a8_static",
            "on_w8a8_dynamic_per_channel",
            "on_w8a8_mx_dynamic_per_block",
            "on_online_rotation_wrapper",
            "on_float_linear",
            "on_float_module",
            "on_activation_per_token",
            "on_activation_per_block",
        ],
    )
    def test_MindIEFormatSaver_post_run_artifacts_match_on_method_when_parametrized(
        self, _mock_init, tmp_path, method_name
    ):
        model_src = tmp_path / "src_model"
        model_src.mkdir()
        (model_src / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
        (model_src / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (model_src / "modeling_dummy.py").write_text("DUMMY = True\n", encoding="utf-8")

        save_dir = tmp_path / f"mindie_{method_name}"
        save_dir.mkdir()
        saver_ins = MindIEFormatSaver(
            nn.Linear(4, 4),
            MindIEFormatConfig(save_directory=str(save_dir), part_file_size=1),
            _Adapter(str(model_src)),
        )
        prefix, module, expected_key, quant_suffix = _mindie_post_run_case(method_name)
        expected_desc_value = _mindie_expected_desc_value(method_name)
        getattr(saver_ins, method_name)(prefix, module)

        with patch(
            "msmodelslim.core.quant_service.modelslim_v1.save.mindie_format.safe_copy_file",
            side_effect=_copy_file_stub,
        ):
            saver_ins.post_run()

        if quant_suffix is None:
            desc_path = save_dir / "quant_model_description.json"
        else:
            desc_path = save_dir / f"quant_model_description_{quant_suffix}.json"
        assert desc_path.exists()
        desc = json.loads(desc_path.read_text(encoding="utf-8"))

        index_path = save_dir / "quant_model_weight.safetensors.index.json"
        assert index_path.exists()
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        assert "metadata" in index_data and "weight_map" in index_data
        assert (save_dir / "tokenizer_config.json").exists()
        assert (save_dir / "modeling_dummy.py").exists()
        assert (save_dir / "config.json").exists()

        assert expected_key in desc
        assert desc[expected_key] == expected_desc_value
        if quant_suffix is not None:
            assert desc.get("model_quant_type") == expected_desc_value
        if not expected_key.endswith(".quant_type"):
            assert expected_key in index_data["weight_map"]


class TestMindIEFormatSaverFA3:
    """FA3 新增方法覆盖测试（_save_activation_per_head、update_global_fa_quant_type）。"""

    @pytest.fixture
    def saver(self):
        tmp = tempfile.mkdtemp()
        with patch(
            "msmodelslim.core.quant_service.modelslim_v1.save.mindie_format.dist.is_initialized", return_value=False
        ):
            saver_ins = MindIEFormatSaver(
                nn.Linear(4, 4), MindIEFormatConfig(save_directory=tmp, part_file_size=0), _Adapter(tmp)
            )
        saver_ins.json_writer = MagicMock()
        saver_ins.safetensors_writer = MagicMock()
        return saver_ins

    def test_MindIEFormatSaver_writes_fp8_per_head_scale_offset_when_on_fp8_activation_per_head(self, saver):
        """on_fp8_activation_per_head 应调用 _save_activation_per_head，写入 dtype=float32 的 offset"""
        module = SimpleNamespace(
            input_scale=torch.tensor([[[[0.5]]]], dtype=torch.float32),
            x_q_scheme=SimpleNamespace(dtype=QDType.FP8_E4M3, scope=QScope.PER_HEAD),
        )
        with patch.object(saver, "_save_activation_per_head") as mock_save:
            saver.on_fp8_activation_per_head("blocks.0.self_attn.fa3_q", module)
            mock_save.assert_called_once_with("blocks.0.self_attn.fa3_q", module, torch.float32)

    def test_MindIEFormatSaver_writes_int8_per_head_scale_offset_when_on_int8_activation_per_head(self, saver):
        """on_int8_activation_per_head 应调用 _save_activation_per_head，写入 dtype=int8 的 offset"""
        module = SimpleNamespace(
            input_scale=torch.tensor([[[[0.5]]]], dtype=torch.float32),
            x_q_scheme=SimpleNamespace(dtype=QDType.INT8, scope=QScope.PER_HEAD),
        )
        with patch.object(saver, "_save_activation_per_head") as mock_save:
            saver.on_int8_activation_per_head("blocks.0.self_attn.fa3_q", module)
            mock_save.assert_called_once_with("blocks.0.self_attn.fa3_q", module, torch.int8)

    def test_MindIEFormatSaver_save_activation_per_head_writes_scale_offset_tensors(self, saver):
        """_save_activation_per_head 应写入 scale（float32）和 offset（指定 dtype）。"""
        calls = _capture_write_tensor(saver)
        module = SimpleNamespace(
            input_scale=torch.tensor([[[[0.5]]]], dtype=torch.float32),
            x_q_scheme=SimpleNamespace(dtype=QDType.FP8_E4M3, scope=QScope.PER_HEAD),
        )
        saver._save_activation_per_head("blocks.0.self_attn.fa3_q", module, torch.int8)
        assert ("blocks.0.self_attn.fa3_q.scale", "FAQuant", torch.float32) in calls
        assert ("blocks.0.self_attn.fa3_q.offset", "FAQuant", torch.int8) in calls

    def test_MindIEFormatSaver_save_activation_per_head_unsqueezes_1d_scale(self, saver):
        """_save_activation_per_head 对 1 维 scale 应 unsqueeze 为 2 维。"""
        captured = []

        def _write(prefix, desc, tensor):
            captured.append((prefix, desc, tensor))

        saver.write_tensor = _write
        module = SimpleNamespace(
            input_scale=torch.tensor([0.5], dtype=torch.float32),
            x_q_scheme=SimpleNamespace(dtype=QDType.FP8_E4M3, scope=QScope.PER_HEAD),
        )
        saver._save_activation_per_head("blocks.0.self_attn.fa3_q", module, torch.float32)
        scale_tensor = [t for p, _, t in captured if p == "blocks.0.self_attn.fa3_q.scale"][0]
        assert scale_tensor.ndim == 2, f"Expected 2D scale, got {scale_tensor.ndim}D"

    def test_MindIEFormatSaver_writes_global_fa_quant_type_when_update_global_fa_quant_type(self, saver):
        """update_global_fa_quant_type 应在 fa_quant_states 非空时写入 fa_quant_type。"""
        saver.fa_quant_states = {"blocks.0.self_attn": {"Q": ("FP8", "DYNAMIC")}}
        saver.update_global_fa_quant_type("FAKQuant")
        saver.json_writer.write.assert_called_once_with("fa_quant_type", "FAKQuant")

    def test_MindIEFormatSaver_skips_global_fa_quant_type_when_no_states(self, saver):
        """update_global_fa_quant_type 在 fa_quant_states 为空时应跳过。"""
        saver.fa_quant_states = {}
        saver.update_global_fa_quant_type("FAKQuant")
        saver.json_writer.write.assert_not_called()

    def test_MindIEFormatSaver_update_fa_quant_type_writes_composite_string_when_multiple_acts(self, saver):
        """update_fa_quant_type 在 Q/K/V 配置不同时应输出复合字符串。"""
        # 先写 Q (FP8_DYNAMIC)
        q_module = SimpleNamespace(x_q_scheme=SimpleNamespace(dtype=QDType.FP8_E4M3, scope=QScope.PER_TOKEN))
        saver.on_activation_per_token("blocks.0.self_attn.fa3_q", q_module)
        # 再写 K (INT8_STATIC)
        k_module = SimpleNamespace(x_q_scheme=SimpleNamespace(dtype=QDType.INT8, scope=QScope.PER_HEAD))
        saver.on_activation_per_token("blocks.0.self_attn.fa3_k", k_module)
        # 最后写入应为合并结果
        saver.json_writer.write.assert_called_with("blocks.0.self_attn.quant_type", "Q_FP8_DYNAMIC_K_INT8")

    def test_MindIEFormatSaver_update_fa_quant_type_merges_qkv_when_same_config(self, saver):
        """update_fa_quant_type 在 Q/K/V 配置一致时应省略 act 前缀。"""
        module = SimpleNamespace(x_q_scheme=SimpleNamespace(dtype=QDType.FP8_E4M3, scope=QScope.PER_TOKEN))
        saver.on_activation_per_token("blocks.0.self_attn.fa3_q", module)
        saver.on_activation_per_token("blocks.0.self_attn.fa3_k", module)
        saver.on_activation_per_token("blocks.0.self_attn.fa3_v", module)
        saver.json_writer.write.assert_called_with("blocks.0.self_attn.quant_type", "FP8_DYNAMIC")

    def test_MindIEFormatSaver_update_fa_quant_type_raises_when_unsupported_dtype(self, saver):
        """update_fa_quant_type 遇到不支持的 dtype 应抛出 SchemaValidateError。"""
        module = SimpleNamespace(x_q_scheme=SimpleNamespace(dtype=QDType.INT4, scope=QScope.PER_TOKEN))
        with pytest.raises(SchemaValidateError):
            saver.update_fa_quant_type("blocks.0.self_attn.fa3_q", module)

    def test_MindIEFormatSaver_json_writes_mxfp8_dynamic_quant_type_when_on_activation_per_block(self, saver):
        """on_activation_per_block 在 dtype=MXFP8 时应写入 Q_MXFP8_DYNAMIC。"""
        module = SimpleNamespace(x_q_scheme=SimpleNamespace(dtype=QDType.MXFP8, scope=QScope.PER_BLOCK))
        saver.on_activation_per_block("model.layers.0.self_attn.fa3_q", module)
        saver.json_writer.write.assert_called_once_with("model.layers.0.self_attn.quant_type", "Q_MXFP8_DYNAMIC")

    def test_MindIEFormatSaver_update_fa_quant_type_merges_mxfp8_qkv_when_same_config(self, saver):
        """update_fa_quant_type 在 Q/K/V 均为 MXFP8 PER_BLOCK 时应输出 MXFP8_DYNAMIC。"""
        module = SimpleNamespace(x_q_scheme=SimpleNamespace(dtype=QDType.MXFP8, scope=QScope.PER_BLOCK))
        saver.on_activation_per_block("blocks.0.self_attn.fa3_q", module)
        saver.on_activation_per_block("blocks.0.self_attn.fa3_k", module)
        saver.on_activation_per_block("blocks.0.self_attn.fa3_v", module)
        saver.json_writer.write.assert_called_with("blocks.0.self_attn.quant_type", "MXFP8_DYNAMIC")

    def test_on_mxfp8_activation_per_channel_raises_UnsupportedError_when_called(self, saver):
        """MindIE 暂不支持 MXFP8 per-channel FA 导出，应提示改用 AscendV1。"""
        from msmodelslim.utils.exception import UnsupportedError

        v = SimpleNamespace(
            x_q_scheme=SimpleNamespace(dtype=QDType.MXFP8, scope=QScope.PER_CHANNEL),
            input_scale=torch.tensor([0.0, 1.0], dtype=torch.float32),
        )
        with pytest.raises(UnsupportedError, match="AscendV1Saver"):
            saver.on_mxfp8_activation_per_channel("blocks.0.self_attn.fa_v", v)
