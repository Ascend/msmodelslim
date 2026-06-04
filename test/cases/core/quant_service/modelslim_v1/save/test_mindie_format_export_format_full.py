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
    raise AssertionError(method_name)


def _mindie_expected_desc_value(method_name: str):
    return {
        "on_w8a8_static": "W8A8",
        "on_w8a8_dynamic_per_channel": "W8A8_DYNAMIC",
        "on_w8a8_mx_dynamic_per_block": "W8A8_MXFP8",
        "on_online_rotation_wrapper": "FLOAT",
        "on_float_linear": "FLOAT",
        "on_float_module": "FLOAT",
        "on_activation_per_token": "FP8_DYNAMIC",
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
        saver.json_writer.write.assert_called_once_with("model.layers.0.self_attn.quant_type", "FP8_DYNAMIC")

    def test_MindIEFormatSaver_raises_schema_validate_error_when_activation_scheme_invalid(self, saver):
        module = SimpleNamespace(x_q_scheme=SimpleNamespace(dtype=QDType.INT8, scope=QScope.PER_CHANNEL))
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
