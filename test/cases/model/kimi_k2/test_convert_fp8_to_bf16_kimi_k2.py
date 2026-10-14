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
"""

import json
import os
from typing import Dict
from unittest.mock import patch

import pytest
import torch
from torch import nn
from safetensors.torch import save_file

from msmodelslim.model.kimi_k2 import convert_fp8_to_bf16 as target


@pytest.fixture(autouse=True)
def cleanup_inv_weight_map_cache():
    target.get_inv_weight_map.cache_clear()
    yield
    target.get_inv_weight_map.cache_clear()


def test_weight_dequant_given_full_block_weight_when_called_then_return_bfloat16_tensor():
    weight = torch.randn(128, 128, dtype=torch.float32)
    scale = torch.full((1, 1), 0.5, dtype=torch.float32)

    out = target.weight_dequant(weight.clone(), scale)

    assert out.dtype == torch.bfloat16
    assert out.shape == (128, 128)
    expected = (weight * scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[:128, :128]).to(
        torch.bfloat16
    )
    assert torch.allclose(out.float(), expected.float(), atol=0, rtol=0)


def test_weight_dequant_given_non_block_aligned_weight_when_called_then_trim_scale_to_weight():
    weight = torch.randn(120, 130, dtype=torch.float32)
    scale = torch.full((1, 2), 0.25, dtype=torch.float32)

    out = target.weight_dequant(weight.clone(), scale)

    assert out.dtype == torch.bfloat16
    assert out.shape == (120, 130)
    expected = (weight * scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[:120, :130]).to(
        torch.bfloat16
    )
    assert torch.allclose(out.float(), expected.float(), atol=0, rtol=0)


def test_weight_dequant_given_custom_block_size_when_called_then_apply_block_size():
    weight = torch.randn(64, 64, dtype=torch.float32)
    scale = torch.full((2, 2), 2.0, dtype=torch.float32)

    out = target.weight_dequant(weight.clone(), scale, block_size=32)

    assert out.dtype == torch.bfloat16
    assert out.shape == (64, 64)
    expected = (weight * scale.repeat_interleave(32, dim=0).repeat_interleave(32, dim=1)[:64, :64]).to(torch.bfloat16)
    assert torch.allclose(out.float(), expected.float(), atol=0, rtol=0)


def test_weight_dequant_given_ones_weight_ones_scale_when_called_then_return_ones_bfloat16():
    weight = torch.ones((128, 128), dtype=torch.float32)
    scale = torch.ones((1, 1), dtype=torch.float32)

    out = target.weight_dequant(weight, scale)

    assert out.dtype == torch.bfloat16
    assert torch.allclose(out.float(), torch.ones_like(out).float(), atol=0, rtol=0)


def test_weight_dequant_given_does_not_modify_input_weight_in_place_shape():
    weight = torch.randn(128, 128, dtype=torch.float32)
    scale = torch.full((1, 1), 0.5, dtype=torch.float32)
    original_shape = weight.shape

    target.weight_dequant(weight, scale)

    assert weight.shape == original_shape


def test_get_inv_weight_map_given_index_json_when_called_then_filter_weight_scale_inv_entries(tmp_path):
    index_content = {
        "weight_map": {
            "model.layer0.weight": "model-00001.safetensors",
            "model.layer0.weight_scale_inv": "model-00001.safetensors",
            "model.layer1.weight": "model-00002.safetensors",
            "model.layer1.weight_scale_inv": "model-00002.safetensors",
        }
    }
    index_path = os.path.join(str(tmp_path), "model.safetensors.index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_content, f)

    with patch.object(target, "json_safe_load", return_value=index_content):
        result = target.get_inv_weight_map(str(tmp_path))

    assert result == {
        "model.layer0": "model-00001.safetensors",
        "model.layer1": "model-00002.safetensors",
    }


def test_get_inv_weight_map_given_no_scale_inv_entries_when_called_then_return_empty_map(tmp_path):
    index_content = {
        "weight_map": {
            "model.layer0.weight": "model-00001.safetensors",
        }
    }
    with patch.object(target, "json_safe_load", return_value=index_content):
        result = target.get_inv_weight_map(str(tmp_path))

    assert result == {}


def test_get_inv_weight_map_given_repeated_calls_when_called_then_use_cached_value(tmp_path):
    index_content = {
        "weight_map": {
            "model.layer0.weight_scale_inv": "model-00001.safetensors",
        }
    }
    with patch.object(target, "json_safe_load", return_value=index_content) as mock_load:
        first = target.get_inv_weight_map(str(tmp_path))
        second = target.get_inv_weight_map(str(tmp_path))

    assert first == second
    assert mock_load.call_count == 1


def test_get_inv_tensor_given_valid_weight_map_when_called_then_return_scale_tensor(tmp_path):
    safetensor_file = "model-00001.safetensors"
    safetensor_path = os.path.join(str(tmp_path), safetensor_file)

    scale_tensor = torch.randn(2, 2)
    tensors_dict = {"model.layer0.weight_scale_inv": scale_tensor}
    save_file(tensors_dict, safetensor_path)

    weight_map = {"model.layer0": safetensor_file}

    with patch.object(target, "get_valid_read_path", return_value=safetensor_path):
        result = target.get_inv_tensor("model.layer0", str(tmp_path), weight_map)

    assert result.shape == scale_tensor.shape
    assert torch.allclose(result, scale_tensor)


def test_get_inv_tensor_given_tensor_name_uses_suffix_when_called_then_lookup_with_suffix(tmp_path):
    safetensor_file = "model-00001.safetensors"
    safetensor_path = os.path.join(str(tmp_path), safetensor_file)

    scale_tensor = torch.ones((1, 1))
    tensors_dict = {"linear.weight_scale_inv": scale_tensor}
    save_file(tensors_dict, safetensor_path)
    weight_map = {"linear": safetensor_file}

    captured = {}

    class _FakeSafeOpen:
        def __init__(self, path, framework, device):
            captured["path"] = path

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get_tensor(self, name):
            captured["name"] = name
            return scale_tensor

    with (
        patch.object(target, "get_valid_read_path", return_value=safetensor_path),
        patch.object(target, "safe_open", _FakeSafeOpen),
    ):
        target.get_inv_tensor("linear", str(tmp_path), weight_map)

    assert captured["name"] == "linear.weight_scale_inv"


def test_convert_module_fp8_to_bf16_given_weight_map_when_called_then_dequantize_matching_modules():
    class Container(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(128, 128, bias=False)

    container = Container()
    container.linear.weight.data = torch.randn(128, 128, dtype=torch.bfloat16)
    original_weight = container.linear.weight.data.clone()
    scale = torch.full((1, 1), 0.5, dtype=torch.float32)

    weight_map: Dict[str, str] = {"linear": "chunk-00001.safetensors"}

    def fake_get_inv_tensor(tensor_name, fp8_path, wm):
        return scale

    with patch.object(target, "get_inv_tensor", side_effect=fake_get_inv_tensor):
        target.convert_module_fp8_to_bf16("", container, "IGNORED", weight_map=weight_map)

    expected = target.weight_dequant(original_weight, scale)
    assert torch.allclose(container.linear.weight.data.float(), expected.float(), atol=0, rtol=0)
    assert container.linear.weight.dtype == torch.bfloat16


def test_convert_module_fp8_to_bf16_given_unmatched_submodule_when_called_then_skip_module():
    class Container(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(128, 128, bias=False)
            self.linear_other = nn.Linear(64, 64, bias=False)

    container = Container()
    original_weight = container.linear.weight.data.clone()
    original_weight_other = container.linear_other.weight.data.clone()
    weight_map = {"linear": "chunk-00001.safetensors"}

    with patch.object(target, "get_inv_tensor", return_value=torch.full((1, 1), 0.5)):
        target.convert_module_fp8_to_bf16("", container, "IGNORED", weight_map=weight_map)

    assert not torch.equal(container.linear.weight.data, original_weight)
    assert torch.equal(container.linear_other.weight.data, original_weight_other)


def test_convert_module_fp8_to_bf16_given_prefix_when_called_then_use_prefixed_names():
    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = nn.Linear(128, 128, bias=False)

    outer = Outer()
    weight_map = {"model.layers.0.inner": "chunk.safetensors"}
    scale = torch.full((1, 1), 0.25, dtype=torch.float32)

    with patch.object(target, "get_inv_tensor", return_value=scale) as mock_get:
        target.convert_module_fp8_to_bf16("model.layers.0", outer, "IGNORED", weight_map=weight_map)

    mock_get.assert_called_once()
    args, _ = mock_get.call_args
    assert args[0] == "model.layers.0.inner"


def test_auto_convert_module_fp8_to_bf16_given_empty_weight_map_when_called_then_skip_conversion():
    called = {"flag": False}

    def fake_convert(*args, **kwargs):
        called["flag"] = True

    with (
        patch.object(target, "get_inv_weight_map", return_value={}),
        patch.object(target, "convert_module_fp8_to_bf16", side_effect=fake_convert),
    ):
        target.auto_convert_module_fp8_to_bf16("root", nn.Linear(2, 2, bias=False), "/tmp/model")

    assert called["flag"] is False


def test_auto_convert_module_fp8_to_bf16_given_weight_map_with_match_when_called_then_call_convert():
    captured = {}

    def fake_convert(name, module, model_path, weight_map):
        captured["name"] = name
        captured["module"] = module
        captured["model_path"] = model_path
        captured["weight_map"] = weight_map

    model = nn.ModuleDict({"linear": nn.Linear(1, 1, bias=False)})
    weight_map = {"linear": "f.safetensors", "linear_other": "f.safetensors"}

    with (
        patch.object(target, "get_inv_weight_map", return_value=weight_map),
        patch.object(target, "convert_module_fp8_to_bf16", side_effect=fake_convert),
    ):
        target.auto_convert_module_fp8_to_bf16("", model, "/tmp/model")

    assert captured["name"] == ""
    assert captured["module"] is model
    assert captured["model_path"] == "/tmp/model"
    assert captured["weight_map"] == {"linear": "f.safetensors"}


def test_auto_convert_module_fp8_to_bf16_given_keyerror_in_convert_when_called_then_log_warning():
    class FakeLogger:
        def __init__(self):
            self.warnings = []

        def warning(self, msg):
            self.warnings.append(msg)

    fake_logger = FakeLogger()

    def fake_convert(*args, **kwargs):
        raise KeyError("missing")

    with (
        patch.object(target, "get_inv_weight_map", return_value={"linear": "f.safetensors"}),
        patch.object(target, "convert_module_fp8_to_bf16", side_effect=fake_convert),
        patch.object(target, "get_logger", return_value=fake_logger),
    ):
        target.auto_convert_module_fp8_to_bf16(
            "m", nn.ModuleDict({"linear": nn.Linear(1, 1, bias=False)}), "/tmp/model"
        )

    assert len(fake_logger.warnings) == 2
    assert "Safetensors files not match index.json" in fake_logger.warnings[0]
    assert "Skip fp8 to bf16" in fake_logger.warnings[1]
