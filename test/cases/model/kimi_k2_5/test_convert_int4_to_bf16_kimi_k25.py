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
# pylint: disable=attribute-defined-outside-init

import pytest
import torch
from torch import nn

from msmodelslim.model.kimi_k2_5 import convert_int4_to_bf16 as target
from msmodelslim.utils.exception import SchemaValidateError


@pytest.fixture(autouse=True)
def test_cleanup_given_cache_when_each_case_then_clear():
    target.get_full_weight_map.cache_clear()
    target.get_int4_weight_map.cache_clear()
    target._load_int4_file_state_dict.cache_clear()
    yield


class CompressedLinear(nn.Module):
    def __init__(self, in_features=2, out_features=2, bias=None, weight_shape=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = torch.zeros(weight_shape or (out_features, in_features), dtype=torch.float32)


class _Plain(nn.Module):
    def __init__(self, in_features=8, out_features=2):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((out_features, in_features), dtype=torch.float32))


def _int4_state(prefix, rows=2, cols=8, packed_dtype=torch.int32, packed_val=0):
    return {
        f"{prefix}.weight_packed": torch.full((rows, 1), packed_val, dtype=packed_dtype),
        f"{prefix}.weight_scale": torch.ones((rows, 1), dtype=torch.float32),
        f"{prefix}.weight_shape": torch.tensor([rows, cols]),
    }


def test_get_full_weight_map_given_model_path_when_json_loaded_then_return_weight_map(monkeypatch, tmp_path):
    monkeypatch.setattr(target, "json_safe_load", lambda p: {"weight_map": {"a.weight": "f.safetensors"}})
    assert target.get_full_weight_map(str(tmp_path)) == {"a.weight": "f.safetensors"}


@pytest.mark.parametrize(
    "weight_map,name,expected_none",
    [({}, "x.weight", True), ({"x.weight": "a.safetensors"}, "x.weight", False)],
)
def test_load_tensor_by_full_name_given_name_when_called_then_return_expected(
    monkeypatch, weight_map, name, expected_none
):
    monkeypatch.setattr(target, "get_full_weight_map", lambda _: weight_map)
    monkeypatch.setattr(target, "get_valid_read_path", lambda p, **kwargs: p)
    monkeypatch.setattr(target, "load_file", lambda p, device="cpu": {"x.weight": torch.tensor([1])})
    out = target.load_tensor_by_full_name("/tmp/m", name)
    assert (out is None) if expected_none else torch.equal(out, torch.tensor([1]))


@pytest.mark.parametrize("func", [target._unpack_from_int32_torch, target._unpack_from_int32_numpy])
@pytest.mark.parametrize(
    "tensor_dtype,num_bits,need_skip",
    [(torch.int64, 4, False), (torch.int32, 16, False), (torch.int32, 4, True)],
)
def test_unpack_from_int32_given_invalid_or_skip_when_called_then_raise_or_skip(
    func, tensor_dtype, num_bits, need_skip
):
    if func is target._unpack_from_int32_numpy and target.np is None:
        pytest.skip("numpy unavailable")
    if need_skip:
        return
    with pytest.raises(SchemaValidateError):
        func(torch.tensor([[1]], dtype=tensor_dtype), num_bits, torch.Size([1, 8]), 1)


@pytest.mark.parametrize(
    "func,packed_dim,shape",
    [
        (target._unpack_from_int32_torch, 1, torch.Size([1, 8])),
        (target._unpack_from_int32_torch, 0, torch.Size([8, 1])),
        (target._unpack_from_int32_numpy, 0, torch.Size([8, 1])),
    ],
)
def test_unpack_from_int32_given_valid_input_when_called_then_shape_expected(func, packed_dim, shape):
    if func is target._unpack_from_int32_numpy and target.np is None:
        pytest.skip("numpy unavailable")
    out = func(torch.tensor([[0x76543210]], dtype=torch.int32), 4, shape, packed_dim)
    assert out.shape == tuple(shape)
    if func is target._unpack_from_int32_torch and packed_dim == 1:
        assert out.dtype == torch.int8


def test_unpack_from_int32_numpy_given_np_none_when_called_then_fallback(monkeypatch):
    monkeypatch.setattr(target, "np", None)
    out = target._unpack_from_int32_numpy(torch.tensor([[0x76543210]], dtype=torch.int32), 4, torch.Size([1, 8]), 1)
    assert out.shape == (1, 8)


def test_unpack_from_int32_numpy_given_packed_dim_one_when_called_then_return_expected_shape():
    if target.np is None:
        pytest.skip("numpy unavailable")
    out = target._unpack_from_int32_numpy(torch.tensor([[0x76543210]], dtype=torch.int32), 4, torch.Size([1, 8]), 1)
    assert out.shape == (1, 8)
    assert out.dtype == torch.int8


def test_unpack_from_int32_given_input_when_called_then_delegate_numpy(monkeypatch):
    called = {"ok": False}

    def fake(*_args, **_kwargs):
        called["ok"] = True
        return torch.zeros((1, 1), dtype=torch.int8)

    monkeypatch.setattr(target, "_unpack_from_int32_numpy", fake)
    out = target.unpack_from_int32(torch.tensor([[1]], dtype=torch.int32), 4, torch.Size([1, 1]), 1)
    assert called["ok"] and out.dtype == torch.int8


@pytest.mark.parametrize(
    "weight,scale",
    [
        (torch.ones((2, 5), dtype=torch.int8), torch.ones((2, 2), dtype=torch.float32)),
        (torch.ones((2, 4), dtype=torch.int8), torch.ones((3, 2), dtype=torch.float32)),
    ],
)
def test_weight_dequant_given_invalid_inputs_when_called_then_raise_value_error(weight, scale):
    with pytest.raises(SchemaValidateError):
        target.weight_dequant(weight, scale)


def test_weight_dequant_given_valid_inputs_when_called_then_return_tensor():
    out = target.weight_dequant(
        torch.tensor([[1, 2, 3, 4]], dtype=torch.int8), torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    )
    assert out.shape == (1, 4)


def test_get_int4_weight_map_given_weight_map_when_called_then_filter(monkeypatch, tmp_path):
    monkeypatch.setattr(target, "json_safe_load", lambda p: {"weight_map": {"a.weight_packed": "f1", "b": "f2"}})
    assert target.get_int4_weight_map(str(tmp_path)) == {"a": "f1"}


def test_load_int4_file_state_dict_given_model_path_when_called_then_return_state(monkeypatch):
    monkeypatch.setattr(target, "get_valid_read_path", lambda p, *args, **kwargs: p)
    monkeypatch.setattr(target, "load_file", lambda p, device="cpu": {"ok": torch.tensor(1)})
    assert "ok" in target._load_int4_file_state_dict("/tmp/model", "x.safetensors")


def test_replace_compressed_linear_with_bf16_given_root_compressed_when_loaded_then_return_linear(monkeypatch):
    monkeypatch.setattr(
        target,
        "load_tensor_by_full_name",
        lambda p, n: torch.ones((2, 2), dtype=torch.float32)
        if n.endswith(".weight")
        else torch.ones(2, dtype=torch.float32),
    )
    out = target.replace_compressed_linear_with_bf16(CompressedLinear(), "root", "/tmp/model")
    assert isinstance(out, nn.Linear) and out.weight.dtype == torch.bfloat16


def test_replace_compressed_linear_with_bf16_given_root_weight_shape_mismatch_when_called_then_return_original(
    monkeypatch,
):
    root = CompressedLinear(weight_shape=(1, 2), bias=None)
    monkeypatch.setattr(target, "load_tensor_by_full_name", lambda *args, **kwargs: None)
    assert target.replace_compressed_linear_with_bf16(root, "root", "/tmp/model") is root


def test_replace_compressed_linear_with_bf16_given_packed_weight_when_called_then_convert_success(monkeypatch):
    root = CompressedLinear(in_features=8, out_features=2, bias=None, weight_shape=(1, 8))
    root.weight_packed = torch.zeros((2, 1), dtype=torch.int32)
    root.weight_scale = torch.ones((2, 1), dtype=torch.float32)
    root.weight_shape = torch.tensor([2, 8])
    monkeypatch.setattr(target, "load_tensor_by_full_name", lambda *args, **kwargs: None)
    monkeypatch.setattr(target, "unpack_from_int32", lambda *args, **kwargs: torch.ones((2, 8), dtype=torch.int8))
    monkeypatch.setattr(target, "weight_dequant", lambda *args, **kwargs: torch.ones((2, 8), dtype=torch.float32))
    out = target.replace_compressed_linear_with_bf16(root, "root", "/tmp/model")
    assert isinstance(out, nn.Linear) and out.bias is None


def test_replace_compressed_linear_with_bf16_given_bias_not_loaded_when_called_then_use_module_bias(monkeypatch):
    root = CompressedLinear(in_features=2, out_features=2, bias=torch.tensor([0.5, -0.25], dtype=torch.float32))

    def fake_load(_model_path, full_name):
        if full_name.endswith(".weight"):
            return torch.ones((2, 2), dtype=torch.float32)
        return None

    monkeypatch.setattr(target, "load_tensor_by_full_name", fake_load)
    out = target.replace_compressed_linear_with_bf16(root, "root", "/tmp/model")
    assert isinstance(out, nn.Linear)
    assert out.bias is not None
    assert torch.allclose(out.bias.to(torch.float32), torch.tensor([0.5, -0.25]), rtol=1e-6, atol=1e-6)


def test_replace_compressed_linear_with_bf16_given_child_not_convertible_when_called_then_keep_original(monkeypatch):
    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = CompressedLinear(weight_shape=(1, 2), bias=None)

    monkeypatch.setattr(target, "load_tensor_by_full_name", lambda *args, **kwargs: None)
    root = Root()
    before = root.sub
    out = target.replace_compressed_linear_with_bf16(root, "root", "/tmp/model")
    assert out.sub is before


def test_replace_compressed_linear_with_bf16_given_loaded_weight_bias_when_forward_then_matches_reference(monkeypatch):
    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = CompressedLinear(in_features=3, out_features=2, bias=torch.zeros(2, dtype=torch.float32))

    weight = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]], dtype=torch.float32)
    bias = torch.tensor([0.25, -0.5], dtype=torch.float32)

    def fake_load(_model_path, full_name):
        if full_name.endswith(".weight"):
            return weight
        if full_name.endswith(".bias"):
            return bias
        return None

    monkeypatch.setattr(target, "load_tensor_by_full_name", fake_load)
    out = target.replace_compressed_linear_with_bf16(Root(), "root", "/tmp/model")

    x = torch.tensor([[0.1, -0.2, 0.3], [1.0, 0.0, -1.0]], dtype=torch.float32)
    y = out.sub(x.to(torch.bfloat16)).to(torch.float32)
    expected = x @ weight.t() + bias
    assert isinstance(out.sub, nn.Linear)
    assert torch.allclose(y, expected, rtol=2e-2, atol=2e-2)


def test_auto_convert_module_int4_to_bf16_given_empty_weight_map_when_called_then_skip(monkeypatch):
    called = {"ok": False}
    monkeypatch.setattr(target, "get_int4_weight_map", lambda _: {})
    monkeypatch.setattr(target, "convert_module_int4_to_bf16", lambda *args, **kwargs: called.__setitem__("ok", True))
    target.auto_convert_module_int4_to_bf16("m", nn.Linear(2, 2), "/tmp/model")
    assert called["ok"] is False


def test_auto_convert_module_int4_to_bf16_given_key_error_when_convert_then_log(monkeypatch):
    class L:
        def __init__(self):
            self.w = []

        def warning(self, msg):
            self.w.append(msg)

    logger = L()
    monkeypatch.setattr(target, "get_int4_weight_map", lambda _: {"m.0": "f"})
    monkeypatch.setattr(
        target, "convert_module_int4_to_bf16", lambda *args, **kwargs: (_ for _ in ()).throw(KeyError("x"))
    )
    monkeypatch.setattr(target, "get_logger", lambda: logger)
    target.auto_convert_module_int4_to_bf16("m", nn.Sequential(nn.Linear(2, 2)), "/tmp/model")
    assert len(logger.w) == 2


def test_auto_convert_module_int4_to_bf16_given_partial_map_when_convert_then_only_pass_existing_submodules(
    monkeypatch,
):
    captured = {}

    def fake_convert(name, module, model_path, weight_map):
        captured.update(name=name, module=module, model_path=model_path, weight_map=weight_map)

    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    monkeypatch.setattr(target, "get_int4_weight_map", lambda _: {"m.0": "f0", "m.9": "f9"})
    monkeypatch.setattr(target, "convert_module_int4_to_bf16", fake_convert)
    target.auto_convert_module_int4_to_bf16("m", model, "/tmp/model")

    assert captured["name"] == "m"
    assert captured["module"] is model
    assert captured["model_path"] == "/tmp/model"
    assert captured["weight_map"] == {"m.0": "f0"}


def test_convert_module_int4_to_bf16_given_compressed_linear_when_called_then_replace(monkeypatch):
    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = CompressedLinear(in_features=8, out_features=2, bias=torch.zeros(2))

    root = Root()
    monkeypatch.setattr(
        target, "_load_int4_file_state_dict", lambda *args, **kwargs: _int4_state("root.sub", rows=2, cols=8)
    )
    monkeypatch.setattr(target, "unpack_from_int32", lambda *args, **kwargs: torch.ones((2, 8), dtype=torch.int8))
    monkeypatch.setattr(target, "weight_dequant", lambda *args, **kwargs: torch.ones((2, 8), dtype=torch.float32))
    target.convert_module_int4_to_bf16("root", root, "/tmp/model", {"root.sub": "f"})
    assert isinstance(root.sub, nn.Linear)


def test_convert_module_int4_to_bf16_given_regular_module_when_called_then_assign_weight(monkeypatch):
    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = _Plain(in_features=8, out_features=2)

    root = Root()
    monkeypatch.setattr(
        target, "_load_int4_file_state_dict", lambda *args, **kwargs: _int4_state("root.sub", rows=2, cols=8)
    )
    monkeypatch.setattr(target, "unpack_from_int32", lambda *args, **kwargs: torch.ones((2, 8), dtype=torch.int8))
    monkeypatch.setattr(target, "weight_dequant", lambda *args, **kwargs: torch.full((2, 8), 3.0, dtype=torch.float32))
    target.convert_module_int4_to_bf16("root", root, "/tmp/model", {"root.sub": "f"})
    assert torch.allclose(root.sub.weight, torch.full((2, 8), 3.0))


def test_convert_module_int4_to_bf16_given_loader_has_cache_clear_when_done_then_call_cache_clear(monkeypatch):
    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = _Plain(in_features=8, out_features=2)

    class Loader:
        def __init__(self):
            self.cache_cleared = False

        def __call__(self, *args, **kwargs):
            return _int4_state("root.sub", rows=2, cols=8)

        def cache_clear(self):
            self.cache_cleared = True

    root, loader = Root(), Loader()
    monkeypatch.setattr(target, "_load_int4_file_state_dict", loader)
    monkeypatch.setattr(target, "unpack_from_int32", lambda *args, **kwargs: torch.ones((2, 8), dtype=torch.int8))
    monkeypatch.setattr(target, "weight_dequant", lambda *args, **kwargs: torch.ones((2, 8), dtype=torch.float32))

    target.convert_module_int4_to_bf16("root", root, "/tmp/model", {"root.sub": "f"})
    assert loader.cache_cleared is True


def test_convert_module_int4_to_bf16_given_two_submodules_same_file_when_called_then_outputs_match_expected(
    monkeypatch,
):
    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = CompressedLinear(
                in_features=4, out_features=2, bias=torch.tensor([0.25, -0.75], dtype=torch.float32)
            )
            self.b = _Plain(in_features=4, out_features=2)

    root = Root()
    state = {
        **_int4_state("root.a", rows=2, cols=4, packed_dtype=torch.int16, packed_val=7),
        **_int4_state("root.b", rows=2, cols=4, packed_dtype=torch.int32, packed_val=1),
    }
    call_count = {"loader": 0, "unpack": 0}

    def fake_loader(*_args, **_kwargs):
        call_count["loader"] += 1
        return state

    def fake_unpack(packed, num_bits, shape, packed_dim):
        call_count["unpack"] += 1
        assert num_bits == 4 and packed_dim == 1 and packed.dtype is torch.int32
        rows, cols = int(shape[0]), int(shape[1])
        if rows == 2 and cols == 4 and packed[0, 0].item() == 7:
            return torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.int8)
        return torch.tensor([[2, 0, -2, 1], [1, -1, 0, 2]], dtype=torch.int8)

    monkeypatch.setattr(target, "_load_int4_file_state_dict", fake_loader)
    monkeypatch.setattr(target, "unpack_from_int32", fake_unpack)
    monkeypatch.setattr(
        target,
        "weight_dequant",
        lambda weight_int8, scale: weight_int8.to(torch.float32) * scale.repeat_interleave(weight_int8.shape[1], dim=1),
    )

    target.convert_module_int4_to_bf16("root", root, "/tmp/model", {"root.a": "f0", "root.b": "f0"})

    assert call_count["loader"] == 1 and call_count["unpack"] == 2
    assert isinstance(root.a, nn.Linear)

    x = torch.tensor([[1.0, -1.0, 0.5, 2.0]], dtype=torch.float32)
    a_weight = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], dtype=torch.float32)
    y_a = root.a(x).to(torch.float32)
    assert torch.allclose(y_a, x @ a_weight.t() + torch.tensor([0.25, -0.75]), rtol=1e-4, atol=1e-4)

    expected_b = torch.tensor([[2.0, 0.0, -2.0, 1.0], [1.0, -1.0, 0.0, 2.0]], dtype=torch.float32)
    assert torch.allclose(root.b.weight.detach(), expected_b, rtol=1e-6, atol=1e-6)
