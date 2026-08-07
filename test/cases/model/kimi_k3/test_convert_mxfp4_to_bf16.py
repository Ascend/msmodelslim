# -*- coding: UTF-8 -*-
"""Unit tests for msmodelslim.model.kimi_k3.convert_mxfp4_to_bf16."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from msmodelslim.model.kimi_k3 import convert_mxfp4_to_bf16 as target
from msmodelslim.utils.exception import SchemaValidateError


@pytest.fixture(autouse=True)
def _clear_lru_caches():
    target.get_full_weight_map.cache_clear()
    target.get_mxfp4_weight_map.cache_clear()
    target._load_shard_state_dict.cache_clear()
    yield
    target.get_full_weight_map.cache_clear()
    target.get_mxfp4_weight_map.cache_clear()
    target._load_shard_state_dict.cache_clear()


class CompressedLinear(nn.Module):
    def __init__(
        self,
        in_features=32,
        out_features=2,
        bias=None,
        weight_shape=None,
        weight_packed=None,
        weight_scale=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        # Placeholder dense weight (overwritten / unused when mxfp4 buffers present).
        shape = weight_shape or (out_features, in_features)
        self.register_buffer("weight", torch.zeros(shape, dtype=torch.float32))
        if weight_packed is not None:
            self.register_buffer("weight_packed", weight_packed)
        if weight_scale is not None:
            self.register_buffer("weight_scale", weight_scale)


class Mxfp4BufferModule(nn.Module):
    """Plain module with mxfp4 buffers (not named CompressedLinear)."""

    def __init__(self, packed: torch.Tensor, scale: torch.Tensor, bias=None):
        super().__init__()
        self.register_buffer("weight_packed", packed)
        self.register_buffer("weight_scale", scale)
        self.bias = bias
        n, k_half = packed.shape
        self.out_features = n
        self.in_features = k_half * 2


class _Plain(nn.Module):
    def __init__(self, in_features=32, out_features=2):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((out_features, in_features), dtype=torch.float32))


def _mxfp4_tensors(n=2, k=32, scale_val=127):
    """Build packed uint8 [N, K/2] and scale uint8 [N, K/32]."""
    packed = torch.zeros((n, k // 2), dtype=torch.uint8)
    # low nibble=2 → +1.0, high nibble=2 → +1.0
    packed[:] = 0x22
    scale = torch.full((n, k // 32), scale_val, dtype=torch.uint8)
    return packed, scale


def _mxfp4_state(prefix: str, n=2, k=32, scale_val=127):
    packed, scale = _mxfp4_tensors(n=n, k=k, scale_val=scale_val)
    return {
        f"{prefix}.weight_packed": packed,
        f"{prefix}.weight_scale": scale,
    }


# ---------------------------------------------------------------------------
# Low-level dequant helpers
# ---------------------------------------------------------------------------


def test_unpack_fp4_from_uint8_given_valid_packed_when_called_then_shape_and_values():
    packed = torch.tensor([[0x22]], dtype=torch.uint8)  # low=2→1.0, high=2→1.0
    out = target.unpack_fp4_from_uint8(packed)
    assert out.shape == (1, 2)
    assert torch.allclose(out, torch.tensor([[1.0, 1.0]]))


def test_unpack_fp4_from_uint8_given_non_uint8_when_called_then_cast():
    packed = torch.tensor([[0x0A]], dtype=torch.int32)  # low=10→-1.0 (sign+abs 2), high=0→0.0
    out = target.unpack_fp4_from_uint8(packed)
    assert out.dtype == torch.float32
    assert out.shape == (1, 2)


def test_unpack_fp4_from_uint8_given_empty_last_dim_when_called_then_raise():
    with pytest.raises(SchemaValidateError, match="Empty packed last dim"):
        target.unpack_fp4_from_uint8(torch.zeros((2, 0), dtype=torch.uint8))


def test_e8m0_uint8_to_scale_given_bias_127_when_called_then_one():
    scale = target.e8m0_uint8_to_scale(torch.tensor([[127]], dtype=torch.uint8))
    assert torch.allclose(scale, torch.tensor([[1.0]]))


def test_e8m0_uint8_to_scale_given_non_uint8_when_called_then_cast():
    scale = target.e8m0_uint8_to_scale(torch.tensor([[128]], dtype=torch.int32))
    assert torch.allclose(scale, torch.tensor([[2.0]]))


def test_dequant_mxfp4_ct_given_n2_k32_scale127_when_called_then_shape_and_scale_one():
    packed, scale = _mxfp4_tensors(n=2, k=32, scale_val=127)
    assert packed.shape == (2, 16)
    assert scale.shape == (2, 1)
    out = target.dequant_mxfp4_ct(packed, scale)
    assert out.shape == (2, 32)
    # values are 1.0 * scale 1.0
    assert torch.allclose(out, torch.ones((2, 32)))


def test_dequant_mxfp4_ct_given_wrong_ndim_when_called_then_raise():
    with pytest.raises(SchemaValidateError, match="Expected 2D weight_packed"):
        target.dequant_mxfp4_ct(torch.zeros((2, 16, 1), dtype=torch.uint8), torch.zeros((2, 1), dtype=torch.uint8))


def test_dequant_mxfp4_ct_given_scale_not_2d_after_squeeze_when_called_then_raise():
    packed = torch.zeros((2, 16), dtype=torch.uint8)
    scale = torch.zeros((2,), dtype=torch.uint8)  # 1D after no squeeze to 2D
    with pytest.raises(SchemaValidateError, match="Expected 2D weight_scale"):
        target.dequant_mxfp4_ct(packed, scale)


def test_dequant_mxfp4_ct_given_row_mismatch_when_called_then_raise():
    packed = torch.zeros((2, 16), dtype=torch.uint8)
    scale = torch.zeros((3, 1), dtype=torch.uint8)
    with pytest.raises(SchemaValidateError, match="Mismatch in scale rows"):
        target.dequant_mxfp4_ct(packed, scale)


def test_dequant_mxfp4_ct_given_k_mismatch_when_called_then_raise():
    packed = torch.zeros((2, 16), dtype=torch.uint8)  # K=32
    scale = torch.zeros((2, 2), dtype=torch.uint8)  # 2*32=64 != 32
    with pytest.raises(SchemaValidateError, match="is not n_blocks"):
        target.dequant_mxfp4_ct(packed, scale)


def test_dequant_mxfp4_ct_given_scale_extra_trailing_ones_when_called_then_squeeze():
    packed, scale = _mxfp4_tensors(n=2, k=32)
    scale4d = scale.unsqueeze(-1).unsqueeze(-1)  # [2,1,1,1]
    out = target.dequant_mxfp4_ct(packed, scale4d)
    assert out.shape == (2, 32)


def test_dequant_mxfp4_to_bf16_given_non_uint8_when_called_then_cast_and_bf16():
    packed = torch.zeros((2, 16), dtype=torch.int32)
    packed[:] = 0x22
    scale = torch.full((2, 1), 127, dtype=torch.int32)
    out = target._dequant_mxfp4_to_bf16(packed, scale)
    assert out.dtype == torch.bfloat16
    assert out.shape == (2, 32)


# ---------------------------------------------------------------------------
# Weight map / shard loading
# ---------------------------------------------------------------------------


def test_get_full_weight_map_given_model_path_when_json_loaded_then_return_weight_map(monkeypatch, tmp_path):
    monkeypatch.setattr(target, "json_safe_load", lambda p: {"weight_map": {"a.weight": "f.safetensors"}})
    assert target.get_full_weight_map(str(tmp_path)) == {"a.weight": "f.safetensors"}


def test_get_mxfp4_weight_map_given_weight_map_when_called_then_filter(monkeypatch, tmp_path):
    monkeypatch.setattr(
        target,
        "json_safe_load",
        lambda p: {"weight_map": {"a.weight_packed": "f1", "a.weight_scale": "f1", "b.weight": "f2"}},
    )
    assert target.get_mxfp4_weight_map(str(tmp_path)) == {"a": "f1"}


def test_load_shard_state_dict_given_model_path_when_called_then_return_state(monkeypatch):
    monkeypatch.setattr(target, "get_valid_read_path", lambda p, *args, **kwargs: p)
    monkeypatch.setattr(target, "load_file", lambda p, device="cpu": {"ok": torch.tensor(1)})
    assert "ok" in target._load_shard_state_dict("/tmp/model", "x.safetensors")


@pytest.mark.parametrize(
    "weight_map,name,expected_none",
    [({}, "x.weight", True), ({"x.weight": "a.safetensors"}, "x.weight", False)],
)
def test_load_tensor_by_full_name_given_name_when_called_then_return_expected(
    monkeypatch, weight_map, name, expected_none
):
    monkeypatch.setattr(target, "get_full_weight_map", lambda _: weight_map)
    monkeypatch.setattr(target, "get_valid_read_path", lambda p, *args, **kwargs: p)
    monkeypatch.setattr(target, "load_file", lambda p, device="cpu": {"x.weight": torch.tensor([1])})
    out = target.load_tensor_by_full_name("/tmp/m", name)
    assert (out is None) if expected_none else torch.equal(out, torch.tensor([1]))


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def test_is_mxfp4_compressed_module_given_buffers_when_called_then_true_false():
    packed, scale = _mxfp4_tensors()
    mod = Mxfp4BufferModule(packed, scale)
    assert target._is_mxfp4_compressed_module(mod) is True
    assert target._is_mxfp4_compressed_module(nn.Linear(2, 2)) is False


def test_module_to_bf16_linear_given_packed_scale_when_called_then_linear(monkeypatch):
    packed, scale = _mxfp4_tensors()
    mod = CompressedLinear(in_features=32, out_features=2)
    monkeypatch.setattr(target, "load_tensor_by_full_name", lambda *a, **k: None)
    out = target._module_to_bf16_linear("root.sub", mod, "/tmp/m", packed=packed, scale=scale)
    assert isinstance(out, nn.Linear)
    assert out.weight.dtype == torch.bfloat16
    assert out.weight.shape == (2, 32)
    assert out.bias is None


def test_module_to_bf16_linear_given_loaded_weight_bias_when_called_then_use_loaded(monkeypatch):
    mod = CompressedLinear(in_features=4, out_features=2, bias=None)
    weight = torch.ones((2, 4), dtype=torch.float32)
    bias = torch.tensor([0.5, -0.25], dtype=torch.float32)

    def fake_load(_p, full_name):
        if full_name.endswith(".weight"):
            return weight
        if full_name.endswith(".bias"):
            return bias
        return None

    monkeypatch.setattr(target, "load_tensor_by_full_name", fake_load)
    out = target._module_to_bf16_linear("root", mod, "/tmp/m")
    assert isinstance(out, nn.Linear)
    assert out.bias is not None
    assert torch.allclose(out.bias.to(torch.float32), bias, rtol=1e-2, atol=1e-2)


def test_module_to_bf16_linear_given_mod_buffers_when_no_loaded_then_dequant(monkeypatch):
    packed, scale = _mxfp4_tensors()
    mod = Mxfp4BufferModule(packed, scale, bias=torch.zeros(2))
    monkeypatch.setattr(target, "load_tensor_by_full_name", lambda *a, **k: None)
    out = target._module_to_bf16_linear("root", mod, "/tmp/m")
    assert isinstance(out, nn.Linear)
    assert out.bias is not None


def test_module_to_bf16_linear_given_plain_linear_when_no_loaded_then_copy_weight(monkeypatch):
    mod = nn.Linear(4, 2, bias=True)
    monkeypatch.setattr(target, "load_tensor_by_full_name", lambda *a, **k: None)
    out = target._module_to_bf16_linear("root", mod, "/tmp/m")
    assert isinstance(out, nn.Linear)
    assert out.weight.dtype == torch.bfloat16


def test_module_to_bf16_linear_given_no_weight_when_called_then_none(monkeypatch):
    mod = nn.Module()
    monkeypatch.setattr(target, "load_tensor_by_full_name", lambda *a, **k: None)
    assert target._module_to_bf16_linear("root", mod, "/tmp/m") is None


def test_module_to_bf16_linear_given_shape_mismatch_when_called_then_raise(monkeypatch):
    packed, scale = _mxfp4_tensors(n=2, k=32)
    mod = CompressedLinear(in_features=16, out_features=2)  # expects (2,16) but dequant is (2,32)
    monkeypatch.setattr(target, "load_tensor_by_full_name", lambda *a, **k: None)
    with pytest.raises(SchemaValidateError, match="dequant shape"):
        target._module_to_bf16_linear("root", mod, "/tmp/m", packed=packed, scale=scale)


# ---------------------------------------------------------------------------
# replace / auto / convert / subtree
# ---------------------------------------------------------------------------


def test_replace_compressed_linear_with_bf16_given_root_compressed_when_called_then_linear(monkeypatch):
    packed, scale = _mxfp4_tensors()
    root = CompressedLinear(in_features=32, out_features=2, weight_packed=packed, weight_scale=scale)
    monkeypatch.setattr(target, "load_tensor_by_full_name", lambda *a, **k: None)
    out = target.replace_compressed_linear_with_bf16(root, "root", "/tmp/m")
    assert isinstance(out, nn.Linear)


def test_replace_compressed_linear_with_bf16_given_root_not_convertible_when_called_then_original(monkeypatch):
    # Named CompressedLinear but dense weight shape mismatches Linear (out, in) → skip convert.
    root = CompressedLinear(in_features=32, out_features=2, weight_shape=(1, 2))
    monkeypatch.setattr(target, "load_tensor_by_full_name", lambda *a, **k: None)
    assert target.replace_compressed_linear_with_bf16(root, "root", "/tmp/m") is root


def test_replace_compressed_linear_with_bf16_given_child_mxfp4_when_called_then_replace(monkeypatch):
    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            packed, scale = _mxfp4_tensors()
            self.sub = Mxfp4BufferModule(packed, scale)

    monkeypatch.setattr(target, "load_tensor_by_full_name", lambda *a, **k: None)
    root = Root()
    out = target.replace_compressed_linear_with_bf16(root, "root", "/tmp/m")
    assert isinstance(out.sub, nn.Linear)


def test_replace_compressed_linear_with_bf16_given_child_not_convertible_when_called_then_keep(monkeypatch):
    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = CompressedLinear(weight_shape=(1, 2))

    monkeypatch.setattr(target, "load_tensor_by_full_name", lambda *a, **k: None)
    root = Root()
    before = root.sub
    out = target.replace_compressed_linear_with_bf16(root, "root", "/tmp/m")
    assert out.sub is before


def test_auto_convert_module_mxfp4_to_bf16_given_empty_weight_map_when_called_then_skip(monkeypatch):
    called = {"ok": False}
    monkeypatch.setattr(target, "get_mxfp4_weight_map", lambda _: {})
    monkeypatch.setattr(target, "convert_module_mxfp4_to_bf16", lambda *a, **k: called.__setitem__("ok", True))
    target.auto_convert_module_mxfp4_to_bf16("m", nn.Linear(2, 2), "/tmp/m")
    assert called["ok"] is False


def test_auto_convert_module_mxfp4_to_bf16_given_key_error_when_building_map_then_warn(monkeypatch):
    class WeirdMap(dict):
        def __contains__(self, key):
            return True

        def __getitem__(self, key):
            raise KeyError(key)

    class L:
        def __init__(self):
            self.w = []

        def warning(self, msg, *args):
            self.w.append(msg)

        def info(self, msg, *args):
            pass

    logger = L()
    monkeypatch.setattr(target, "get_mxfp4_weight_map", lambda _: WeirdMap({"m.0": "f"}))
    monkeypatch.setattr(target, "get_logger", lambda: logger)
    target.auto_convert_module_mxfp4_to_bf16("m", nn.Sequential(nn.Linear(2, 2)), "/tmp/m")
    assert any("skip mxfp4" in str(m) for m in logger.w)


def test_auto_convert_module_mxfp4_to_bf16_given_partial_map_when_convert_then_pass_existing(monkeypatch):
    captured = {}

    def fake_convert(name, module, model_path, weight_map):
        captured.update(name=name, module=module, model_path=model_path, weight_map=weight_map)

    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    monkeypatch.setattr(target, "get_mxfp4_weight_map", lambda _: {"m.0": "f0", "m.9": "f9"})
    monkeypatch.setattr(target, "convert_module_mxfp4_to_bf16", fake_convert)
    target.auto_convert_module_mxfp4_to_bf16("m", model, "/tmp/m")
    assert captured["weight_map"] == {"m.0": "f0"}


def test_auto_convert_module_mxfp4_to_bf16_given_no_matching_submodules_when_called_then_return(monkeypatch):
    called = {"ok": False}
    monkeypatch.setattr(target, "get_mxfp4_weight_map", lambda _: {"other.x": "f"})
    monkeypatch.setattr(target, "convert_module_mxfp4_to_bf16", lambda *a, **k: called.__setitem__("ok", True))
    target.auto_convert_module_mxfp4_to_bf16("m", nn.Sequential(nn.Linear(2, 2)), "/tmp/m")
    assert called["ok"] is False


def test_convert_module_mxfp4_to_bf16_given_compressed_when_called_then_replace(monkeypatch):
    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = CompressedLinear(in_features=32, out_features=2)

    root = Root()
    monkeypatch.setattr(target, "_load_shard_state_dict", lambda *a, **k: _mxfp4_state("root.sub"))
    monkeypatch.setattr(target, "load_tensor_by_full_name", lambda *a, **k: None)
    target.convert_module_mxfp4_to_bf16("root", root, "/tmp/m", {"root.sub": "f"})
    assert isinstance(root.sub, nn.Linear)
    assert root.sub.weight.dtype == torch.bfloat16


def test_convert_module_mxfp4_to_bf16_given_plain_matching_shape_when_called_then_copy_weight(monkeypatch):
    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = _Plain(in_features=32, out_features=2)

    root = Root()
    monkeypatch.setattr(target, "_load_shard_state_dict", lambda *a, **k: _mxfp4_state("root.sub"))
    target.convert_module_mxfp4_to_bf16("root", root, "/tmp/m", {"root.sub": "f"})
    assert torch.allclose(root.sub.weight.detach().float(), torch.ones((2, 32)), rtol=1e-2, atol=1e-2)


def test_convert_module_mxfp4_to_bf16_given_shape_mismatch_when_called_then_replace(monkeypatch):
    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = _Plain(in_features=16, out_features=2)  # wrong K

    root = Root()
    monkeypatch.setattr(target, "_load_shard_state_dict", lambda *a, **k: _mxfp4_state("root.sub"))
    monkeypatch.setattr(target, "load_tensor_by_full_name", lambda *a, **k: None)
    # Shape mismatch path: replace; _module_to_bf16_linear may raise if in/out_features set.
    # _Plain has no in/out_features → replace succeeds.
    target.convert_module_mxfp4_to_bf16("root", root, "/tmp/m", {"root.sub": "f"})
    assert isinstance(root.sub, nn.Linear)
    assert root.sub.weight.shape == (2, 32)


def test_convert_module_mxfp4_to_bf16_given_missing_keys_when_called_then_warn_skip(monkeypatch):
    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = CompressedLinear()

    class L:
        def __init__(self):
            self.w = []

        def warning(self, msg, *args):
            self.w.append((msg, args))

        def info(self, msg, *args):
            pass

    logger = L()
    root = Root()
    monkeypatch.setattr(target, "_load_shard_state_dict", lambda *a, **k: {})
    monkeypatch.setattr(target, "get_logger", lambda: logger)
    target.convert_module_mxfp4_to_bf16("root", root, "/tmp/m", {"root.sub": "f"})
    assert any("Missing mxfp4" in str(m[0]) for m in logger.w)
    assert isinstance(root.sub, CompressedLinear)


def test_convert_module_mxfp4_to_bf16_given_cache_clear_and_npu_when_done_then_call(monkeypatch):
    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = _Plain(in_features=32, out_features=2)

    class Loader:
        def __init__(self):
            self.cache_cleared = False

        def __call__(self, *args, **kwargs):
            return _mxfp4_state("root.sub")

        def cache_clear(self):
            self.cache_cleared = True

    class Npu:
        def __init__(self):
            self.cleared = False

        def empty_cache(self):
            self.cleared = True

    root, loader, npu = Root(), Loader(), Npu()
    monkeypatch.setattr(target, "_load_shard_state_dict", loader)
    monkeypatch.setattr(target, "npu_available", True)
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    target.convert_module_mxfp4_to_bf16("root", root, "/tmp/m", {"root.sub": "f"})
    assert loader.cache_cleared is True
    assert npu.cleared is True


def test_convert_module_mxfp4_to_bf16_given_npu_empty_cache_fail_when_called_then_warn(monkeypatch):
    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = _Plain(in_features=32, out_features=2)

    class Npu:
        def empty_cache(self):
            raise RuntimeError("boom")

    class L:
        def __init__(self):
            self.w = []

        def warning(self, msg, *args):
            self.w.append(msg)

        def info(self, msg, *args):
            pass

    logger = L()
    monkeypatch.setattr(target, "_load_shard_state_dict", lambda *a, **k: _mxfp4_state("root.sub"))
    monkeypatch.setattr(target, "npu_available", True)
    monkeypatch.setattr(torch, "npu", Npu(), raising=False)
    monkeypatch.setattr(target, "get_logger", lambda: logger)
    target.convert_module_mxfp4_to_bf16("root", Root(), "/tmp/m", {"root.sub": "f"})
    assert any("Failed to clear NPU" in str(m) for m in logger.w)


def test_dequant_subtree_mxfp4_to_bf16_given_prefix_when_called_then_replace_path(monkeypatch):
    calls = []

    monkeypatch.setattr(
        target,
        "auto_convert_module_mxfp4_to_bf16",
        lambda prefix, module, model_path: calls.append(("auto", prefix)),
    )
    monkeypatch.setattr(
        target,
        "replace_compressed_linear_with_bf16",
        lambda module, prefix, model_path: calls.append(("replace", prefix)) or module,
    )
    mod = nn.Linear(2, 2)
    out = target.dequant_subtree_mxfp4_to_bf16(mod, "language_model", "/tmp/m")
    assert out is mod
    assert calls == [("auto", "language_model"), ("replace", "language_model")]


def test_dequant_subtree_mxfp4_to_bf16_given_empty_prefix_when_called_then_rebind_tops(monkeypatch):
    replace_prefixes = []

    def fake_replace(module, prefix, model_path):
        replace_prefixes.append(prefix)
        return module

    class Lang(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(2, 2)

    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower = nn.Linear(2, 2)
            self.mm_projector = nn.Linear(2, 2)
            self.language_model = Lang()

    monkeypatch.setattr(target, "auto_convert_module_mxfp4_to_bf16", lambda *a, **k: None)
    monkeypatch.setattr(target, "replace_compressed_linear_with_bf16", fake_replace)
    root = Root()
    target.dequant_subtree_mxfp4_to_bf16(root, "", "/tmp/m")
    assert replace_prefixes == ["vision_tower", "mm_projector", "language_model.lm_head"]


def test_dequant_subtree_mxfp4_to_bf16_given_empty_prefix_missing_attrs_when_called_then_ok(monkeypatch):
    monkeypatch.setattr(target, "auto_convert_module_mxfp4_to_bf16", lambda *a, **k: None)
    monkeypatch.setattr(
        target, "replace_compressed_linear_with_bf16", lambda m, p, mp: (_ for _ in ()).throw(AssertionError())
    )
    # No vision_tower / mm_projector / language_model → no replace calls
    out = target.dequant_subtree_mxfp4_to_bf16(nn.Module(), "", "/tmp/m")
    assert isinstance(out, nn.Module)
