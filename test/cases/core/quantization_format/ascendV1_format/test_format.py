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

msmodelslim/core/quantization_format/ascendV1_format/format.py 的单元测试。
"""

import io
import json
import os
import shutil
import tempfile
from unittest.mock import patch

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from msmodelslim.core import quantization_schema as qir
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.quant_service.modelslim_v1.save.ascendv1 import AscendV1Config, AscendV1Saver
from msmodelslim.core.quantization_format.ascendV1_format.decoder import _parse_fa_quant_type
from msmodelslim.core.quantization_format.ascendV1_format.format import (
    ASCENDV1_DESC_JSON_NAME,
    ASCENDV1_SAFETENSORS_NAME,
    AscendV1Format,
)
from msmodelslim.core.quantization_schema.attention import FakeQuantDynamicCache
from msmodelslim.core.quantization_schema.const import (
    fp8_e4m3_per_head_sym,
    fp8_e4m3_per_token_sym,
    hif8_per_head_sym,
    hif8_per_tensor_sym,
    int8_per_channel_sym,
    int8_per_head_sym,
    int8_per_token_sym,
)
from msmodelslim.core.quantization_schema.int8_activation_static import INT8FakeQuantActivationPerHead
from msmodelslim.core.quantization_schema.qal import QParam
from msmodelslim.core.quantization_schema.w8a8_static import W8A8StaticFakeQuantLinear
from msmodelslim.model.base import BaseModelAdapter
from msmodelslim.core.quantization_algorithm.quant.fa3.interface import FA3QuantAdapterInterface
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError


class _FA3Adapter(FA3QuantAdapterInterface):
    def inject_fa3_placeholders(self, root_name, root_module, should_inject):
        return None


class _TwoLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2)
        self.head = nn.Linear(3, 2)


class _Layers(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_TwoLinear(), _TwoLinear()])


class _Attn(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.fa_q = nn.Identity()
        self.fa_k = nn.Identity()
        self.fa_v = nn.Identity()


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attn()


class _Blocks(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block()])


class _DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(3, 2)


def _f32_bits(value):
    return torch.tensor([value]).view(torch.int32).to(torch.int64)


_FORMAT_TMP_DIRS = []


@pytest.fixture(autouse=True)
def _cleanup_format_tmp_dirs():
    yield
    for tmp_dir in _FORMAT_TMP_DIRS:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    _FORMAT_TMP_DIRS.clear()


def _make_format(description, store, adapter=None):
    """Build AscendV1Format through its public constructor, patching only the store factory."""
    model_path = tempfile.mkdtemp(prefix="ascendv1_fmt_").replace(os.sep, "/")
    _FORMAT_TMP_DIRS.append(model_path)
    with io.open(os.path.join(model_path, ASCENDV1_DESC_JSON_NAME), "w", encoding="utf-8") as f:
        json.dump(description, f)

    with patch(
        "msmodelslim.core.quantization_format.ascendV1_format.tensor_store.AscendV1TensorStore",
        return_value=store,
    ):
        fmt = AscendV1Format(model_path, device="cpu")
    fmt.bind_adapter(adapter)
    return fmt


def _fill_w8a8(store, prefix, weight_shape=(2, 3), deq=2.0, input_scale=1.0):
    store.put_shape(f"{prefix}.weight", weight_shape)
    store.put_shape(f"{prefix}.input_scale", (1,))
    store.put_shape(f"{prefix}.input_offset", (1,))
    store.put_shape(f"{prefix}.deq_scale", (weight_shape[0],))
    store.put(
        f"{prefix}.weight", torch.arange(weight_shape[0] * weight_shape[1], dtype=torch.int8).reshape(weight_shape)
    )
    store.put(f"{prefix}.input_scale", torch.tensor([input_scale]))
    store.put(f"{prefix}.input_offset", torch.tensor([0.0]))
    bits = _f32_bits(deq)
    store.put(f"{prefix}.deq_scale", bits.repeat(weight_shape[0]))
    return store


class TestAscendV1FormatInit:
    """对应 AscendV1Format 构造与 _load_description。"""

    def test_init_raises_when_description_missing(self, tmp_path):
        with pytest.raises(SchemaValidateError):
            AscendV1Format(str(tmp_path))

    def test_init_raises_when_description_not_object(self, tmp_path):
        (tmp_path / ASCENDV1_DESC_JSON_NAME).write_text(json.dumps([1, 2]), encoding="utf-8")

        with pytest.raises(SchemaValidateError):
            AscendV1Format(str(tmp_path))

    def test_init_reads_description_and_store_when_export_valid(self, tmp_path):
        save_file({"w": torch.zeros(2)}, str(tmp_path / ASCENDV1_SAFETENSORS_NAME))
        (tmp_path / ASCENDV1_DESC_JSON_NAME).write_text(json.dumps({"version": "1", "group_size": 0}), encoding="utf-8")

        fmt = AscendV1Format(str(tmp_path), device="cpu")

        assert fmt.model_path == str(tmp_path)
        assert fmt.description["version"] == "1"
        assert fmt.group_size == 0


class TestApplyLinearIrAndHydrate:
    """对应 apply_ir/hydrate 的 Linear + FLOAT 主流程。"""

    def test_apply_ir_replaces_only_described_linear(self, fake_store_cls):
        store = _fill_w8a8(fake_store_cls(), "fc")
        fmt = _make_format({"fc.weight": "W8A8"}, store)
        model = _TwoLinear()

        fmt.apply_ir(model)

        assert isinstance(model.fc, W8A8StaticFakeQuantLinear)
        assert isinstance(model.head, nn.Linear)  # 未描述，保持不变
        assert model.fc.weight.device.type == "meta"

    def test_apply_ir_skips_float_and_unlisted(self, fake_store_cls):
        store = _fill_w8a8(fake_store_cls(), "fc")
        fmt = _make_format({"fc.weight": "FLOAT"}, store)
        model = _TwoLinear()

        fmt.apply_ir(model)

        assert isinstance(model.fc, nn.Linear)
        assert isinstance(model.head, nn.Linear)

    def test_hydrate_loads_w8a8_and_float_weights_when_full_scope(self, fake_store_cls):
        store = _fill_w8a8(fake_store_cls(), "fc", deq=2.0)
        store.put_shape("head.weight", (2, 3))
        store.put("head.weight", torch.full((2, 3), 5.0))
        fmt = _make_format({"fc.weight": "W8A8", "head.weight": "FLOAT"}, store)
        model = _TwoLinear()
        fmt.apply_ir(model)

        fmt.hydrate(model)

        assert isinstance(model.fc, W8A8StaticFakeQuantLinear)
        assert model.fc.weight.device.type == "cpu"
        assert torch.equal(model.fc.weight.data, torch.arange(6, dtype=torch.int8).reshape(2, 3))
        assert model.fc.weight_scale.tolist() == [2.0, 2.0]  # deq 2.0 / input_scale 1.0
        assert model.fc.bias is None
        assert torch.equal(model.head.weight.data, torch.full((2, 3), 5.0))

    def test_hydrate_respects_prefix_scope(self, fake_store_cls):
        store = fake_store_cls()
        for i in (0, 1):
            _fill_w8a8(store, f"layers.{i}.fc", deq=2.0)
        fmt = _make_format(
            {f"layers.{i}.fc.weight": "W8A8" for i in (0, 1)},
            store,
        )
        model = _Layers()
        fmt.apply_ir(model)
        fmt.hydrate(model, prefix="layers.0")

        assert model.layers[0].fc.weight.device.type == "cpu"
        assert model.layers[1].fc.weight.device.type == "meta"

    def test_hydrate_skips_prefixes_when_given(self, fake_store_cls):
        store = fake_store_cls()
        for i in (0, 1):
            _fill_w8a8(store, f"layers.{i}.fc", deq=2.0)
        fmt = _make_format(
            {f"layers.{i}.fc.weight": "W8A8" for i in (0, 1)},
            store,
        )
        model = _Layers()
        fmt.apply_ir(model)
        fmt.hydrate(model, skip_prefixes={"layers.1"})

        assert model.layers[0].fc.weight.device.type == "cpu"
        assert model.layers[1].fc.weight.device.type == "meta"

    def test_hydrate_raises_when_float_weight_missing_in_store(self, fake_store_cls):
        fmt = _make_format({"fc.weight": "FLOAT"}, fake_store_cls())
        model = _TwoLinear()

        with pytest.raises(SchemaValidateError):
            fmt.hydrate(model)

    def test_copy_float_keys_raises_when_shape_mismatch(self, fake_store_cls):
        store = fake_store_cls()
        store.put("head.weight", torch.zeros(2, 9))
        fmt = _make_format({"head.weight": "FLOAT"}, store)
        model = _TwoLinear()

        with pytest.raises(SchemaValidateError):
            fmt._copy_float_keys(model, ["head.weight"])

    def test_materialize_runtime_tensors_empty_meta_leaf_when_full_scope(self, fake_store_cls):
        fmt = _make_format({}, fake_store_cls())
        leaf = nn.Module()
        leaf.register_buffer("x", torch.empty(2, device="meta"))

        fmt._materialize_runtime_tensors(leaf)

        assert leaf.x.device.type == "cpu"


class TestFa3AndKvCacheFlow:
    """对应 FA3 / KV-cache 的 apply_ir + hydrate 主流程。"""

    @staticmethod
    def _attn_store(fake_store_cls):
        store = fake_store_cls()
        for branch in ("fa_q", "fa_k", "fa_v"):
            store.put_shape(f"layers.0.attn.{branch}.scale", (4, 1))
            store.put(f"layers.0.attn.{branch}.scale", torch.ones(4))
        for proj in ("k_proj", "v_proj"):
            store.put_shape(f"layers.0.attn.{proj}.kv_cache_scale", (2, 1))
            store.put_shape(f"layers.0.attn.{proj}.kv_cache_offset", (2, 1))
            store.put(f"layers.0.attn.{proj}.kv_cache_scale", torch.ones(2, 1) * 3.0)
            store.put(f"layers.0.attn.{proj}.kv_cache_offset", torch.ones(2, 1) * -1.0)
        return store

    @staticmethod
    def _fa3_description():
        return {
            "layers.0.attn.quant_type": "INT8",
            "layers.0.attn.fa_q.scale": "FAQuant",
            "layers.0.attn.fa_k.scale": "FAQuant",
            "layers.0.attn.fa_v.scale": "FAQuant",
            "kv_cache_type": "C8",
        }

    def test_apply_ir_inserts_activation_and_kv_cache_modules(self, fake_store_cls):
        fmt = _make_format(self._fa3_description(), self._attn_store(fake_store_cls), _FA3Adapter())
        model = _Blocks()

        fmt.apply_ir(model)

        attn = model.layers[0].attn
        assert isinstance(attn.fa_q, INT8FakeQuantActivationPerHead)
        assert isinstance(attn.fa_k, FakeQuantDynamicCache)
        assert isinstance(attn.fa_v, FakeQuantDynamicCache)

    def test_hydrate_binds_activation_and_kv_cache_params(self, fake_store_cls):
        fmt = _make_format(self._fa3_description(), self._attn_store(fake_store_cls), _FA3Adapter())
        model = _Blocks()
        fmt.apply_ir(model)

        fmt.hydrate(model)

        attn = model.layers[0].attn
        assert attn.fa_q.input_scale.device.type == "cpu"
        assert attn.fa_q.input_scale.tolist() == [1.0] * 4
        assert attn.fa_k.kv_cache_scale.device.type == "cpu"
        assert attn.fa_k.kv_cache_scale.tolist() == [[3.0], [3.0]]
        assert attn.fa_k.kv_cache_offset.tolist() == [[-1.0], [-1.0]]

    def test_apply_ir_raises_when_fa_targets_without_fa3_adapter(self, fake_store_cls):
        fmt = _make_format(self._fa3_description(), self._attn_store(fake_store_cls), None)
        with pytest.raises(UnsupportedError):
            fmt.apply_ir(_Blocks())

    def test_apply_kv_cache_raises_when_c8_without_fa3_adapter(self, fake_store_cls):
        fmt = _make_format({"kv_cache_type": "C8"}, fake_store_cls(), None)
        with pytest.raises(UnsupportedError):
            fmt.apply_ir(_Blocks())

    def test_apply_kv_cache_noop_when_type_not_c8(self, fake_store_cls):
        fmt = _make_format({"kv_cache_type": "F8"}, fake_store_cls(), _FA3Adapter())
        model = _Blocks()

        fmt.apply_ir(model)

        assert isinstance(model.layers[0].attn.fa_k, nn.Identity)


class TestHelpers:
    """对应 layer_quant_type / decoder_prefixes / hydrate 内部判定。"""

    def test_layer_quant_type_returns_float_without_label_check(self, fake_store_cls):
        fmt = _make_format({"fc.weight": "FLOAT"}, fake_store_cls())

        assert fmt.layer_quant_type("fc") == "FLOAT"

    def test_layer_quant_type_raises_when_key_absent(self, fake_store_cls):
        fmt = _make_format({}, fake_store_cls())

        with pytest.raises(SchemaValidateError):
            fmt.layer_quant_type("fc")

    def test_layer_quant_type_raises_when_label_unsupported(self, fake_store_cls):
        fmt = _make_format({"fc.weight": "W4A8"}, fake_store_cls())

        with pytest.raises(UnsupportedError):
            fmt.layer_quant_type("fc")

    def test_decoder_prefixes_returns_decoder_layer_names(self, fake_store_cls):
        fmt = _make_format({}, fake_store_cls())
        model = nn.Module()
        model.decoder_layer = _DecoderLayer()
        model.other = nn.Linear(3, 2)

        assert fmt.decoder_prefixes(model) == {"decoder_layer"}

    def test_hydrate_skips_activation_with_unknown_scheme(self, fake_store_cls):
        fmt = _make_format({}, fake_store_cls())
        model = nn.Module()
        module = INT8FakeQuantActivationPerHead(
            QParam(scheme=int8_per_head_sym, ext={"scale": torch.empty(4, device="meta")})
        )
        module.x_q_scheme = int8_per_channel_sym  # 不属于 _FA3_SCHEMES
        model.quantizer = module

        assert fmt.hydrate(model) is None  # hydrate 无返回值；关键是跳过不抛错


class TestUnsupportedLinearLabel:
    """AscendV1Format 对未注册 linear label 应 fail-closed。"""

    def test_apply_ir_raises_when_export_contains_unsupported_linear_label(self, tmp_path):
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 3)

        tensors = {
            "proj.weight": torch.tensor([[-2, 1, 0, 3], [4, -1, 2, 0], [1, 1, 1, 1]], dtype=torch.int8),
            "proj.weight_scale": torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32),
            "proj.weight_offset": torch.zeros(3, 1, dtype=torch.float32),
        }
        save_file(tensors, str(tmp_path / ASCENDV1_SAFETENSORS_NAME))
        description = {
            "proj.weight": "W4A16",
            "proj.weight_scale": "W4A16",
            "proj.weight_offset": "W4A16",
        }
        (tmp_path / ASCENDV1_DESC_JSON_NAME).write_text(json.dumps(description), encoding="utf-8")
        fmt = AscendV1Format(str(tmp_path))

        with pytest.raises(UnsupportedError, match="W4A16"):
            fmt.apply_ir(_Model())


class _HiF8ToyModel(nn.Module):
    def __init__(self, activation=None):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.attn = nn.Module()
        self.attn.fa_q = activation if activation is not None else nn.Identity()
        self.attn.fa_k = nn.Identity()

    def forward(self, values):
        return self.attn.fa_q(self.linear(values))


def _save_hif8_toy(model, output_path, source_path):
    output_path.mkdir()
    saver = AscendV1Saver(
        model,
        AscendV1Config(save_directory=str(output_path), part_file_size=0),
        BaseModelAdapter("FA3HiF8Toy", source_path),
    )
    saver.postprocess(BatchProcessRequest(name="", module=model))
    saver.post_run()
    return json.loads((output_path / ASCENDV1_DESC_JSON_NAME).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "scheme,scales,dynamic_k",
    [
        (hif8_per_tensor_sym, [8.0], False),
        (hif8_per_tensor_sym, [15.0], False),
        (hif8_per_tensor_sym, [32768.0], False),
        (hif8_per_head_sym, [8.0, 8.0], False),
        (hif8_per_head_sym, [15.0, 15.0], False),
        (hif8_per_head_sym, [32768.0, 32768.0], False),
        (hif8_per_head_sym, [1.0, 16.0], False),
        (hif8_per_head_sym, [15.0], False),
        (hif8_per_tensor_sym, [15.0], True),
        (hif8_per_head_sym, [15.0], True),
    ],
)
def test_hif8_export_roundtrip_preserves_scales_and_numerics_when_static(tmp_path, scheme, scales, dynamic_k):
    original = tmp_path / "original"
    original.mkdir()
    (original / "config.json").write_text('{"model_type":"FA3HiF8Toy"}', encoding="utf-8")
    activation = qir.AutoFakeQuantActivation.create(QParam(scheme, ext={"scale": torch.tensor(scales)}))
    source = _HiF8ToyModel(activation)
    if dynamic_k:
        source.attn.fa_k = qir.AutoFakeQuantActivation.create(QParam(int8_per_token_sym))
    with torch.no_grad():
        source.linear.weight.copy_(torch.eye(2))
        source.linear.bias.zero_()
    heads = len(scales) if scheme == hif8_per_head_sym else 2
    values = torch.tensor([25.0, 26.0]).reshape(1, 1, 1, 2).expand(2, heads, 3, 2)
    expected = source(values)
    export = tmp_path / ("mixed" if dynamic_k else "export")
    description = _save_hif8_toy(source, export, original)
    expected_type = "Q_HIF8_PER_TENSOR" if scheme == hif8_per_tensor_sym else "Q_HIF8"
    if dynamic_k:
        expected_type += "_K_INT8_DYNAMIC"
    assert description["attn.quant_type"] == expected_type
    assert description["fa_quant_type"] == "FAKQuant"
    assert description["attn.fa_q.scale"] == description["attn.fa_q.offset"] == "FAQuant"
    assert not any(key.startswith("fa3.") for key in description.get("optional", {}))
    scale_shape = (1,) if scheme == hif8_per_tensor_sym else (heads, 1)
    with safe_open(str(export / ASCENDV1_SAFETENSORS_NAME), framework="pt", device="cpu") as tensors:
        scale = tensors.get_tensor("attn.fa_q.scale")
        offset = tensors.get_tensor("attn.fa_q.offset")
        assert scale.dtype == offset.dtype == torch.float32
        assert tuple(scale.shape) == tuple(offset.shape) == scale_shape
        torch.testing.assert_close(scale.reshape(-1), torch.tensor(scales))
        assert not offset.any()
    restored = _HiF8ToyModel()
    loader = AscendV1Format(str(export))
    loader.bind_adapter(_FA3Adapter())
    loader.apply_ir(restored)
    assert restored.attn.fa_q.input_scale.is_meta
    loader.hydrate(restored)
    torch.testing.assert_close(restored(values), expected, rtol=0, atol=0)
    torch.testing.assert_close(restored.attn.fa_q.input_scale, source.attn.fa_q.input_scale, rtol=0, atol=0)
    if dynamic_k:
        mixed_values = torch.tensor([[[[25.0, 26.0]]]])
        torch.testing.assert_close(restored.attn.fa_q(mixed_values), source.attn.fa_q(mixed_values), rtol=0, atol=0)
        torch.testing.assert_close(restored.attn.fa_k(mixed_values), source.attn.fa_k(mixed_values), rtol=0, atol=0)
        assert restored.attn.fa_k.x_q_scheme == int8_per_token_sym
    loader.apply_ir(restored)
    loader.hydrate(restored)
    torch.testing.assert_close(restored(values), expected, rtol=0, atol=0)
    resaved = _save_hif8_toy(restored, tmp_path / "resaved", original)
    assert resaved == description


@pytest.mark.parametrize(
    "quant_type,expected",
    [
        ("INT8", {branch: int8_per_head_sym for branch in "QKV"}),
        ("FP8", {branch: fp8_e4m3_per_head_sym for branch in "QKV"}),
        ("INT8_DYNAMIC", {branch: int8_per_token_sym for branch in "QKV"}),
        ("FP8_DYNAMIC", {branch: fp8_e4m3_per_token_sym for branch in "QKV"}),
        ("HIF8", {branch: hif8_per_head_sym for branch in "QKV"}),
        ("HIF8_PER_TENSOR", {branch: hif8_per_tensor_sym for branch in "QKV"}),
        ("Q_HIF8_PER_TENSOR_K_INT8_DYNAMIC", {"Q": hif8_per_tensor_sym, "K": int8_per_token_sym}),
        ("Q_HIF8_K_HIF8_PER_TENSOR", {"Q": hif8_per_head_sym, "K": hif8_per_tensor_sym}),
    ],
)
def test_fa3_descriptor_preserves_scope_and_static_semantics_when_format_varies(quant_type, expected):
    assert _parse_fa_quant_type(quant_type) == expected


@pytest.mark.parametrize("quant_type", ["Q_HIF8_PER_CHANNEL", "Q_HIF8_PER_TOKEN", "Q_HIF8_DYNAMIC"])
def test_fa3_descriptor_rejects_unregistered_scheme_when_scope_or_dynamic_suffix_is_present(quant_type):
    with pytest.raises(SchemaValidateError, match="Unsupported FA3 quant_type"):
        _parse_fa_quant_type(quant_type)
