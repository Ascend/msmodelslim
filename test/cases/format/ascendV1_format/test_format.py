# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You may use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------

msmodelslim/format/ascendV1_format/format.py 的单元测试。
"""

import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from msmodelslim.format.ascendV1_format.decoder import AscendV1FormatDecoder
from msmodelslim.format.ascendV1_format.format import (
    ASCENDV1_DESC_JSON_NAME,
    ASCENDV1_SAFETENSORS_NAME,
    AscendV1Format,
)
from msmodelslim.ir.attention import FakeQuantDynamicCache
from msmodelslim.ir.const import int8_per_channel_sym, int8_per_head_sym
from msmodelslim.ir.int8_activation_static import INT8FakeQuantActivationPerHead
from msmodelslim.ir.qal import QParam
from msmodelslim.ir.w8a8_static import W8A8StaticFakeQuantLinear
from msmodelslim.processor.quant.fa3.interface import FA3QuantAdapterInterface
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


def _make_format(description, store, adapter=None):
    fmt = AscendV1Format.__new__(AscendV1Format)
    fmt.model_path = "/export"
    fmt.device = torch.device("cpu")
    fmt.description = description
    fmt.store = store
    fmt.group_size = 0
    fmt._decoder = AscendV1FormatDecoder(store)
    fmt._adapter = adapter
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
