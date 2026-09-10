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

msmodelslim/format/ascendV1_format/decoders/ 其余 decoder 的单元测试。
"""

import pytest
import torch

from msmodelslim.format.ascendV1_format.decoders.fa3_per_block import Fa3PerBlockDecoder
from msmodelslim.format.ascendV1_format.decoders.fa3_per_head import Fa3PerHeadDecoder
from msmodelslim.format.ascendV1_format.decoders.fa3_per_token import Fa3PerTokenDecoder
from msmodelslim.format.ascendV1_format.decoders.kv_cache_c8 import KvCacheC8Decoder
from msmodelslim.format.ascendV1_format.decoders.w8a8_dynamic import W8A8DynamicDecoder
from msmodelslim.format.ascendV1_format.decoders.w8a8_mx import (
    W8A8MxDecoder,
    _unsqueeze_shape_at,
    decode_mxfp8_weight_scale,
)
from msmodelslim.ir.activation_dynamic import FakeQuantActivationPerBlock, FakeQuantActivationPerToken
from msmodelslim.ir.attention import FakeQuantDynamicCache
from msmodelslim.ir.const import fp8_e4m3_per_head_sym, int8_per_channel_sym, int8_per_head_sym
from msmodelslim.ir.fp8_activation_static import FP8FakeQuantActivationPerHead
from msmodelslim.ir.int8_activation_static import INT8FakeQuantActivationPerHead
from msmodelslim.ir.w8a8_dynamic import W8A8DynamicPerChannelFakeQuantLinear, W8A8DynamicPerGroupFakeQuantLinear
from msmodelslim.ir.w8a8_mx_dynamic import W8A8MXDynamicPerBlockFakeQuantLinear
from msmodelslim.utils.exception import SchemaValidateError

_P = "lm_attn"


class TestW8A8DynamicDecoder:
    """对应 W8A8DynamicDecoder。"""

    def test_ir_type_per_channel_when_no_group(self, fake_store_cls):
        assert W8A8DynamicDecoder(fake_store_cls()).ir_type(0) is W8A8DynamicPerChannelFakeQuantLinear

    def test_ir_type_per_group_when_group_size(self, fake_store_cls):
        assert W8A8DynamicDecoder(fake_store_cls()).ir_type(128) is W8A8DynamicPerGroupFakeQuantLinear

    def test_structure_per_channel_squeezes_trailing_one(self, fake_store_cls):
        store = fake_store_cls()
        store.put_shape(f"{_P}.weight", (2, 3))
        store.put_shape(f"{_P}.weight_scale", (2, 1))

        spec = W8A8DynamicDecoder(store).structure(_P)

        assert spec.group_size == 0
        assert spec.weight_scale_shape == (2,)
        assert spec.bias_shape is None

    def test_structure_per_group_keeps_offset_shape_or_defaults(self, fake_store_cls):
        store = fake_store_cls()
        store.put_shape(f"{_P}.weight", (2, 3))
        store.put_shape(f"{_P}.weight_scale", (2, 2))
        store.put_shape(f"{_P}.bias", (2,))

        spec = W8A8DynamicDecoder(store).structure(_P, group_size=128)

        assert spec.group_size == 128
        assert spec.weight_offset_shape == (2, 2)
        assert spec.bias_shape == (2,)

    def test_structure_per_group_defaults_offset_to_scale_shape(self, fake_store_cls):
        store = fake_store_cls()
        store.put_shape(f"{_P}.weight", (2, 3))
        store.put_shape(f"{_P}.weight_scale", (2, 2))

        assert W8A8DynamicDecoder(store).structure(_P, group_size=128).weight_offset_shape == (2, 2)

    def test_params_decodes_int8_and_optional_tensors(self, fake_store_cls):
        store = fake_store_cls()
        store.put(f"{_P}.weight", torch.ones(2, 3, dtype=torch.int8))
        store.put(f"{_P}.weight_scale", torch.full((2,), 0.5))
        store.put(f"{_P}.weight_offset", torch.zeros(2))
        store.put(f"{_P}.bias", torch.ones(2))

        ps = W8A8DynamicDecoder(store).params(_P, torch.device("cpu"), group_size=0)

        assert ps.weight_int8.dtype == torch.int8
        assert ps.weight_offset is not None
        assert ps.bias.tolist() == [1.0, 1.0]
        assert ps.group_size == 0

    def test_params_optional_tensors_none_when_absent(self, fake_store_cls):
        store = fake_store_cls()
        store.put(f"{_P}.weight", torch.ones(2, 3, dtype=torch.int8))
        store.put(f"{_P}.weight_scale", torch.full((2,), 0.5))

        ps = W8A8DynamicDecoder(store).params(_P, torch.device("cpu"), group_size=128)

        assert ps.weight_offset is None
        assert ps.bias is None
        assert ps.group_size == 128

    def test_structure_raises_when_required_key_missing(self, fake_store_cls):
        with pytest.raises(SchemaValidateError):
            W8A8DynamicDecoder(fake_store_cls()).structure(_P)


class TestW8A8MxDecoder:
    """对应 W8A8MxDecoder 与解码函数。"""

    def test_unsqueeze_shape_at_positive_and_negative(self):
        assert _unsqueeze_shape_at((2, 3), -1) == (2, 3, 1)
        assert _unsqueeze_shape_at((2, 3), 0) == (1, 2, 3)

    def test_decode_mxfp8_weight_scale_inverts_uint8_bias(self):
        stored = torch.tensor([127 + 3, 127 - 4], dtype=torch.uint8)

        scale = decode_mxfp8_weight_scale(stored)

        assert scale.tolist() == [[3.0], [-4.0]]
        assert scale.shape == (2, 1)

    def test_decode_mxfp8_weight_scale_raises_when_w_axes_not_int(self):
        with pytest.raises(SchemaValidateError):
            decode_mxfp8_weight_scale(torch.tensor([127]), w_axes=(1,))

    def test_label_ir_type_and_structure(self, fake_store_cls):
        store = fake_store_cls()
        store.put_shape(f"{_P}.weight", (2, 3))
        store.put_shape(f"{_P}.weight_scale", (2,))
        store.put_shape(f"{_P}.bias", (2,))
        decoder = W8A8MxDecoder(store)

        assert decoder.label == "W8A8_MXFP8"
        assert decoder.ir_type() is W8A8MXDynamicPerBlockFakeQuantLinear
        spec = decoder.structure(_P)
        assert spec.weight_scale_shape == (2, 1)
        assert spec.w_axes == -1
        assert spec.bias_shape == (2,)

    def test_params_decodes_bf16_and_zero_offset(self, fake_store_cls):
        store = fake_store_cls()
        store.put(f"{_P}.weight", torch.ones(2, 3, dtype=torch.bfloat16))
        store.put(f"{_P}.weight_scale", torch.tensor([127 + 2, 127 - 1], dtype=torch.uint8))
        store.put(f"{_P}.bias", torch.ones(2, dtype=torch.bfloat16))

        ps = W8A8MxDecoder(store).params(_P, torch.device("cpu"))

        assert ps.weight.dtype == torch.bfloat16
        assert ps.weight_scale[:, 0].tolist() == [2.0, -1.0]
        assert ps.weight_offset.tolist() == [[0.0], [0.0]]


class TestFa3PerHeadDecoder:
    """对应 Fa3PerHeadDecoder。"""

    def test_ir_type_maps_schemes_when_called(self, fake_store_cls):
        decoder = Fa3PerHeadDecoder(fake_store_cls())

        assert decoder.ir_type(int8_per_head_sym) is INT8FakeQuantActivationPerHead
        assert decoder.ir_type(fp8_e4m3_per_head_sym) is FP8FakeQuantActivationPerHead

    def test_structure_squeezes_scale_shape(self, fake_store_cls):
        store = fake_store_cls()
        store.put_shape(f"{_P}.scale", (4, 1))

        spec = Fa3PerHeadDecoder(store).structure(_P, scheme=int8_per_head_sym)

        assert spec.input_scale_shape == (4,)
        assert spec.scheme == int8_per_head_sym

    def test_params_squeezes_trailing_singleton(self, fake_store_cls):
        store = fake_store_cls()
        store.put(f"{_P}.scale", torch.ones(4, 1))

        ps = Fa3PerHeadDecoder(store).params(_P, torch.device("cpu"), scheme=int8_per_head_sym)

        assert ps.scale.shape == (4,)
        assert ps.scheme == int8_per_head_sym

    def test_structure_raises_when_scale_missing(self, fake_store_cls):
        with pytest.raises(SchemaValidateError):
            Fa3PerHeadDecoder(fake_store_cls()).structure(_P, scheme=int8_per_head_sym)


class TestFa3PerTokenDecoder:
    """对应 Fa3PerTokenDecoder。"""

    def test_ir_type_and_no_tensor_reading(self, fake_store_cls):
        decoder = Fa3PerTokenDecoder(fake_store_cls())

        assert decoder.ir_type(int8_per_head_sym) is FakeQuantActivationPerToken
        assert decoder.structure(_P, scheme=int8_per_head_sym).scheme == int8_per_head_sym
        assert decoder.params(_P, torch.device("cpu"), scheme=int8_per_head_sym).scale is None


class TestFa3PerBlockDecoder:
    """对应 Fa3PerBlockDecoder。"""

    def test_ir_type_and_no_tensor_reading(self, fake_store_cls):
        decoder = Fa3PerBlockDecoder(fake_store_cls())

        assert decoder.ir_type(int8_per_head_sym) is FakeQuantActivationPerBlock
        assert decoder.structure(_P, scheme=int8_per_head_sym).scheme == int8_per_head_sym
        assert decoder.params(_P, torch.device("cpu"), scheme=int8_per_head_sym).scale is None


class TestKvCacheC8Decoder:
    """对应 KvCacheC8Decoder。"""

    def test_label_ir_type_and_structure(self, fake_store_cls):
        store = fake_store_cls()
        store.put_shape(f"{_P}.kv_cache_scale", (2, 1))
        store.put_shape(f"{_P}.kv_cache_offset", (2, 1))
        decoder = KvCacheC8Decoder(store)

        assert decoder.label == "C8"
        assert decoder.ir_type is FakeQuantDynamicCache
        spec = decoder.structure(_P)
        assert spec.scheme == int8_per_channel_sym
        assert spec.scale_shape == (2, 1)

    def test_structure_raises_when_tensors_missing(self, fake_store_cls):
        with pytest.raises(SchemaValidateError):
            KvCacheC8Decoder(fake_store_cls()).structure(_P)

    def test_params_decodes_scale_and_offset(self, fake_store_cls):
        store = fake_store_cls()
        store.put(f"{_P}.kv_cache_scale", torch.ones(2, 1))
        store.put(f"{_P}.kv_cache_offset", torch.zeros(2, 1))

        ps = KvCacheC8Decoder(store).params(_P, torch.device("cpu"))

        assert ps.kv_cache_scale.shape == (2, 1)
        assert ps.scheme == int8_per_channel_sym
