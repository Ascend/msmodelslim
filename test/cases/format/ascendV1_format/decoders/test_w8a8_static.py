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

msmodelslim/format/ascendV1_format/decoders/w8a8_static.py 的单元测试。
"""

import pytest
import torch

from msmodelslim.format.ascendV1_format.decoders.w8a8_static import W8A8StaticDecoder, _reconstruct_bias
from msmodelslim.ir.w8a8_static import W8A8StaticFakeQuantLinear
from msmodelslim.utils.exception import SchemaValidateError

_PREFIX = "fc"


def _f32_bits(value):
    return torch.tensor([value]).view(torch.int32).to(torch.int64)


def _add_required(store):
    store.put_shape(f"{_PREFIX}.weight", (2, 3))
    store.put_shape(f"{_PREFIX}.input_scale", (1,))
    store.put_shape(f"{_PREFIX}.input_offset", (1,))
    store.put_shape(f"{_PREFIX}.deq_scale", (2,))
    return store


class TestW8A8StaticDecoder:
    """对应 W8A8StaticDecoder。"""

    def test_label_and_ir_type_when_called(self, fake_store_cls):
        assert W8A8StaticDecoder.label == "W8A8"
        assert W8A8StaticDecoder(fake_store_cls()).ir_type(group_size=128) is W8A8StaticFakeQuantLinear

    def test_structure_builds_spec_when_keys_present(self, fake_store_cls):
        store = _add_required(fake_store_cls())
        store.put_shape(f"{_PREFIX}.bias", (2,))

        spec = W8A8StaticDecoder(store).structure(_PREFIX)

        assert spec.weight_shape == (2, 3)
        assert spec.input_scale_shape == (1,)
        assert spec.input_offset_shape == (1,)
        assert spec.weight_scale_shape == (2,)
        assert spec.bias_shape == (2,)

    def test_structure_falls_back_to_quant_bias_shape_when_no_bias(self, fake_store_cls):
        store = _add_required(fake_store_cls())
        store.put_shape(f"{_PREFIX}.quant_bias", (2,))

        spec = W8A8StaticDecoder(store).structure(_PREFIX)

        assert spec.bias_shape == (2,)

    def test_structure_raises_when_required_key_missing(self, fake_store_cls):
        store = fake_store_cls()
        store.put_shape(f"{_PREFIX}.weight", (2, 3))

        with pytest.raises(SchemaValidateError):
            W8A8StaticDecoder(store).structure(_PREFIX)

    def test_params_decode_and_keep_bias_when_bias_tensor(self, fake_store_cls):
        store = _add_required(fake_store_cls())
        store.put(f"{_PREFIX}.weight", torch.ones(2, 3, dtype=torch.int8))
        store.put(f"{_PREFIX}.input_scale", torch.tensor([2.0]))
        store.put(f"{_PREFIX}.input_offset", torch.tensor([0.0]))
        store.put(f"{_PREFIX}.deq_scale", _f32_bits(4.0).repeat(2))
        store.put(f"{_PREFIX}.bias", torch.tensor([1.0, 2.0]))

        ps = W8A8StaticDecoder(store).params(_PREFIX, torch.device("cpu"))

        assert ps.weight_int8.dtype == torch.int8
        assert ps.weight_scale.tolist() == [2.0, 2.0]  # deq 4.0 / input_scale 2.0
        assert ps.bias.tolist() == [1.0, 2.0]

    def test_params_reconstructs_from_quant_bias_when_no_bias(self, fake_store_cls):
        store = _add_required(fake_store_cls())
        store.put(f"{_PREFIX}.weight", torch.zeros(2, 3, dtype=torch.int8))
        store.put(f"{_PREFIX}.input_scale", torch.tensor([1.0]))
        store.put(f"{_PREFIX}.input_offset", torch.tensor([0.5]))
        store.put(f"{_PREFIX}.deq_scale", _f32_bits(1.0))
        store.put(f"{_PREFIX}.quant_bias", torch.tensor([3.0, 4.0]))

        ps = W8A8StaticDecoder(store).params(_PREFIX, torch.device("cpu"))

        assert ps.bias.tolist() == [3.0, 4.0]

    def test_params_bias_none_when_no_bias_keys(self, fake_store_cls):
        store = _add_required(fake_store_cls())
        store.put(f"{_PREFIX}.weight", torch.zeros(2, 3, dtype=torch.int8))
        store.put(f"{_PREFIX}.input_scale", torch.tensor([1.0]))
        store.put(f"{_PREFIX}.input_offset", torch.tensor([0.0]))
        store.put(f"{_PREFIX}.deq_scale", _f32_bits(1.0))

        assert W8A8StaticDecoder(store).params(_PREFIX, torch.device("cpu")).bias is None

    def test_reconstruct_bias_returns_none_when_no_keys(self, fake_store_cls):
        store = fake_store_cls()

        bias = _reconstruct_bias(_PREFIX, store, torch.zeros(2, 3), torch.zeros(1), torch.ones(1), torch.device("cpu"))

        assert bias is None
