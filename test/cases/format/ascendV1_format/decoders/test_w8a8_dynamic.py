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

msmodelslim/format/ascendV1_format/decoders/w8a8_dynamic.py 的单元测试。
"""

import pytest
import torch

from msmodelslim.format.ascendV1_format.decoders.w8a8_dynamic import W8A8DynamicDecoder
from msmodelslim.ir.w8a8_dynamic import W8A8DynamicPerChannelFakeQuantLinear, W8A8DynamicPerGroupFakeQuantLinear
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
        assert ps.weight_offset.dtype == torch.float32
        assert ps.weight_offset.tolist() == [0.0, 0.0]
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
