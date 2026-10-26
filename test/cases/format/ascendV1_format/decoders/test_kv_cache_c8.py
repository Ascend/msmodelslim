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

msmodelslim/format/ascendV1_format/decoders/kv_cache_c8.py 的单元测试。
"""

import pytest
import torch

from msmodelslim.format.ascendV1_format.decoders.kv_cache_c8 import KvCacheC8Decoder
from msmodelslim.ir.attention import FakeQuantDynamicCache
from msmodelslim.ir.const import int8_per_channel_sym
from msmodelslim.utils.exception import SchemaValidateError

_P = "lm_attn"


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
        assert ps.kv_cache_offset.tolist() == [[0.0], [0.0]]
        assert ps.scheme == int8_per_channel_sym
