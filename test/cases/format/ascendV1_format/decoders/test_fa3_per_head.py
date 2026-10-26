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

msmodelslim/format/ascendV1_format/decoders/fa3_per_head.py 的单元测试。
"""

import pytest
import torch

from msmodelslim.format.ascendV1_format.decoders.fa3_per_head import Fa3PerHeadDecoder
from msmodelslim.ir.const import fp8_e4m3_per_head_sym, int8_per_head_sym
from msmodelslim.ir.fp8_activation_static import FP8FakeQuantActivationPerHead
from msmodelslim.ir.int8_activation_static import INT8FakeQuantActivationPerHead
from msmodelslim.utils.exception import SchemaValidateError

_P = "lm_attn"


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
