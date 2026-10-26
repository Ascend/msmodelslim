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

msmodelslim/format/ascendV1_format/decoders/fa3_per_token.py 的单元测试。
"""

import torch

from msmodelslim.format.ascendV1_format.decoders.fa3_per_token import Fa3PerTokenDecoder
from msmodelslim.ir.activation_dynamic import FakeQuantActivationPerToken
from msmodelslim.ir.const import int8_per_token_sym

_P = "lm_attn"


class TestFa3PerTokenDecoder:
    """对应 Fa3PerTokenDecoder。"""

    def test_ir_type_returns_per_token_activation(self, fake_store_cls):
        decoder = Fa3PerTokenDecoder(fake_store_cls())

        assert decoder.ir_type(int8_per_token_sym) is FakeQuantActivationPerToken

    def test_structure_and_params_carry_scheme_without_tensor_reads(self, fake_store_cls):
        decoder = Fa3PerTokenDecoder(fake_store_cls())

        spec = decoder.structure(_P, scheme=int8_per_token_sym)
        params = decoder.params(_P, torch.device("cpu"), scheme=int8_per_token_sym)

        assert spec.scheme == int8_per_token_sym
        assert params.scale is None
        assert params.scheme == int8_per_token_sym
