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

msmodelslim/format/ascendV1_format/decoders/w8a8_mx.py 的单元测试。
"""

import pytest
import torch

from msmodelslim.format.ascendV1_format.decoders.w8a8_mx import W8A8MxDecoder, decode_mxfp8_weight_scale
from msmodelslim.ir.w8a8_mx_dynamic import W8A8MXDynamicPerBlockFakeQuantLinear
from msmodelslim.utils.exception import SchemaValidateError

_P = "lm_attn"


class TestDecodeMxfp8WeightScale:
    """对应模块级函数 decode_mxfp8_weight_scale。"""

    def test_inverts_uint8_bias_and_unsqueezes_last_axis(self):
        stored = torch.tensor([127 + 3, 127 - 4], dtype=torch.uint8)

        scale = decode_mxfp8_weight_scale(stored)

        assert scale.tolist() == [[3.0], [-4.0]]
        assert scale.shape == (2, 1)

    def test_honours_explicit_w_axes_when_given(self):
        stored = torch.tensor([127 + 5, 127 + 6], dtype=torch.uint8)

        scale = decode_mxfp8_weight_scale(stored, w_axes=0)

        assert scale.shape == (1, 2)
        assert scale.flatten().tolist() == [5.0, 6.0]

    def test_raises_when_w_axes_not_int(self):
        with pytest.raises(SchemaValidateError):
            decode_mxfp8_weight_scale(torch.tensor([127]), w_axes=(1,))


class TestW8A8MxDecoder:
    """对应 W8A8MxDecoder。"""

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

    def test_structure_raises_when_required_key_missing(self, fake_store_cls):
        store = fake_store_cls()
        store.put_shape(f"{_P}.weight", (2, 3))

        with pytest.raises(SchemaValidateError):
            W8A8MxDecoder(store).structure(_P)

    def test_params_decodes_bf16_and_zero_offset(self, fake_store_cls):
        store = fake_store_cls()
        store.put(f"{_P}.weight", torch.ones(2, 3, dtype=torch.bfloat16))
        store.put(f"{_P}.weight_scale", torch.tensor([127 + 2, 127 - 1], dtype=torch.uint8))
        store.put(f"{_P}.bias", torch.ones(2, dtype=torch.bfloat16))

        ps = W8A8MxDecoder(store).params(_P, torch.device("cpu"))

        assert ps.weight.dtype == torch.bfloat16
        assert ps.weight_scale[:, 0].tolist() == [2.0, -1.0]
        assert ps.weight_offset.tolist() == [[0.0], [0.0]]

    def test_params_bias_none_when_absent(self, fake_store_cls):
        store = fake_store_cls()
        store.put(f"{_P}.weight", torch.ones(2, 3, dtype=torch.bfloat16))
        store.put(f"{_P}.weight_scale", torch.tensor([127, 127], dtype=torch.uint8))

        assert W8A8MxDecoder(store).params(_P, torch.device("cpu")).bias is None
