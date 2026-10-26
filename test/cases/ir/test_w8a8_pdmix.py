#!/usr/bin/env python
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
"""

import torch

from msmodelslim.ir.const import int8_pd_mix_asym, int8_per_channel_sym
from msmodelslim.ir.qal import QDType, QParam, QStorage
from msmodelslim.ir.w8a8_pdmix import PDMixState, W8A8PDMixFakeQuantLinear

_IN_FEATURES = 8
_OUT_FEATURES = 4


def _make_module(input_scale: float = 1.0) -> W8A8PDMixFakeQuantLinear:
    x_q_param = QParam(
        scheme=int8_pd_mix_asym,
        ext={"scale": torch.full((1,), input_scale), "offset": torch.zeros(1)},
    )
    w_q_param = QParam(scheme=int8_per_channel_sym, ext={"scale": torch.ones(_OUT_FEATURES)})
    w_q = QStorage(QDType.INT8, torch.ones(_OUT_FEATURES, _IN_FEATURES))
    return W8A8PDMixFakeQuantLinear(x_q_param, w_q_param, w_q, None)


def test_default_state_is_prefilling():
    module = _make_module()

    assert module.state == PDMixState.PREFILLING


def test_set_state_switches_module_state():
    module = _make_module()

    module.set_state(PDMixState.DECODING)

    assert module.state == PDMixState.DECODING


def test_prefilling_forward_uses_dynamic_activation_quantization():
    # 预填充按 per-token 动态量化：0.6 精确还原 -> 0.6 * sum(ones(8)) = 4.8
    module = _make_module(input_scale=0.5)
    x = torch.full((1, _IN_FEATURES), 0.6, dtype=torch.float32)

    output = module(x)

    assert output.shape == (1, _OUT_FEATURES)
    assert torch.allclose(output, torch.full((1, _OUT_FEATURES), 4.8), atol=1e-5)


def test_decoding_forward_uses_static_activation_quantization():
    # 解码回退为静态量化：round(0.6/0.5)=1 -> 0.5 -> 0.5 * sum(ones(8)) = 4.0
    module = _make_module(input_scale=0.5)
    module.set_state(PDMixState.DECODING)
    x = torch.full((1, _IN_FEATURES), 0.6, dtype=torch.float32)

    output = module(x)

    assert output.shape == (1, _OUT_FEATURES)
    assert torch.allclose(output, torch.full((1, _OUT_FEATURES), 4.0), atol=1e-5)


def test_switch_back_to_prefilling_restores_dynamic_path():
    module = _make_module(input_scale=0.5)
    module.set_state(PDMixState.DECODING)
    module.set_state(PDMixState.PREFILLING)
    x = torch.full((1, _IN_FEATURES), 0.6, dtype=torch.float32)

    output = module(x)

    assert torch.allclose(output, torch.full((1, _OUT_FEATURES), 4.8), atol=1e-5)


def test_pdmix_state_values():
    assert PDMixState.PREFILLING.value == "prefilling"
    assert PDMixState.DECODING.value == "decoding"
