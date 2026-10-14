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

import pytest  # noqa: F401
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.config import FlatQuantConfig


def test_flat_quant_config_defaults():
    cfg = FlatQuantConfig()
    # General
    assert cfg.seed == 0
    assert cfg.quant_by_quant is False
    assert cfg.diag_relu is True
    assert cfg.amp_dtype == 'bfloat16'
    # Activation
    assert cfg.a_bits == 4
    assert cfg.a_groupsize == -1
    assert cfg.a_asym is False
    # Weight
    assert cfg.w_bits == 4
    assert cfg.w_groupsize == -1
    assert cfg.w_asym is False
    assert cfg.gptq is False
    assert cfg.gptq_mse is False
    assert cfg.percdamp == 0.01
    assert cfg.act_order is False
    # Calibration
    assert cfg.epochs == 15
    assert cfg.nsamples is None
    assert cfg.cali_bsz == 1
    assert cfg.flat_lr == 5e-3
    assert cfg.cali_trans is True
    assert cfg.add_diag is True
    assert cfg.lwc is True
    assert cfg.lac is True
    assert cfg.diag_init == 'sq_style'
    assert cfg.diag_alpha == 0.3
    assert cfg.warmup is False
    assert cfg.deactive_amp is False
    assert cfg.direct_inv is False
    # quantize分支
    assert cfg.quantize is True


def test_flat_quant_config_quantize_logic():
    cfg = FlatQuantConfig()
    # w_bits >= 16, a_bits < 16
    cfg.w_bits = 16
    cfg.a_bits = 4
    cfg.quantize = (cfg.w_bits < 16) or (cfg.a_bits < 16)
    assert cfg.quantize is True
    # w_bits < 16, a_bits >= 16
    cfg.w_bits = 4
    cfg.a_bits = 16
    cfg.quantize = (cfg.w_bits < 16) or (cfg.a_bits < 16)
    assert cfg.quantize is True
    # w_bits >= 16, a_bits >= 16
    cfg.w_bits = 16
    cfg.a_bits = 16
    cfg.quantize = (cfg.w_bits < 16) or (cfg.a_bits < 16)
    assert cfg.quantize is False
    # w_bits < 16, a_bits < 16
    cfg.w_bits = 8
    cfg.a_bits = 8
    cfg.quantize = (cfg.w_bits < 16) or (cfg.a_bits < 16)
    assert cfg.quantize is True


def test_flat_quant_config_types():
    cfg = FlatQuantConfig()
    assert isinstance(cfg.seed, int)
    assert isinstance(cfg.epochs, int)
    assert isinstance(cfg.flat_lr, float)
    assert isinstance(cfg.diag_alpha, float)
    assert isinstance(cfg.diag_init, str)
    assert isinstance(cfg.lwc, bool)
    assert isinstance(cfg.lac, bool)
    assert isinstance(cfg.add_diag, bool)
    assert isinstance(cfg.deactive_amp, bool)
    assert isinstance(cfg.direct_inv, bool)
    assert isinstance(cfg.quantize, bool)
