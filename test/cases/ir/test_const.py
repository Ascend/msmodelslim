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

import pytest

from msmodelslim.ir import const
from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.ir.qal.qbase import QScheme

# (常量名, dtype, scope, symmetric)
_SCHEME_TABLE = [
    ("int8_per_token_asym", QDType.INT8, QScope.PER_TOKEN, False),
    ("int8_per_token_sym", QDType.INT8, QScope.PER_TOKEN, True),
    ("int8_per_tensor_asym", QDType.INT8, QScope.PER_TENSOR, False),
    ("int8_per_tensor_sym", QDType.INT8, QScope.PER_TENSOR, True),
    ("int8_per_channel_asym", QDType.INT8, QScope.PER_CHANNEL, False),
    ("int8_per_channel_sym", QDType.INT8, QScope.PER_CHANNEL, True),
    ("int8_per_channel_asym_neg_offset", QDType.INT8, QScope.PER_CHANNEL_NEG_OFFSET, False),
    ("int8_per_group_sym", QDType.INT8, QScope.PER_GROUP, True),
    ("int8_per_group_asym", QDType.INT8, QScope.PER_GROUP, False),
    ("int8_pd_mix_asym", QDType.INT8, QScope.PD_MIX, False),
    ("int8_per_head_sym", QDType.INT8, QScope.PER_HEAD, True),
    ("int4_per_token_asym", QDType.INT4, QScope.PER_TOKEN, False),
    ("int4_per_token_sym", QDType.INT4, QScope.PER_TOKEN, True),
    ("int4_per_tensor_asym", QDType.INT4, QScope.PER_TENSOR, False),
    ("int4_per_tensor_sym", QDType.INT4, QScope.PER_TENSOR, True),
    ("int4_per_channel_asym", QDType.INT4, QScope.PER_CHANNEL, False),
    ("int4_per_channel_sym", QDType.INT4, QScope.PER_CHANNEL, True),
    ("int4_per_group_asym", QDType.INT4, QScope.PER_GROUP, False),
    ("int4_per_group_sym", QDType.INT4, QScope.PER_GROUP, True),
    ("mxfp4_per_block_sym", QDType.MXFP4, QScope.PER_BLOCK, True),
    ("mxfp4_dual_scale_sym", QDType.MXFP4, QScope.DUAL_SCALE, True),
    ("mxfp8_per_block_sym", QDType.MXFP8, QScope.PER_BLOCK, True),
    ("mxfp8_per_channel_sym", QDType.MXFP8, QScope.PER_CHANNEL, True),
    ("fp8_e4m3_per_token_sym", QDType.FP8_E4M3, QScope.PER_TOKEN, True),
    ("fp8_e4m3_per_tensor_sym", QDType.FP8_E4M3, QScope.PER_TENSOR, True),
    ("fp8_e4m3_per_channel_sym", QDType.FP8_E4M3, QScope.PER_CHANNEL, True),
    ("fp8_e4m3_per_head_sym", QDType.FP8_E4M3, QScope.PER_HEAD, True),
    ("float_per_tensor_sym", QDType.FLOAT, QScope.PER_TENSOR, True),
]


@pytest.mark.parametrize("name,dtype,scope,symmetric", _SCHEME_TABLE)
def test_module_constant_is_qscheme_with_expected_fields(name, dtype, scope, symmetric):
    scheme = getattr(const, name)

    assert isinstance(scheme, QScheme)
    assert scheme.dtype == dtype
    assert scheme.scope == scope
    assert scheme.symmetric is symmetric


def test_symmetric_and_asymmetric_variants_differ_for_same_dtype_and_scope():
    assert const.int8_per_channel_sym != const.int8_per_channel_asym
    assert const.int8_per_channel_sym.symmetric is True
    assert const.int8_per_channel_asym.symmetric is False
