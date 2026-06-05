#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

from msmodelslim.ir.qal import QDType
from msmodelslim.ir.qal.qbase import QScheme, QScope

int8_per_token_asym = QScheme(scope=QScope.PER_TOKEN, dtype=QDType.INT8, symmetric=False)
int8_per_token_sym = QScheme(scope=QScope.PER_TOKEN, dtype=QDType.INT8, symmetric=True)
int8_per_tensor_asym = QScheme(scope=QScope.PER_TENSOR, dtype=QDType.INT8, symmetric=False)
int8_per_tensor_sym = QScheme(scope=QScope.PER_TENSOR, dtype=QDType.INT8, symmetric=True)
int8_per_channel_asym = QScheme(scope=QScope.PER_CHANNEL, dtype=QDType.INT8, symmetric=False)
int8_per_channel_sym = QScheme(scope=QScope.PER_CHANNEL, dtype=QDType.INT8, symmetric=True)
int8_per_group_sym = QScheme(scope=QScope.PER_GROUP, dtype=QDType.INT8, symmetric=True)
int8_per_group_asym = QScheme(scope=QScope.PER_GROUP, dtype=QDType.INT8, symmetric=False)
int8_pd_mix_asym = QScheme(scope=QScope.PD_MIX, dtype=QDType.INT8, symmetric=False)
mxfp4_per_block_sym = QScheme(scope=QScope.PER_BLOCK, dtype=QDType.MXFP4, symmetric=True)
mxfp4_dual_scale_sym = QScheme(scope=QScope.DUAL_SCALE, dtype=QDType.MXFP4, symmetric=True)
mxfp8_per_block_sym = QScheme(scope=QScope.PER_BLOCK, dtype=QDType.MXFP8, symmetric=True)
int8_per_head_sym = QScheme(scope=QScope.PER_HEAD, dtype=QDType.INT8, symmetric=True)

int4_per_token_asym = QScheme(scope=QScope.PER_TOKEN, dtype=QDType.INT4, symmetric=False)
int4_per_token_sym = QScheme(scope=QScope.PER_TOKEN, dtype=QDType.INT4, symmetric=True)
int4_per_tensor_asym = QScheme(scope=QScope.PER_TENSOR, dtype=QDType.INT4, symmetric=False)
int4_per_tensor_sym = QScheme(scope=QScope.PER_TENSOR, dtype=QDType.INT4, symmetric=True)
int4_per_channel_asym = QScheme(scope=QScope.PER_CHANNEL, dtype=QDType.INT4, symmetric=False)
int4_per_channel_sym = QScheme(scope=QScope.PER_CHANNEL, dtype=QDType.INT4, symmetric=True)
int4_per_group_asym = QScheme(scope=QScope.PER_GROUP, dtype=QDType.INT4, symmetric=False)
int4_per_group_sym = QScheme(scope=QScope.PER_GROUP, dtype=QDType.INT4, symmetric=True)

fp8_e4m3_per_token_sym = QScheme(scope=QScope.PER_TOKEN, dtype=QDType.FP8_E4M3, symmetric=True)
fp8_e4m3_per_tensor_sym = QScheme(scope=QScope.PER_TENSOR, dtype=QDType.FP8_E4M3, symmetric=True)
fp8_e4m3_per_channel_sym = QScheme(scope=QScope.PER_CHANNEL, dtype=QDType.FP8_E4M3, symmetric=True)
fp8_e4m3_per_head_sym = QScheme(scope=QScope.PER_HEAD, dtype=QDType.FP8_E4M3, symmetric=True)

float_per_tensor_sym = QScheme(scope=QScope.PER_TENSOR, dtype=QDType.FLOAT, symmetric=True)
