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
from msmodelslim.core.quantization_schema.qal import QStorage
from msmodelslim.core.quantization_schema.qal.qbase import QDType, QScope, QParam, QScheme
from msmodelslim.core.quantization_schema.qal.qregistry import QFuncRegistry


MAX_LV3_IDX = 3
MAX_LV2_IDX = 2
MAX_LV1_IDX = 1


@QFuncRegistry.register(dispatch_key=(QDType.HiF4, QScope.PER_BLOCK, True), api_name="calculate_qparam")
def calculate_hifx_qparam(
    min_val: torch.Tensor, max_val: torch.Tensor, q_dtype: QDType, q_scope: QScope, symmetric: bool, **kwargs
) -> QParam:
    axes = kwargs.get("axes", -1)
    scale_ratio = 1 if kwargs.get("scale_ratio", 1) == 0 else kwargs.get("scale_ratio", 1)
    clip2 = 4 if kwargs.get("clip2", 4) == 0 else kwargs.get("clip2", 4)
    clip3 = 2 if kwargs.get("clip3", 2) == 0 else kwargs.get("clip3", 2)
    max_levels = kwargs.get("max_levels", 0)
    max_lv3 = max_levels[MAX_LV3_IDX] if MAX_LV3_IDX in max_levels.keys() else 0
    max_lv2 = max_levels[MAX_LV2_IDX] if MAX_LV2_IDX in max_levels.keys() else 0
    max_lv1 = max_levels[MAX_LV1_IDX] if MAX_LV1_IDX in max_levels.keys() else 0

    # The algorithm allows for subsequent local scaling to shrink by up to 4 times
    # (a 2x reduction each at Level 2 and Level 3),
    #  resulting in $7 / 4 = 1.75$. This ensures that after all the scaling,
    # the largest value will fall within the range $[0, 2)$,
    # facilitating the quantization of the tail.
    div7 = torch.ones_like(max_lv1) / (7.0 * scale_ratio)
    div7 = div7.to(torch.bfloat16).to(torch.float32)
    scale_factor = max_lv1 * div7
    # Prevent abnormal boundary conditions caused by max_lv1 being 0
    scale_factor = scale_factor.to(torch.bfloat16).to(torch.float32)
    scale_factor = torch.clip(scale_factor, min=2 ** (-48), max=49152)
    # change scale_factor to bf16(rint) Simulating Hardware Behavior
    e_sf = torch.floor(torch.log2(scale_factor))
    mant_sf = scale_factor / (2**e_sf) * (2**7)
    scale_factor = torch.round(mant_sf) / (2**7) * (2**e_sf)
    # scale_factor to e6m2 (8-bit float: 1 sign, 6 ebits, 2 mbits)
    e_sf = torch.floor(torch.log2(scale_factor))
    scale_factor = torch.round(scale_factor * torch.exp2(2 - e_sf)) * torch.exp2(e_sf - 2)
    # Computing the local dynamic shift
    rec_sf = (1.0 / scale_factor).to(torch.bfloat16).to(torch.float32)
    # level2 scaling
    scale_lv2 = max_lv2 * rec_sf
    # 1 bit is allocated to determine whether the sub-block needs to be downscaled by a factor of 2.
    scale_lv2 = torch.exp2((scale_lv2.clip(0, clip2) / clip2).floor())

    # level3 scaling, same as level2
    scale_lv3 = torch.exp2(((max_lv3 * rec_sf / scale_lv2).clip(0, clip3) / clip3).floor())
    return QParam(
        scheme=QScheme(
            dtype=q_dtype,
            scope=q_scope,
            symmetric=symmetric,
        ),
        ext={
            "weight_scale_lv3": scale_lv3,
            "weight_scale_lv2": scale_lv2,
            "weight_scale_factor": scale_factor,
            "axes": axes,
        },
    )


@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.HiF4, QScope.PER_BLOCK, True), api_name="quantize")
def hif4_per_block_quantize(tensor: QStorage, q_param: QParam) -> QStorage:
    hifx_finfo = q_param.scheme.dtype.hifx_finfo
    inp = tensor.value
    dtype = inp.dtype

    # pad to fit block size
    scale_lv3 = q_param.ext['weight_scale_lv3']
    scale_lv2 = q_param.ext['weight_scale_lv2']
    scale_factor = q_param.ext['weight_scale_factor']

    rec_sf = (1.0 / scale_factor).to(torch.bfloat16).to(dtype)
    # mantissa quantification
    inp_sign = torch.sign(inp)
    inp = torch.abs(inp / scale_lv2 / scale_lv3 * rec_sf)
    # Map the value to the specified fractional precision mbits (3)
    mant = torch.floor(inp * 2 ** (hifx_finfo.mbits - 1) + 0.5) / 2 ** (hifx_finfo.mbits - 1)
    mant[mant >= 2] = 2 - 2 ** (-hifx_finfo.mbits + 1)

    inp = (inp_sign * mant).to(dtype)
    tensor_q = tensor.same_like(inp).to(q_param.scheme.dtype)
    return tensor_q


@QFuncRegistry.register(dispatch_key=(QDType.HiF4, QDType.HiF4, QScope.PER_BLOCK, True), api_name="dequantize")
@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.HiF4, QScope.PER_BLOCK, True), api_name="dequantize")
def hif4_per_block_dequantize(tensor: QStorage, q_param: QParam) -> QStorage:
    scale_lv3 = q_param.ext['weight_scale_lv3']
    scale_lv2 = q_param.ext['weight_scale_lv2']
    scale_factor = q_param.ext['weight_scale_factor']
    quant_inp = tensor.value
    dtype = quant_inp.dtype
    inp = quant_inp * scale_lv3 * scale_lv2 * scale_factor

    tensor_q = tensor.same_like(inp).to(dtype)
    return tensor_q
