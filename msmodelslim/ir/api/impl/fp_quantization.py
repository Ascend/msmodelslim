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

import torch

from msmodelslim.ir.qal import QStorage
from msmodelslim.ir.qal.qbase import QDType, QScope, QParam, QScheme
from msmodelslim.ir.qal.qregistry import QFuncRegistry
from msmodelslim.utils.exception import SchemaValidateError


# FP8_E4M3的量化范围为[-448, 448]
FP8_E4M3_MAX = 448
FP8_E4M3_MIN = -448


@QFuncRegistry.register(dispatch_key=(QDType.FP8_E4M3, QScope.PER_TOKEN, True), api_name="calculate_qparam")
@QFuncRegistry.register(dispatch_key=(QDType.FP8_E4M3, QScope.PER_CHANNEL, True), api_name="calculate_qparam")
@QFuncRegistry.register(dispatch_key=(QDType.FP8_E4M3, QScope.PER_TENSOR, True), api_name="calculate_qparam")
@QFuncRegistry.register(dispatch_key=(QDType.FP8_E4M3, QScope.PER_HEAD, True), api_name="calculate_qparam")
def calculate_fp8_qparam(
    min_val: torch.Tensor, max_val: torch.Tensor, q_dtype: QDType, q_scope: QScope, symmetric: bool, **kwargs
) -> QParam:
    amax = torch.maximum(min_val.abs(), max_val.abs())
    scale = amax.clamp(min=1e-12) / FP8_E4M3_MAX
    offset = torch.zeros_like(scale)

    return QParam(
        scheme=QScheme(
            dtype=q_dtype,
            scope=q_scope,
            symmetric=symmetric,
        ),
        ext={"scale": scale, "offset": offset},
    )


@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.FP8_E4M3, QScope.PER_CHANNEL, True), api_name="quantize")
@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.FP8_E4M3, QScope.PER_TOKEN, True), api_name="quantize")
@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.FP8_E4M3, QScope.PER_TENSOR, True), api_name="quantize")
def fp8_quantize(tensor: QStorage, q_param: QParam) -> QStorage:
    scale = q_param.ext["scale"]
    if scale is None:
        raise SchemaValidateError("scale is None", action="Please check the q_param")
    offset = q_param.ext["offset"] if "offset" in q_param.ext else torch.zeros_like(scale)
    inplace = q_param.ext.get("inplace", False)
    input_tensor = tensor.value
    if inplace:
        input_tensor = input_tensor.div_(scale).add_(offset)
    else:
        input_tensor = input_tensor / scale + offset

    q_dtype = q_param.scheme.dtype
    if q_dtype not in [QDType.FP8_E4M3]:  # 使用 [] 便于后续扩展
        raise TypeError("q_param.scheme.dtype: {} is not in [QDType.FP8_E4M3]".format(q_dtype))

    input_tensor = input_tensor.clamp(min=FP8_E4M3_MIN, max=FP8_E4M3_MAX)

    return tensor.same_like(input_tensor).to(q_dtype)


@QFuncRegistry.register(
    dispatch_key=(QDType.FP8_E4M3, QDType.FP8_E4M3, QScope.PER_CHANNEL, True), api_name="dequantize"
)
@QFuncRegistry.register(dispatch_key=(QDType.FP8_E4M3, QDType.FP8_E4M3, QScope.PER_TOKEN, True), api_name="dequantize")
@QFuncRegistry.register(dispatch_key=(QDType.FP8_E4M3, QDType.FP8_E4M3, QScope.PER_TENSOR, True), api_name="dequantize")
def fp8_dequantize(tensor: QStorage, q_param: QParam) -> QStorage:
    scale = q_param.ext["scale"]
    offset = q_param.ext["offset"] if "offset" in q_param.ext else torch.zeros_like(scale)
    inplace = q_param.ext.get("inplace", False)
    input_tensor = tensor.value
    if inplace:
        input_tensor = input_tensor.sub_(offset).mul_(scale)
    else:
        input_tensor = (input_tensor.to(offset.dtype) - offset) * scale
    return tensor.same_like(input_tensor).to(QDType.FLOAT)
