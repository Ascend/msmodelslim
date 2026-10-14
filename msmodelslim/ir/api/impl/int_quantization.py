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

from msmodelslim.ir.qal import QStorage
from msmodelslim.ir.qal.qbase import QDType, QScope, QParam, QScheme
from msmodelslim.ir.qal.qregistry import QFuncRegistry
from msmodelslim.utils.exception import SchemaValidateError


@QFuncRegistry.register(dispatch_key=(QDType.INT8, QScope.PER_TOKEN, False), api_name="calculate_qparam")
@QFuncRegistry.register(dispatch_key=(QDType.INT8, QScope.PER_TOKEN, True), api_name="calculate_qparam")
@QFuncRegistry.register(dispatch_key=(QDType.INT8, QScope.PER_CHANNEL, False), api_name="calculate_qparam")
@QFuncRegistry.register(dispatch_key=(QDType.INT8, QScope.PER_CHANNEL, True), api_name="calculate_qparam")
@QFuncRegistry.register(dispatch_key=(QDType.INT8, QScope.PER_TENSOR, False), api_name="calculate_qparam")
@QFuncRegistry.register(dispatch_key=(QDType.INT8, QScope.PER_TENSOR, True), api_name="calculate_qparam")
@QFuncRegistry.register(dispatch_key=(QDType.INT8, QScope.PER_HEAD, True), api_name="calculate_qparam")
@QFuncRegistry.register(dispatch_key=(QDType.INT4, QScope.PER_CHANNEL, True), api_name="calculate_qparam")
@QFuncRegistry.register(dispatch_key=(QDType.INT4, QScope.PER_CHANNEL, False), api_name="calculate_qparam")
@QFuncRegistry.register(dispatch_key=(QDType.INT4, QScope.PER_TOKEN, False), api_name="calculate_qparam")
@QFuncRegistry.register(dispatch_key=(QDType.INT4, QScope.PER_TOKEN, True), api_name="calculate_qparam")
def calculate_int_qparam(
    min_val: torch.Tensor, max_val: torch.Tensor, q_dtype: QDType, q_scope: QScope, symmetric: bool, **kwargs
) -> QParam:
    eps = torch.tensor([torch.finfo(torch.float32).eps]).type_as(min_val)
    min_val = torch.min(min_val, torch.zeros_like(min_val))
    max_val = torch.max(max_val, torch.zeros_like(max_val))
    q_signed = kwargs.get("q_signed", True)
    integral_zero_point = kwargs.get("integral_zero_point", True)
    max_bound = kwargs.get("max_bound", None)  # for w4a8   max_bound=119

    if q_dtype == QDType.INT4:
        quant_bitwidth = 4
    elif q_dtype == QDType.INT8:
        quant_bitwidth = 8
    else:
        raise TypeError("q_dtype: {} is not in [QDType.INT4, QDType.INT8]".format(q_dtype))

    if not symmetric:
        max_bound = 2**quant_bitwidth - 1 if max_bound is None else max_bound
        # asymmetric quantization
        scale = max_val / max_bound - min_val / max_bound
        scale = torch.max(scale, eps)
        offset = -1 * min_val / scale
        if integral_zero_point:
            if isinstance(offset, torch.Tensor):
                offset = offset.round()
            else:
                offset = float(round(offset))

        if q_signed:
            qmin = -1 * (2 ** (quant_bitwidth - 1))
            offset += qmin

    else:
        # symmetric quantization
        max_bound = 2 ** (quant_bitwidth - 1) - 1 if max_bound is None else max_bound
        max_val_pos = torch.max(-min_val, max_val)
        scale = max_val_pos / float(max_bound)
        scale = torch.max(scale, eps)
        offset = torch.tensor(0.0).type_as(min_val).expand(scale.shape)

    return QParam(
        scheme=QScheme(
            dtype=q_dtype,
            scope=q_scope,
            symmetric=symmetric,
        ),
        ext={"scale": scale, "offset": offset},
    )


@QFuncRegistry.register(dispatch_key=(QDType.INT8, QScope.PER_GROUP, False), api_name="calculate_qparam")
@QFuncRegistry.register(dispatch_key=(QDType.INT8, QScope.PER_GROUP, True), api_name="calculate_qparam")
@QFuncRegistry.register(dispatch_key=(QDType.INT4, QScope.PER_GROUP, False), api_name="calculate_qparam")
@QFuncRegistry.register(dispatch_key=(QDType.INT4, QScope.PER_GROUP, True), api_name="calculate_qparam")
def int_per_group_param(
    min_val: torch.Tensor, max_val: torch.Tensor, q_dtype: QDType, q_scope: QScope, symmetric: bool, **kwargs
) -> QParam:
    q_param = calculate_int_qparam(min_val, max_val, q_dtype, q_scope, symmetric, **kwargs)
    # 此处无法获取group_size，需要在外层填充group_size值
    q_param.ext['group_size'] = -1
    return q_param


@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.INT8, QScope.PER_CHANNEL, True), api_name="quantize")
@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.INT8, QScope.PER_CHANNEL, False), api_name="quantize")
@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.INT8, QScope.PER_TOKEN, False), api_name="quantize")
@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.INT8, QScope.PER_TOKEN, True), api_name="quantize")
@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.INT8, QScope.PER_TENSOR, False), api_name="quantize")
@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.INT8, QScope.PER_TENSOR, True), api_name="quantize")
@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.INT4, QScope.PER_CHANNEL, True), api_name="quantize")
@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.INT4, QScope.PER_CHANNEL, False), api_name="quantize")
@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.INT4, QScope.PER_TOKEN, False), api_name="quantize")
@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.INT4, QScope.PER_TOKEN, True), api_name="quantize")
@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.INT4, QScope.PER_TENSOR, False), api_name="quantize")
@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.INT4, QScope.PER_TENSOR, True), api_name="quantize")
def int_quantize(tensor: QStorage, q_param: QParam) -> QStorage:
    scale = q_param.ext["scale"]
    offset = q_param.ext["offset"] if "offset" in q_param.ext else torch.zeros_like(scale)
    inplace = q_param.ext.get("inplace", False)
    max_bound = q_param.ext.get("max_bound", False)
    input_tensor = tensor.value
    if inplace:
        input_tensor = input_tensor.div_(scale).add_(offset).round_()
    else:
        input_tensor = (input_tensor / scale + offset).round_()
    if max_bound:
        input_tensor = input_tensor.clamp_(min=-max_bound, max=max_bound)
    # int4与int8 input_tensor.clamp_范围不同
    q_dtype = q_param.scheme.dtype
    if q_dtype not in [QDType.INT4, QDType.INT8]:
        raise TypeError("q_param.scheme.dtype: {} is not in [QDType.INT4, QDType.INT8]".format(q_dtype))
    if q_dtype == QDType.INT4:
        quant_bitwidth = 4
    elif q_dtype == QDType.INT8:
        quant_bitwidth = 8
    else:
        raise ValueError("q_param.scheme.dtype must be INT4 or INT8")

    max_bound = 2 ** (quant_bitwidth - 1) - 1
    min_bound = -max_bound - 1
    input_tensor = input_tensor.clamp_(min=min_bound, max=max_bound)

    return tensor.same_like(input_tensor).to(q_dtype)


@QFuncRegistry.register(dispatch_key=(QDType.INT8, QDType.INT8, QScope.PER_CHANNEL, True), api_name="dequantize")
@QFuncRegistry.register(dispatch_key=(QDType.INT8, QDType.INT8, QScope.PER_CHANNEL, False), api_name="dequantize")
@QFuncRegistry.register(dispatch_key=(QDType.INT8, QDType.INT8, QScope.PER_TOKEN, False), api_name="dequantize")
@QFuncRegistry.register(dispatch_key=(QDType.INT8, QDType.INT8, QScope.PER_TOKEN, True), api_name="dequantize")
@QFuncRegistry.register(dispatch_key=(QDType.INT8, QDType.INT8, QScope.PER_TENSOR, False), api_name="dequantize")
@QFuncRegistry.register(dispatch_key=(QDType.INT8, QDType.INT8, QScope.PER_TENSOR, True), api_name="dequantize")
@QFuncRegistry.register(dispatch_key=(QDType.INT4, QDType.INT4, QScope.PER_CHANNEL, True), api_name="dequantize")
@QFuncRegistry.register(dispatch_key=(QDType.INT4, QDType.INT4, QScope.PER_CHANNEL, False), api_name="dequantize")
@QFuncRegistry.register(dispatch_key=(QDType.INT4, QDType.INT4, QScope.PER_TOKEN, False), api_name="dequantize")
@QFuncRegistry.register(dispatch_key=(QDType.INT4, QDType.INT4, QScope.PER_TOKEN, True), api_name="dequantize")
@QFuncRegistry.register(dispatch_key=(QDType.INT4, QDType.INT4, QScope.PER_TENSOR, False), api_name="dequantize")
@QFuncRegistry.register(dispatch_key=(QDType.INT4, QDType.INT4, QScope.PER_TENSOR, True), api_name="dequantize")
def int_dequantize(tensor: QStorage, q_param: QParam) -> QStorage:
    scale = q_param.ext["scale"]
    offset = q_param.ext["offset"] if "offset" in q_param.ext else torch.zeros_like(scale)
    inplace = q_param.ext.get("inplace", False)
    input_tensor = tensor.value
    if inplace:
        input_tensor = input_tensor.sub_(offset).mul_(scale)
    else:
        input_tensor = (input_tensor - offset) * scale
    return tensor.same_like(input_tensor).to(QDType.FLOAT)


@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.INT8, QScope.PER_GROUP, False), api_name="quantize")
@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.INT8, QScope.PER_GROUP, True), api_name="quantize")
@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.INT4, QScope.PER_GROUP, False), api_name="quantize")
@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.INT4, QScope.PER_GROUP, True), api_name="quantize")
def int8_per_group_quantize(tensor: QStorage, q_param: QParam) -> QStorage:
    group_size = q_param.ext.get("group_size", -1)
    if group_size < 0:
        raise SchemaValidateError(
            f"group quantize group_size must be greater than 0 but got group_size = {group_size}",
            action="Please make sure group_size is greater than 0",
        )
    tmp_tensor = tensor.value.t()
    org_shape = tmp_tensor.shape
    tensor_reshaped = tmp_tensor.reshape(-1, group_size)
    q_param_reshaped = QParam(
        scheme=q_param.scheme,
        ext={
            "scale": q_param.ext["scale"].view(-1, 1),
            "offset": q_param.ext["offset"].view(-1, 1),
            "group_size": group_size,
        },
    )
    tensor_q = int_quantize(tensor.same_like(tensor_reshaped), q_param_reshaped)
    tensor_q.value = tensor_q.value.reshape(org_shape)
    tensor_q.value = tensor_q.value.t()
    return tensor_q


@QFuncRegistry.register(dispatch_key=(QDType.INT8, QDType.INT8, QScope.PER_GROUP, False), api_name="dequantize")
@QFuncRegistry.register(dispatch_key=(QDType.INT8, QDType.INT8, QScope.PER_GROUP, True), api_name="dequantize")
@QFuncRegistry.register(dispatch_key=(QDType.INT4, QDType.INT4, QScope.PER_GROUP, False), api_name="dequantize")
@QFuncRegistry.register(dispatch_key=(QDType.INT4, QDType.INT4, QScope.PER_GROUP, True), api_name="dequantize")
def int_per_group_dequantize(tensor: QStorage, q_param: QParam) -> QStorage:
    group_size = q_param.ext.get("group_size", -1)
    if group_size < 0:
        raise SchemaValidateError(
            f"group quantize group_size must be greater than 0 but got group_size = {group_size}",
            action="Please make sure group_size is greater than 0",
        )
    tmp_tensor = tensor.value.t()
    org_shape = tmp_tensor.shape
    tensor_reshaped = tmp_tensor.reshape(-1, group_size)
    q_param_reshaped = QParam(
        scheme=q_param.scheme,
        ext={
            "scale": q_param.ext["scale"].view(-1, 1),
            "offset": q_param.ext["offset"].view(-1, 1),
            "group_size": group_size,
        },
    )
    tensor_q = int_dequantize(tensor.same_like(tensor_reshaped), q_param_reshaped)
    tensor_q.value = tensor_q.value.reshape(org_shape)
    tensor_q.value = tensor_q.value.t()
    return tensor_q


def reshape_pad_tensor_by_group_size(data: torch.Tensor, group_size: int):
    """Reshapes and pads the tensor to ensure that it can be quantized in groups of `group_size`.

    This function adjusts the
    input tensor's shape so that its last dimension is a multiple
    of the specified `group_size`. If padding is required, it adds padding to the tensor
    to achieve this. If the tensor's last dimension is already divisible by `group_size`,
    no padding is applied.

    Args:
        data (torch.Tensor): The input tensor to be reshaped and padded.
        group_size (int): The size of the groups that the tensor should be reshaped into.

    Returns:
        torch.Tensor: The reshaped and padded tensor, if necessary.
        tuple: The original shape of the input tensor.
        int: The padding length applied to the tensor. Returns 0 if no padding is applied.
    """
    orig_shape = data.shape
    pad_len = 0
    if group_size == 0:
        data = data.reshape(1, -1)
        return data, orig_shape, pad_len
    if len(data.shape) > 2:
        data = data.reshape(-1, orig_shape[-1])
    if group_size == -1 or data.shape[1] < group_size:
        return data, orig_shape, pad_len
    elif data.shape[1] % group_size == 0:
        data = data.reshape(-1, group_size)
        return data, orig_shape, pad_len
    else:
        pad_len = (data.shape[1] + group_size - 1) // group_size * group_size - data.shape[1]
        data_new = torch.nn.functional.pad(data, (0, pad_len))
        data_new = data_new.reshape(-1, group_size)
        return data_new, orig_shape, pad_len


def revert_tensor_by_pad(data: torch.Tensor, orig_shape: tuple, pad_len: int):
    """Reverts the tensor to its original shape by removing padding.

    This function removes the padding added during reshaping and returns the tensor to
    its original shape.

    Args:
        data (torch.Tensor): The reshaped and possibly padded tensor.
        orig_shape (tuple): The original shape of the tensor before reshaping.
        pad_len (int): The length of the padding to be removed.

    Returns:
        torch.Tensor: The tensor restored to its original shape.
    """
    if pad_len == 0:
        return data.reshape(orig_shape)
    else:
        if len(orig_shape) > 2:
            tmp_shape = torch.prod(torch.tensor(orig_shape[:-1])).item()
        else:
            tmp_shape = orig_shape[0]
        data_new = data.reshape(tmp_shape, -1)
        data_new = data_new[:, :-pad_len]
        data_new = data_new.reshape(orig_shape)
        return data_new
