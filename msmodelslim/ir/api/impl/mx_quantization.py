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
#  Adapted from https://github.com/microsoft/microxcaling/blob/main/mx_ops.py

import torch

from msmodelslim.ir.qal import QStorage
from msmodelslim.ir.qal.qbase import QDType, QScope, QParam, QScheme
from msmodelslim.ir.qal.qregistry import QFuncRegistry

FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)


@QFuncRegistry.register(dispatch_key=(QDType.MXFP8, QScope.PER_BLOCK, True), api_name="calculate_qparam")
@QFuncRegistry.register(dispatch_key=(QDType.MXFP8, QScope.PER_CHANNEL, True), api_name="calculate_qparam")
def calculate_mx_qparam(
    min_val: torch.Tensor, max_val: torch.Tensor, q_dtype: QDType, q_scope: QScope, symmetric: bool, **kwargs
) -> QParam:
    mx_finfo = q_dtype.mx_finfo
    is_flush_fp32_subnorms = mx_finfo.flush_fp32_subnorms

    shared_exp = torch.floor(torch.log2(max_val + FP32_MIN_NORMAL * (max_val == 0).to(max_val.dtype)))
    shared_exp = shared_exp - mx_finfo.emax

    if is_flush_fp32_subnorms:
        # 标记需要保留的 shared_exp (bool mask)，调用方可据此清零 A
        keep_mask = shared_exp > -FP32_EXPONENT_BIAS
    else:
        keep_mask = None

    scale_emax = 2 ** (mx_finfo.scale_bits - 1) - 1
    shared_exp[shared_exp > scale_emax] = float("NaN")
    shared_exp[shared_exp < -scale_emax] = -scale_emax

    return QParam(
        scheme=QScheme(
            dtype=q_dtype,
            scope=q_scope,
            symmetric=symmetric,
        ),
        ext={"scale": shared_exp, "offset": torch.zeros_like(shared_exp), "keep_mask": keep_mask},
    )


@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.MXFP8, QScope.PER_BLOCK, True), api_name="quantize")
@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.MXFP8, QScope.PER_CHANNEL, True), api_name="quantize")
def mxfp_per_block_quantize(tensor: QStorage, q_param: QParam) -> QStorage:
    mx_finfo = q_param.scheme.dtype.mx_finfo
    inp = tensor.value
    dtype = inp.dtype

    inp = inp.to(torch.float32)
    shared_exp = q_param.ext['scale']
    keep_mask = q_param.ext.get('keep_mask', None)

    if keep_mask is not None:
        inp = inp * keep_mask.to(inp.dtype)

    # 0 值保持为 0（shared_exp 含 NaN/±Inf 时 0/scale 会产生非有限值）。
    # 用 0 维标量广播替代 torch.zeros_like 的整份分配，避免一个全量 fp32 临时。
    inp = torch.where(inp == 0, torch.zeros((), dtype=inp.dtype, device=inp.device), inp / (2**shared_exp))

    # 用 1B/elem 的布尔掩码记录非有限值位置（与原实现此处整份 fp32 clone 语义一致，
    # 但显存占用降到 1/4），供 clamp 后恢复 ±Inf。
    zero_mask = inp == 0
    pos_inf_mask = torch.isposinf(inp)
    neg_inf_mask = torch.isneginf(inp)

    private_exp = torch.floor(torch.log2(torch.abs(inp) + zero_mask.to(inp.dtype)))

    inp = _quant(inp, mx_finfo.mbits, private_exp, mx_finfo.ebits)
    inp = torch.clamp(inp, min=-mx_finfo.max_norm, max=mx_finfo.max_norm)
    # 恢复量化前的非有限值（对应原实现 _clamp_out(inp, inp_, ...) 的恢复逻辑；
    # NaN 不恢复：与原实现 a == NaN 比较恒为 False 的行为一致）
    inp[pos_inf_mask] = float("Inf")
    inp[neg_inf_mask] = -float("Inf")

    inp = inp.to(dtype)

    tensor_q = tensor.same_like(inp).to(q_param.scheme.dtype)
    return tensor_q


@QFuncRegistry.register(dispatch_key=(QDType.MXFP8, QDType.MXFP8, QScope.PER_BLOCK, True), api_name="dequantize")
@QFuncRegistry.register(dispatch_key=(QDType.MXFP4, QDType.MXFP4, QScope.PER_BLOCK, True), api_name="dequantize")
@QFuncRegistry.register(dispatch_key=(QDType.MXFP8, QDType.MXFP8, QScope.PER_CHANNEL, True), api_name="dequantize")
def mxfp_per_block_dequantize(tensor: QStorage, q_param: QParam) -> QStorage:
    shared_exp = q_param.ext['scale']
    quant_inp = tensor.value
    dtype = quant_inp.dtype
    inp = quant_inp * (2**shared_exp)
    tensor_q = tensor.same_like(inp).to(dtype)
    return tensor_q


def _quant(a, bits, exp, exp_bits):
    # +2 的偏移是为了计算 max_norm ，此处需要进行加减
    min_exp = -(2 ** (exp_bits - 1)) + 2
    exp = exp.clip(min=min_exp)
    bits_ = bits - 2

    if exp is None:  # 私有指数为空
        a = a * (2**bits_)
        a = torch.sign(a) * torch.floor(torch.abs(a) + 0.5)
        a = a / (2**bits_)
    else:
        a = a / (2**exp) * (2**bits_)
        a = torch.sign(a) * torch.floor(torch.abs(a) + 0.5)
        a = a / (2**bits_) * (2**exp)
    return a


@QFuncRegistry.register(dispatch_key=(QDType.MXFP4, QScope.PER_BLOCK, True), api_name="calculate_qparam")
def calculate_mxfp4_qparam(
    min_val: torch.Tensor, max_val: torch.Tensor, q_dtype: QDType, q_scope: QScope, symmetric: bool, **kwargs
) -> QParam:
    """
    Calculate the quantization parameters for MXFP4.
    scale = 2 ** (floor(log2(max(abs(x)) * 8/7 + 9.6e-7)) - 2)
    """
    mx_finfo = q_dtype.mx_finfo
    man_shift_bit = mx_finfo.mbits - 2
    shared_exp = torch.floor(torch.log2(max_val / (1 - 0.5 ** (man_shift_bit + 2)) + 9.6e-7))
    shared_exp = shared_exp - mx_finfo.emax
    scale_emax = 2 ** (mx_finfo.scale_bits - 1) - 1
    shared_exp = torch.clip(shared_exp, -scale_emax - mx_finfo.emax, scale_emax - mx_finfo.emax)

    return QParam(
        scheme=QScheme(
            dtype=q_dtype,
            scope=q_scope,
            symmetric=symmetric,
        ),
        ext={
            "scale": shared_exp,
        },
    )


@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.MXFP4, QScope.PER_BLOCK, True), api_name="quantize")
def mxfp4_quantize(tensor: QStorage, q_param: QParam) -> QStorage:
    mx_finfo = q_param.scheme.dtype.mx_finfo
    inp = tensor.value
    dtype = inp.dtype

    inp = inp.to(torch.float32)
    shared_exp = q_param.ext['scale']

    x_unsigned = torch.abs(inp)
    sign = torch.sign(inp)
    x_biased = x_unsigned / torch.exp2(shared_exp)
    private_exp_with_bias = torch.floor(torch.log2(x_biased + 9.6e-7))
    min_exp = -(2 ** (mx_finfo.ebits - 1)) + 2
    private_exp = torch.clip(private_exp_with_bias, min_exp, mx_finfo.emax)

    man_shift_bit = mx_finfo.mbits - 2
    mant_lshifted = x_biased / (2**private_exp) * (2**man_shift_bit)
    mant_trunc = torch.floor(mant_lshifted + 0.5)
    fp_value = mant_trunc / (2**man_shift_bit) * (2**private_exp)

    fp_val_max = 2**mx_finfo.emax * float(2 ** (man_shift_bit + 1) - 1) / 2**man_shift_bit
    fp_val_max = min(fp_val_max, 1e38)
    fp_value = torch.clip(fp_value, -fp_val_max, fp_val_max)

    ind_nan = shared_exp < -127
    fp_value[torch.broadcast_to(ind_nan, fp_value.shape)] = 0

    out = sign * fp_value
    out = out.to(dtype)

    tensor_q = tensor.same_like(out).to(q_param.scheme.dtype)
    return tensor_q
