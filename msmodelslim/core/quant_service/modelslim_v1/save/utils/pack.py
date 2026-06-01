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


def _pack_int4(weight) -> torch.Tensor:
    """
    Pack int4 weight to int8 weight
    @param weight: torch.Tensor, int4 weight
    @return: torch.Tensor, int8 weight
    """
    weight = weight.to(torch.int8)
    e = 0  # number of experts
    if len(weight.shape) == 2:
        k, n = weight.shape
    elif len(weight.shape) == 3:
        e, k, n = weight.shape
    else:
        raise AssertionError("weight dimension should be 2 or 3")
    n_new = n // 2 + n % 2

    if n_new != n // 2:
        raise AssertionError("n dimension should be even")

    weight = weight.reshape(-1, 2)
    weight0 = weight[:, :1]
    weight1 = weight[:, 1:]

    weight1_4 = torch.bitwise_left_shift(weight1, 4)
    weight2_4 = weight0 & 0b00001111

    weight_add = torch.bitwise_or(weight1_4, weight2_4)
    if e == 0:
        weight_res = weight_add.reshape(k, n_new)
    else:
        weight_res = weight_add.reshape(e, k, n_new)
    return weight_res


def w4a8_pack_int4(save_quant_weight):
    """
    Pack int4 weight to int8 weight
    @param save_quant_weight: torch.Tensor, int4 weight
    @return: torch.Tensor, int8 weight
    """
    weight = save_quant_weight.transpose(-1, -2).contiguous()
    packed_weight_tensor = _pack_int4(weight)
    packed_weight_tensor = packed_weight_tensor.transpose(-1, -2).contiguous()
    return packed_weight_tensor


def process_scale(name, bias, tp_num):
    """
    Pack int4 weight to int8 weight
    @param name: 输入tensor名
    @param bias: sum 前bias
    @param tp_num: 推理时tp数
    @return: bias, fp32格式gmm算子所需的偏置量
    """
    if any(char in name for char in ['up_proj', 'gate_proj', 'q_proj', 'k_proj', 'v_proj']):
        up_bias = bias
        up_bias = 8 * up_bias.sum(dim=1, keepdim=True)
        bias = up_bias

    elif any(char in name for char in ['down_proj', 'o_proj']):
        pre_shape = bias.shape[0]
        sum_shape = bias.shape[1] // tp_num
        down_bias = bias.reshape(-1, sum_shape)
        down_bias = 8 * down_bias.sum(dim=1, keepdim=True)
        bias = down_bias.reshape(pre_shape, -1)
    return bias


def pack_fp4_to_uint8(x: torch.Tensor) -> torch.Tensor:
    FLOAT_TO_E2M1 = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
    ]
    m, n = x.shape
    device = x.device
    # Create lookup table for FP4 values to indices
    # Map the absolute values to 0-7 indices
    kE2M1 = torch.tensor(FLOAT_TO_E2M1, device=device, dtype=x.dtype)
    # Find closest valid FP4 value index for each element
    abs_x = torch.abs(x)
    abs_diff_x = torch.abs(abs_x.unsqueeze(-1) - kE2M1)  # [m, n, 8]
    abs_indices = torch.argmin(abs_diff_x, dim=-1)  # [m, n]
    # Apply sign bit (bit 3) to get final 4-bit representation
    indices = abs_indices + (torch.signbit(x).to(torch.long) << 3)
    # Reshape to prepare for packing pairs of values
    indices = indices.reshape(-1)
    # Handle odd length by padding if necessary
    if indices.numel() % 2 != 0:
        indices = torch.cat([indices, torch.zeros(1, dtype=torch.long, device=device)])
    # Reshape to pair consecutive elements
    indices = indices.reshape(-1, 2)
    # Pack pairs of 4-bit values into 8-bit values
    packed = (indices[:, 0] | (indices[:, 1] << 4)).to(torch.uint8)
    return packed.reshape(m, n // 2)
