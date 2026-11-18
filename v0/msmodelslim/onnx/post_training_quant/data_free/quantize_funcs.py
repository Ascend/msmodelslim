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
from collections import namedtuple

import numpy as np

from msmodelslim.onnx.post_training_quant.data_free.flip import Flip


def get_quant_param(
    bit,
    x_min,
    x_max,
    q_signed=True,
    sym=False,
):
    """
    Compute the scaling factor and zeropoint with the given quantization range.
    x_min: lower bound for quantization range
    x_max: upper bound for quantization range
    """
    eps = 1e-8
    q_max = 2 ** (bit - 1) - 1
    q_min = -2 ** (bit - 1)
    if (q_max - q_min) == 0:
        raise ValueError("Quantization bit can not be 1, please check.")
    if sym:  # symmetric quantization
        q_min = -q_max
        max_val_pos = np.maximum(-x_min, x_max)
        scale = max_val_pos / (float(q_max - q_min) / 2)
        scale = np.maximum(scale, eps)
        zero_point = float((q_max + q_min) // 2)
    else:  # asymmetric quantization
        scale = (x_max - x_min) / (q_max - q_min)
        scale = max(scale, eps)
        zero_point = -1 * x_min / scale
        zero_point = round(zero_point, 4)

        if isinstance(zero_point, np.ndarray):
            zero_point = zero_point.round()
        else:
            zero_point = float(round(zero_point))

        if q_signed:
            zero_point += q_min

    return scale, zero_point


def linear_quantize(data: np.ndarray,
                    scale: np.ndarray,
                    offset: np.ndarray):
    """
    Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.
    data: input data to be quantized
    scale: scale for quantization
    offset: offset for quantization
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if np.any(scale == 0):
        raise ValueError("scale contains zero values. Please check quantization parameters.")
    if len(data.shape) == 4:
        scale = scale.reshape((-1, 1, 1, 1))
    # reshape scale and zeropoint for linear weights
    elif len(data.shape) == 2:
        scale = scale.reshape((-1, 1))
    return data / scale + offset


ElementFlip = namedtuple('ElementFlip', 'data, error, priority, order')


def squant_func(rounding_error_sum, rounding_number, rounding_error, flip_up: Flip, flip_down: Flip):
    shape = rounding_number.shape
    flip_up.reshape(shape)
    flip_down.reshape(shape)
    for oc_idx in range(shape[0]):  # out channel
        for ic_idx in range(shape[1]):  # in channel
            ele_flip_up = ElementFlip(flip_up.data[oc_idx][ic_idx], flip_up.error[oc_idx][ic_idx],
                                      flip_up.priority[oc_idx][ic_idx], flip_up.order[oc_idx][ic_idx])
            ele_flip_down = ElementFlip(flip_down.data[oc_idx][ic_idx], flip_down.error[oc_idx][ic_idx],
                                        flip_down.priority[oc_idx][ic_idx], flip_up.order[oc_idx][ic_idx])
            round_func(rounding_error_sum[oc_idx][ic_idx], rounding_number[oc_idx][ic_idx],
                       rounding_error[oc_idx][ic_idx], ele_flip_up, ele_flip_down)


def round_func(rounding_error_sum, rounding_number_, rounding_error_, ele_flip_up, ele_flip_down):
    if rounding_error_sum < 0:
        number_ = ele_flip_up.data
        error_ = ele_flip_up.error
        priority_ = ele_flip_up.priority
        order_ = ele_flip_up.order
        priority_1 = ele_flip_down.priority
    else:
        number_ = ele_flip_down.data
        error_ = ele_flip_down.error
        priority_ = ele_flip_down.priority
        order_ = ele_flip_down.order
        priority_1 = ele_flip_up.priority

    rounding_error_sum = abs(rounding_error_sum)
    topk = int(rounding_error_sum)
    # 添加边界检查，确保 topk 不超过 order_ 数组的长度
    topk = min(topk, len(order_))

    idx_ = list(order_[0:topk].astype(int))
    rounding_error_[idx_] = error_[idx_]
    rounding_number_[idx_] = number_[idx_]

    over_squant = (topk >= rounding_error_sum)
    if over_squant:
        idx_c = order_[topk - 1].astype(int)
        priority_1[idx_c] = abs(rounding_error_[idx_c])
    else:
        idx_c = order_[topk].astype(int)
        priority_[idx_c] = abs(rounding_error_[idx_c])
