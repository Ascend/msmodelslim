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
# pylint: disable=consider-using-from-import,unnecessary-list-index-lookup

import logging

import mindspore.nn as nn
import numpy as np
from mindspore import Tensor
from mindspore.common import dtype as mstype

try:
    from mindspore.nn.transformer.layers import _Linear
except ModuleNotFoundError:
    try:
        from mindspore.nn.layer.transformer import _Linear
    except ImportError:
        from mindspore.nn import Dense as _Linear


def compute_weight_bias(simulated_quant_cell, scale_input, offset_input, scale_weight, offset_weight):
    weight = getattr(simulated_quant_cell.name_cells()['compute_cell'], 'original_weight').asnumpy()
    if scale_weight.size == 1 and offset_weight.size == 1:
        try:
            weight = np.round(weight / scale_weight) + offset_weight
        except ZeroDivisionError as ex:
            logging.error('scale_weight cannot be zero. %s', str(ex))
            raise ex
    else:
        for index, _ in enumerate(scale_weight):
            scale_w_cur_channel = scale_weight[index]
            offset_w_cur_channel = offset_weight[index]
            try:
                weight_tmp = np.round(weight[index] / scale_w_cur_channel)
            except ZeroDivisionError as ex:
                logging.error('scale_w_cur_channel cannot be zero. %s', str(ex))
                raise ex
            weight[index] = weight_tmp + offset_w_cur_channel
    if isinstance(simulated_quant_cell.compute_cell, (nn.Dense, _Linear)):
        weight_int8 = np.transpose(weight).astype(np.int8)
    else:
        weight_int8 = weight.astype(np.int8)
    weight_tensor = Tensor(weight_int8, mstype.int8)

    bias_tensor = None
    if simulated_quant_cell.compute_cell.has_bias:
        # fp16>int32
        tmp = np.squeeze(scale_weight) * scale_input
        try:
            bias = simulated_quant_cell.compute_cell.bias.data.asnumpy() / tmp
        except Exception as ex:
            logging.error('tmp cannot be zero. %s', str(ex))
            raise ex
        if isinstance(simulated_quant_cell.compute_cell, nn.Dense):
            bias_tmp = weight_int8.astype(np.int32) * offset_input
            bias = bias - bias_tmp.sum(axis=0)
        bias_tensor = Tensor(bias, mstype.int32)
    return weight_tensor, bias_tensor


def compute_fused_deq_scale(scale_input, scale_weight, offset_weight):
    deq_scale = scale_input * scale_weight
    shift_bits = [0] * deq_scale.size

    float32_deq_scale = np.array(deq_scale, np.float32)
    uint32_deq_scale = np.frombuffer(float32_deq_scale, np.uint32)

    int8_offset_w = np.array(offset_weight, np.int8)
    uint8_offset_w = np.frombuffer(int8_offset_w, np.uint8)

    int8_shift_n = np.array(shift_bits, np.int8)
    uint8_shift_n = np.frombuffer(int8_shift_n, np.uint8)

    # fuse parameter
    # |-----------------|47:40|--------|39:32|--------|31:0|
    #                  offset_w [8]    shift_N [8]    deq_scale [32]
    fused_deq_scale = np.zeros(deq_scale.size, dtype=np.uint64)
    for index in range(deq_scale.size):
        fused_deq_scale[index] = uint8_offset_w[index]

        fused_deq_scale[index] = (fused_deq_scale[index] << np.uint32(8)) + uint8_shift_n[index]
        fused_deq_scale[index] = (fused_deq_scale[index] << np.uint32(32)) + uint32_deq_scale[index]
    return Tensor(fused_deq_scale, mstype.uint64)


def get_op_type(cell):
    if isinstance(cell, nn.Dense):
        return "Dense"
    elif isinstance(cell, nn.Conv2d):
        return "Conv2d"
    elif isinstance(cell, nn.Conv1d):
        return "Conv1d"
    elif isinstance(cell, _Linear):
        return "_Linear"
    else:
        raise TypeError("Supported quantize operator type is Conv2d, Dense, _Linear, please check it.")
