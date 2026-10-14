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

import copy
import logging

import numpy as np
import mindspore
from mindspore import Tensor
from mindspore import ops

try:
    from mindspore.nn.transformer.layers import _Linear
except ModuleNotFoundError:
    try:
        from mindspore.nn.layer.transformer import _Linear
    except ImportError:
        from mindspore.nn import Dense as _Linear


def quantize_weight(weight):
    max_ = weight.abs()
    max_shape = (-1,)

    for _ in range(len(weight.shape) - 1):
        max_ = max_.max(1)
        max_shape += (1,)

    max_ = max_.reshape(max_shape)

    return max_, Tensor(np.zeros(max_.shape), mindspore.float32)


def quant_weight(cell):
    origin_weight = cell.weight.data
    tmp_weight = copy.deepcopy(origin_weight)
    shape = tmp_weight.shape
    # for _Linear operator, the shape of weight is (out_channels, in_channels)
    if isinstance(cell, _Linear) and cell.out_channels != shape[0]:
        cell.transpose_b = False
        transpose = ops.Transpose()
        tmp_weight = transpose(tmp_weight, (1, 0))

    if tmp_weight.dtype == mindspore.float16:
        tmp_weight.set_dtype(mindspore.float32)

    int8_min = -128
    int8_max = 127
    scale, offset = quantize_weight(tmp_weight)
    try:
        scale = scale / int8_max
    except ZeroDivisionError as ex:
        logging.error('int8_max cannot be zero. %s', str(ex))
        raise ex
    setattr(cell, 'original_weight', tmp_weight)
    setattr(cell, 'scale_weight', scale)
    setattr(cell, 'offset_weight', offset)

    # Quant and Dequant
    _round = ops.Round()
    try:
        tmp_weight = _round(tmp_weight / scale).clip(int8_min, int8_max)
    except ZeroDivisionError as ex:
        logging.error('scale cannot be zero. %s', str(ex))
        raise ex
    tmp_weight = tmp_weight * scale
    if hasattr(cell, "transpose_b") and not cell.transpose_b:
        transpose = ops.Transpose()
        tmp_weight = transpose(tmp_weight, (1, 0))
    cell.weight.set_data(tmp_weight)
