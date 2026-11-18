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
# pylint: disable=no-else-break,unbalanced-tuple-unpacking,undefined-variable

import re
from collections import namedtuple

import numpy as np

from msmodelslim.common.low_rank_decompose.tucker import tucker
from msmodelslim.common.low_rank_decompose.vbmf import search_rank


# Supports `RankMethods.VBMF` and `xxx in RankMethods`
RankMethods = namedtuple('RankMethods', ['VBMF'])('vbmf')


def make_divisible(inputs, divisor=64, min_value=None, limit_round_down=0.9):
    if min_value is None:
        min_value = divisor
    try:
        outputs = max(min_value, int(inputs + divisor / 2) // divisor * divisor)
    except ZeroDivisionError as ex:
        logging.error('divisor can not be zero. %s', str(ex))  # noqa: F821
        raise ex
    # Make sure that round down does not go down by more than 10%.
    if outputs < limit_round_down * inputs:
        outputs += divisor
    return outputs


def get_hidden_channels_by_layer_name(layer_name, hidden_channels, excludes=None):
    if excludes and layer_name in excludes:
        return 0

    cur_hidden_channels = 0
    if isinstance(hidden_channels, dict):
        for regex, hidden_channel in hidden_channels.items():
            if isinstance(regex, re.Pattern) and regex.search(layer_name):
                cur_hidden_channels = hidden_channel
                break
            elif isinstance(regex, str) and re.search(regex, layer_name):
                cur_hidden_channels = hidden_channel
                break
    else:
        cur_hidden_channels = hidden_channels
    return cur_hidden_channels


def trunked_svd_uv(source_tensor, rank=None):
    # weight: [in, out] -> uu [in, in], ss [out], vv [out, out]
    svd_uu, svd_ss, svd_vv = np.linalg.svd(source_tensor, full_matrices=False)

    ss_sqrt = np.sqrt(svd_ss[:rank])
    svd_uu = svd_uu[:, :rank] * np.expand_dims(ss_sqrt, 0)
    svd_vv = np.expand_dims(ss_sqrt, 1) * svd_vv[:rank]
    return svd_uu, svd_vv


def is_hidden_channels_valid(hidden_channels, digit_only=False):
    if isinstance(hidden_channels, (list, tuple)):
        if len(hidden_channels) == 2 and isinstance(hidden_channels[0], int) and isinstance(hidden_channels[1], int):
            hidden_channels = min(hidden_channels)
        else:
            return False

    if isinstance(hidden_channels, (int, float)):
        return hidden_channels > 0
    elif not digit_only and isinstance(hidden_channels, str):
        return hidden_channels.lower() in RankMethods

    return False


def data_aware_decompose_2d(source_input, input_data, rank=None):
    # source_input: [out_channels, in_channels], input_data: [in_channels, in_channels]
    uu_weight, ss_weight, vv_weight = np.linalg.svd(source_input, full_matrices=False, hermitian=False)
    uu_data, ss_data, _ = np.linalg.svd(input_data.astype("float32"), full_matrices=False, hermitian=True)

    inner_z = (vv_weight * np.expand_dims(ss_weight, 1)) @ (uu_data * np.expand_dims(ss_data, 0))
    uu_inner_z, vv_inner_z = trunked_svd_uv(inner_z, rank=rank)

    uu_result = uu_weight @ uu_inner_z
    vv_result = np.divide(vv_inner_z, np.expand_dims(np.maximum(ss_data, 1e-6), 0)) @ uu_data.T
    return uu_result, vv_result


def get_decompose_channels_2d(source_input, hidden_channels, divisor=64, as_numpy_func=None):
    out_channels, in_channels = source_input.shape[0], source_input.shape[1]
    source_params = out_channels * in_channels

    if isinstance(hidden_channels, str) and hidden_channels.lower() == RankMethods.VBMF:
        source_input = as_numpy_func(source_input) if as_numpy_func is not None else source_input
        hidden_channels = max(search_rank(source_input))
    elif isinstance(hidden_channels, (list, tuple)):
        hidden_channels = max(hidden_channels)
    elif isinstance(hidden_channels, float) and hidden_channels < 1:
        target_params = source_params * hidden_channels
        hidden_channels = int(target_params / (out_channels + in_channels))

    hidden_channels = make_divisible(hidden_channels, divisor=divisor)

    result_params = hidden_channels * in_channels + hidden_channels * out_channels
    if result_params >= source_params:
        hidden_channels = 0  # Don't decompose
    return hidden_channels


def decompose_weight_2d_svd(source_input, hidden_channels: int, input_data=None):
    if not is_hidden_channels_valid(hidden_channels, digit_only=True) or not isinstance(source_input, np.ndarray):
        raise ValueError(f"Parameter hidden_channels={hidden_channels} or source_input is not valid")

    source_input = source_input.astype("float32")
    if input_data is not None:
        svd_uu, svd_vv = data_aware_decompose_2d(source_input, input_data=input_data, rank=hidden_channels)
    else:
        svd_uu, svd_vv = trunked_svd_uv(source_input, hidden_channels)
    actual_hidden_channels = svd_uu.shape[1]
    return svd_uu, svd_vv, actual_hidden_channels


def data_aware_decompose_4d(source_input, input_data, ranks):
    # source_input: [out_channels, in_channels, kernel_size, kernel_size], input_data: [in_channels, in_channels]
    uu_data, ss_data, _ = np.linalg.svd(input_data.astype("float32"), full_matrices=False, hermitian=True)

    inner_z = uu_data * np.expand_dims(ss_data, 0)
    inner_z = (source_input.transpose([2, 3, 0, 1]) @ inner_z).transpose([2, 3, 0, 1])
    core, [last, first] = tucker(inner_z, ranks=ranks, modes=[0, 1])

    first = uu_data @ np.divide(first, np.expand_dims(np.maximum(ss_data, 1e-6), 1))
    return core, [last, first]


def get_decompose_channels_4d(source_input, hidden_channels, divisor=64, as_numpy_func=None):
    out_channels, in_channels, kernel_size = source_input.shape[0], source_input.shape[1], source_input.shape[2]
    kernel_patch = (kernel_size * kernel_size) if isinstance(kernel_size, int) else kernel_size[0] * kernel_size[1]
    source_params = out_channels * in_channels * kernel_patch

    if isinstance(hidden_channels, str) and hidden_channels.lower() == RankMethods.VBMF:
        source_input = as_numpy_func(source_input) if as_numpy_func is not None else source_input
        hidden_channels = search_rank(source_input)
    elif isinstance(hidden_channels, float) and hidden_channels < 1:
        target_params = source_params * hidden_channels
        hidden_channels = np.roots([kernel_patch, (in_channels + out_channels), -target_params]).max()

    if isinstance(hidden_channels, (int, float)):
        hidden_out, hidden_in = hidden_channels, hidden_channels
    else:
        hidden_out, hidden_in = hidden_channels[0], hidden_channels[1]
    hidden_out = make_divisible(hidden_out, divisor=divisor)
    hidden_in = make_divisible(hidden_in, divisor=divisor)

    result_params = hidden_in * in_channels + hidden_in * hidden_out * kernel_patch + hidden_out * out_channels
    if result_params >= source_params:
        return 0, 0  # Total parameters after decompose is larger than original.
    else:
        return hidden_out, hidden_in


def decompose_weight_4d_tucker(source_input, hidden_out: int, hidden_in: int, input_data=None):
    if hidden_in <= 0 or hidden_out <= 0 or not isinstance(source_input, np.ndarray):
        raise ValueError(f"Parameter hidden_in={hidden_in} or hidden_out={hidden_out} or source_input is not valid")

    source_input = source_input.astype("float32")
    ranks = (hidden_out, hidden_in)
    if input_data is not None:
        result = data_aware_decompose_4d(source_input, input_data=input_data, ranks=ranks)
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("data_aware_decompose_4d returned an unexpected result format.")

        core, factors = result
        if not isinstance(factors, (list, tuple)) or len(factors) != 2:
            raise RuntimeError("Expected two factor matrices from data_aware_decompose_4d.")
        last, first = factors
    else:
        result = tucker(source_input, ranks=ranks, modes=[0, 1])
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("tucker decomposition returned an unexpected result format.")

        core, factors = result
        if not isinstance(factors, (list, tuple)) or len(factors) != 2:
            raise RuntimeError("Expected two factor matrices from tucker decomposition.")
        last, first = factors

    first = first.T[:, :, None, None]
    last = last[:, :, None, None]
    actual_hidden_out, actual_hidden_in = core.shape[0], core.shape[1]
    return {"first": first, "core": core, "last": last, 'out_in': (actual_hidden_out, actual_hidden_in)}
