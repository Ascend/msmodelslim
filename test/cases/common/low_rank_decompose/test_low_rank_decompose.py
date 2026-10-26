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

import pytest
import numpy as np

from msmodelslim.common import low_rank_decompose


def test_get_hidden_channels_by_layer_name_returns_configured_channels_when_layer_name_matched():
    assert low_rank_decompose.get_hidden_channels_by_layer_name("aa", 0.5) == 0.5
    assert low_rank_decompose.get_hidden_channels_by_layer_name("aa", 5) == 5
    assert low_rank_decompose.get_hidden_channels_by_layer_name("aa", {"aa": 15}) == 15
    assert low_rank_decompose.get_hidden_channels_by_layer_name("aa", {"a+": 15}) == 15
    assert low_rank_decompose.get_hidden_channels_by_layer_name("aa", {}) == 0
    assert low_rank_decompose.get_hidden_channels_by_layer_name("aa", {"aa": 15}, excludes=["aa"]) == 0


def test_is_hidden_channels_valid_given_valid_when_any_then_true():
    assert low_rank_decompose.is_hidden_channels_valid(12)
    assert low_rank_decompose.is_hidden_channels_valid(0.2)
    assert low_rank_decompose.is_hidden_channels_valid([12, 23])
    assert low_rank_decompose.is_hidden_channels_valid("VBMF")
    assert low_rank_decompose.is_hidden_channels_valid("VBmf")


def test_is_hidden_channels_valid_given_invalid_when_any_then_false():
    assert not low_rank_decompose.is_hidden_channels_valid(0)
    assert not low_rank_decompose.is_hidden_channels_valid(-1)
    assert not low_rank_decompose.is_hidden_channels_valid(None)
    assert not low_rank_decompose.is_hidden_channels_valid([12, -3])
    assert not low_rank_decompose.is_hidden_channels_valid("hello")


def test_get_decompose_channels_2d_returns_channels_floored_to_divisor_when_digit_given():
    source_input = np.random.uniform(size=[32, 64])
    assert low_rank_decompose.get_decompose_channels_2d(source_input, 0.5, divisor=16) == 16
    assert low_rank_decompose.get_decompose_channels_2d(source_input, 12, divisor=16) == 16
    assert low_rank_decompose.get_decompose_channels_2d(source_input, 12, divisor=64) == 0


def test_get_decompose_channels_2d_returns_channels_floored_to_divisor_when_vbmf_given():
    source_input = np.random.uniform(size=[32, 64])
    assert low_rank_decompose.get_decompose_channels_2d(source_input, "vbmf", divisor=16) == 16
    assert low_rank_decompose.get_decompose_channels_2d(source_input, "vbmf", divisor=64) == 0


def test_decompose_weight_2d_svd_returns_expected_factor_shapes_when_hidden_channels_given():
    in_channels, out_channels, hidden_channels = 32, 64, 16
    source_input = np.random.uniform(size=[out_channels, in_channels])
    svd_uu, svd_vv, actual_hidden_channels = low_rank_decompose.decompose_weight_2d_svd(source_input, hidden_channels)
    assert actual_hidden_channels == hidden_channels
    assert svd_uu.shape == (out_channels, hidden_channels)
    assert svd_vv.shape == (hidden_channels, in_channels)


def test_decompose_weight_2d_svd_returns_expected_factor_shapes_when_input_data_given():
    in_channels, out_channels, hidden_channels = 32, 64, 16
    source_input = np.random.uniform(size=[out_channels, in_channels])
    input_data = np.random.uniform(size=[in_channels, in_channels])
    svd_uu, svd_vv, actual_hidden_channels = low_rank_decompose.decompose_weight_2d_svd(
        source_input, hidden_channels, input_data
    )
    assert actual_hidden_channels == hidden_channels
    assert svd_uu.shape == (out_channels, hidden_channels)
    assert svd_vv.shape == (hidden_channels, in_channels)


def test_decompose_weight_2d_svd_given_zero_when_any_then_none():
    source_input = np.random.uniform(size=[32, 64])
    with pytest.raises(ValueError):
        _ = low_rank_decompose.decompose_weight_2d_svd(source_input, 0)
    with pytest.raises(ValueError):
        _ = low_rank_decompose.decompose_weight_2d_svd([1], 16)


def test_get_decompose_channels_4d_returns_channel_pairs_floored_to_divisor_when_digit_given():
    source_input = np.random.uniform(size=[32, 64, 3, 3])
    assert low_rank_decompose.get_decompose_channels_4d(source_input, 0.5, divisor=16) == (32, 32)
    assert low_rank_decompose.get_decompose_channels_4d(source_input, 12, divisor=16) == (16, 16)
    assert low_rank_decompose.get_decompose_channels_4d(source_input, 12, divisor=64) == (0, 0)
    assert low_rank_decompose.get_decompose_channels_4d(source_input, [12, 32], divisor=16) == (16, 32)
    assert low_rank_decompose.get_decompose_channels_4d(source_input, [12, 32], divisor=64) == (0, 0)


def test_get_decompose_channels_4d_returns_channel_pairs_floored_to_divisor_when_vbmf_given():
    source_input = np.random.uniform(size=[32, 64, 3, 3])
    assert low_rank_decompose.get_decompose_channels_4d(source_input, "vbmf", divisor=16) == (16, 16)
    assert low_rank_decompose.get_decompose_channels_4d(source_input, "vbmf", divisor=64) == (0, 0)


def test_decompose_weight_4d_tucker_returns_expected_factor_shapes_when_hidden_channels_given():
    in_channels, out_channels, kernel_size, hidden_in, hidden_out = 32, 64, 3, 16, 32
    source_input = np.random.uniform(size=[out_channels, in_channels, kernel_size, kernel_size])
    res = low_rank_decompose.decompose_weight_4d_tucker(source_input, hidden_out, hidden_in)
    first = res.get('first', None)
    core = res.get('core', None)
    last = res.get('last', None)
    (actual_hidden_out, actual_hidden_in) = res.get('out_in', None)
    assert actual_hidden_out == hidden_out and actual_hidden_in == hidden_in
    assert first.shape == (hidden_in, in_channels, 1, 1)
    assert core.shape == (hidden_out, hidden_in, kernel_size, kernel_size)
    assert last.shape == (out_channels, hidden_out, 1, 1)


def test_decompose_weight_4d_tucker_given_zero_when_any_then_none():
    with pytest.raises(ValueError):
        _ = low_rank_decompose.decompose_weight_4d_tucker(np.random.uniform(size=[64, 32, 3, 3]), 0, 0)
    with pytest.raises(ValueError):
        _ = low_rank_decompose.decompose_weight_4d_tucker([1], 16, 32)
