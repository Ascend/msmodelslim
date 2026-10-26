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

import numpy as np

from msmodelslim.common.low_rank_decompose.vbmf import evbmf, none_zero_divide, search_rank


def test_evbmf_returns_rank_tracking_singular_value_threshold():
    matrix = np.diag([4.0, 2.0, 1.0])

    assert evbmf(matrix, sigma2=1e-6) == 3
    assert evbmf(matrix, sigma2=0.01) == 3
    assert evbmf(matrix, sigma2=0.1) == 2
    assert evbmf(matrix, sigma2=1.0) == 1


def test_evbmf_estimates_sigma2_when_not_provided():
    matrix = np.diag([4.0, 2.0, 1.0])

    assert evbmf(matrix) == 0


def test_search_rank_returns_rank_for_both_leading_dimensions():
    matrix = np.diag([4.0, 2.0, 1.0])

    assert search_rank(matrix) == (1, 1)


def test_none_zero_divide_uses_eps_for_zero_divisor():
    numerator = np.array([2.0, 4.0])
    divisor = np.array([0.0, 2.0])

    result = none_zero_divide(numerator, divisor, eps=1e-6)

    assert np.allclose(result, np.array([2.0e6, 2.0]), rtol=1e-9)
