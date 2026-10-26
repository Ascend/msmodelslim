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

from msmodelslim.common.low_rank_decompose.tucker import (
    fold,
    multi_mode_dot,
    positive_largest_trunked_svd_u,
    tucker,
    unfold,
)


def test_unfold_and_fold_round_trip_restores_tensor():
    tensor = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)

    unfolded = unfold(tensor, 0)
    restored = fold(unfolded, 0, tensor.shape)

    assert unfolded.shape == (2, 12)
    assert np.array_equal(restored, tensor)


def test_positive_largest_trunked_svd_u_returns_orthonormal_non_negative_columns():
    tensor = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0], [7.0, 8.0, 9.0], [1.0, 1.0, 1.0]])

    uu = positive_largest_trunked_svd_u(tensor, 2)

    assert uu.shape == (4, 2)
    gram = uu.T @ uu
    assert np.allclose(gram, np.eye(2), atol=1e-10)
    max_abs_rows = np.argmax(np.abs(uu), axis=0)
    assert np.all(uu[max_abs_rows, np.arange(uu.shape[1])] >= 0.0)


def test_multi_mode_dot_with_identity_factors_keeps_tensor():
    tensor = np.arange(2 * 3, dtype=np.float64).reshape(2, 3)
    identities = [np.eye(2), np.eye(3)]

    result = multi_mode_dot(tensor, identities, modes=[0, 1])

    assert np.allclose(result, tensor)


def test_tucker_with_full_ranks_reconstructs_input_tensor():
    tensor = np.arange(8, dtype=np.float64).reshape(2, 2, 2)

    core, factors = tucker(tensor, ranks=[2, 2, 2], iter_max=1)
    reconstruction = multi_mode_dot(core, factors, modes=[0, 1, 2])

    assert core.shape == (2, 2, 2)
    assert [factor.shape for factor in factors] == [(2, 2), (2, 2), (2, 2)]
    assert np.allclose(reconstruction, tensor, atol=1e-6)


def test_tucker_with_int_rank_broadcasts_to_all_modes():
    tensor = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)

    core, factors = tucker(tensor, ranks=2, iter_max=1)

    assert core.shape == (2, 2, 2)
    assert [factor.shape for factor in factors] == [(2, 2), (3, 2), (4, 2)]


def test_tucker_default_ranks_keep_original_shape():
    tensor = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)

    core, factors = tucker(tensor, iter_max=1)

    assert core.shape == (2, 3, 4)
    assert [factor.shape for factor in factors] == [(2, 2), (3, 3), (4, 4)]
