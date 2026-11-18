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

from msmodelslim import logger


def positive_largest_trunked_svd_u(source_tensor, rank):
    full_matrices = rank > min(source_tensor.shape)
    svd_uu, _, _ = np.linalg.svd(source_tensor, full_matrices=full_matrices)
    svd_uu = svd_uu[:, :rank]

    # Keep the largest absolute value in each column of `u` always positive.
    max_abs_cols = np.argmax(np.abs(svd_uu), axis=0)
    signs = np.sign(svd_uu[max_abs_cols, range(svd_uu.shape[1])])
    return svd_uu * signs


def unfold(source_tensor, axis):
    axes = list(range(len(source_tensor.shape)))
    axes = [axes.pop(axis)] + axes
    source_tensor = source_tensor.transpose(axes)
    return source_tensor.reshape([source_tensor.shape[0], -1])


def fold(source_tensor, axis, target_shape):
    target_shape = list(target_shape)
    axes = list(range(len(target_shape)))

    target_shape.pop(axis)
    source_tensor = source_tensor.reshape([source_tensor.shape[0], *target_shape])

    axes = axes[1 : axis + 1] + [0] + axes[axis + 1 :]
    return source_tensor.transpose(axes)


def multi_mode_dot(source_tensor, uu_for_each_modes, modes, transpose=False):
    result = source_tensor
    for uu_for_each_mode, mode in zip(uu_for_each_modes, modes):
        source_shape = result.shape
        result = unfold(result, mode)
        result = (uu_for_each_mode.T @ result) if transpose else (uu_for_each_mode @ result)
        result = fold(result, mode, source_shape)
    return result


def tucker(source_tensor, ranks=None, modes=None, iter_max=100, error_tolerance=1e-4):
    modes = list(range(np.ndim(source_tensor))) if modes is None else modes
    if ranks is None:
        source_shape = source_tensor.shape
        ranks = [source_shape[mode] for mode in modes]
    elif isinstance(ranks, int):
        ranks = [ranks] * len(modes)

    sorted_modes_with_ranks = sorted(zip(ranks, modes), key=lambda xx: xx[1])
    sorted_modes = [ii[1] for ii in sorted_modes_with_ranks]
    logger.info(
        "modes: %s, ranks: %s, sorted_modes_with_ranks: %s", str(modes), str(ranks), str(sorted_modes_with_ranks)
    )
    # Init by SVD
    uu_for_each_modes = []
    for rank, mode in sorted_modes_with_ranks:
        cur_tensor = unfold(source_tensor, mode)
        svd_uu = positive_largest_trunked_svd_u(cur_tensor, rank)
        uu_for_each_modes.append(svd_uu)

    norm_tensor = np.linalg.norm(source_tensor)
    norm_tensor_square = norm_tensor**2

    core, pre_rec_error = None, None
    for iteration in range(iter_max):
        for index, (rank, mode) in enumerate(sorted_modes_with_ranks):
            cur_modes = sorted_modes[:index] + sorted_modes[index + 1 :]
            cur_uu_for_each_modes = uu_for_each_modes[:index] + uu_for_each_modes[index + 1 :]
            approximate_core = multi_mode_dot(source_tensor, cur_uu_for_each_modes, modes=cur_modes, transpose=True)
            cur_tensor = unfold(approximate_core, mode)
            svd_uu = positive_largest_trunked_svd_u(cur_tensor, rank)
            uu_for_each_modes[index] = svd_uu
        core = multi_mode_dot(source_tensor, uu_for_each_modes, modes=sorted_modes, transpose=True)

        cur_rec_error = np.sqrt(np.abs(norm_tensor_square - np.linalg.norm(core) ** 2))
        cur_rec_error = np.divide(cur_rec_error, np.maximum(norm_tensor, 1e-6))
        if iteration > 1 and error_tolerance and np.abs(pre_rec_error - cur_rec_error) < error_tolerance:
            logger.info("converged in %s iterations.", str(iteration))
            break
        pre_rec_error = cur_rec_error

    return core, uu_for_each_modes
