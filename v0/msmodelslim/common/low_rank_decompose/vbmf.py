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
from scipy.optimize import minimize_scalar


class EvbSigma2Params:
    def __init__(self, low, median, svd_ss, residual, inner_thresh):
        self.low = low
        self.median = median
        self.svd_ss = svd_ss
        self.residual = residual
        self.inner_thresh = inner_thresh


def none_zero_divide(inputs, divisor, eps=1e-6):
    return np.divide(inputs, np.maximum(divisor, eps))


def evbmf(source_input, sigma2=None, high=None) -> int:
    """Empirical Variational Bayes Matrix Factorization.
    Paper: Global analytic solution of fully-observed variational Bayesian matrix factorization.

    Args:
      source_input:
      sigma2: Variance of the noise on source_input.
          Default None for estimated by minimizing the free energy.
      high: Maximum rank of the factorized matrices.
          Default None for the smallest of the sides of the input source_input.

    Returns: VBMF valid rank of input tensor.
    """
    low, median = source_input.shape
    if high is None:
        high = low

    alpha = none_zero_divide(low, median)
    alpha_thresh = 2.5129 * np.sqrt(alpha)

    # SVD of the input matrix, max rank of H
    svd_uu, svd_ss, svd_vv = np.linalg.svd(source_input)
    svd_uu, svd_ss, svd_vv = svd_uu[:, :high], svd_ss[:high], svd_vv[:high].T

    # Estimation of the variance when sigma2 is unspecified
    if sigma2 is None:
        # Calculate residual
        residual = np.sum(np.sum(source_input**2) - np.sum(svd_ss**2)) if high < low else 0

        inner_thresh = (1 + alpha_thresh) * (1 + none_zero_divide(alpha, alpha_thresh))
        eh_ub_base = int(np.min([np.ceil(none_zero_divide(low, 1 + alpha)) - 1, high])) - 1
        # 限制eh_ub的范围：0 ≤ eh_ub ≤ high-2，确保eh_ub + 1 < high（即不超过svd_ss的长度）
        eh_ub = np.clip(eh_ub_base, 0, high - 2)
        upper_bound = none_zero_divide((np.sum(svd_ss**2) + residual), low * median)
        ss_median = none_zero_divide(np.mean(svd_ss[eh_ub + 1 :] ** 2), median)
        lower_bound = np.max([none_zero_divide(svd_ss[eh_ub + 1] ** 2, median * inner_thresh), ss_median])

        if upper_bound < lower_bound:
            lower_bound, upper_bound = upper_bound, lower_bound

        params = EvbSigma2Params(low=low, median=median, svd_ss=svd_ss, residual=residual, inner_thresh=inner_thresh)

        sigma2_opt = minimize_scalar(
            evb_sigma2,
            args=params,
            bounds=[lower_bound, upper_bound],
            method='Bounded',
        )
        sigma2 = sigma2_opt.x

    # Threshold gamma term
    threshold = np.sqrt(median * sigma2 * (1 + alpha_thresh) * (1 + none_zero_divide(alpha, alpha_thresh)))
    threshold_pos = np.sum(svd_ss > threshold)
    return int(threshold_pos)


def evb_sigma2(sigma2, params):
    low = params.low
    median = params.median
    svd_ss = params.svd_ss
    residual = params.residual
    inner_thresh = params.inner_thresh
    high = len(svd_ss)

    alpha = none_zero_divide(low, median)
    inner_xx = none_zero_divide(svd_ss**2, median * sigma2)

    inner_upper, inner_lower = inner_xx[inner_xx > inner_thresh], inner_xx[inner_xx <= inner_thresh]
    tau_z1 = 0.5 * (inner_upper - (1 + alpha) + np.sqrt((inner_upper - (1 + alpha)) ** 2 - 4 * alpha))

    term_z2 = np.sum(inner_lower - np.log(inner_lower))
    term_z1_tau = np.sum(inner_upper - tau_z1)
    term_tau_z1 = np.sum(np.log(none_zero_divide(tau_z1 + 1, inner_upper)))
    term_z1 = alpha * np.sum(np.log(none_zero_divide(tau_z1, alpha) + 1))

    residual_median = none_zero_divide(residual, median * sigma2)
    obj = term_z2 + term_z1_tau + term_tau_z1 + term_z1 + residual_median + (low - high) * np.log(sigma2)
    return obj


def search_rank(source_weight):
    diag_0 = evbmf(source_weight.reshape([source_weight.shape[0], -1]))
    perm = [1, 0] + list(range(2, len(source_weight.shape)))
    source_weight_trans = np.transpose(source_weight, perm)
    diag_1 = evbmf(source_weight_trans.reshape([source_weight_trans.shape[0], -1]))
    return max(diag_0, 1), max(diag_1, 1)
