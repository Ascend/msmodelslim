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

from typing import Dict, Literal

import torch
from pydantic import ConfigDict

from msmodelslim.utils.plugin import register_plugin

from .base import (
    BaseLosslessMetric,
    LosslessMetricConfig,
    MetricContext,
    MetricResult,
    MetricViolation,
)


def _max_abs_diff(reference: torch.Tensor, processed: torch.Tensor) -> float:
    if reference.numel() == 0:
        return 0.0
    return torch.max(torch.abs(reference - processed)).item()


def _max_relative_diff(reference: torch.Tensor, processed: torch.Tensor) -> float:
    if reference.numel() == 0:
        return 0.0
    denominator = torch.maximum(reference.abs(), processed.abs()).clamp_min(torch.finfo(reference.dtype).tiny)
    return torch.max(torch.abs(reference - processed) / denominator).item()


def _distribution_metrics(reference: torch.Tensor, processed: torch.Tensor) -> Dict[str, float]:
    if reference.numel() == 0:
        return {
            "mean_abs_diff": 0.0,
            "rmse": 0.0,
            "reference_rms": 0.0,
            "processed_rms": 0.0,
            "normalized_rmse": 0.0,
        }
    difference = reference - processed
    mean_abs_diff = difference.abs().mean().item()
    rmse = difference.square().mean().sqrt().item()
    reference_rms = reference.square().mean().sqrt().item()
    processed_rms = processed.square().mean().sqrt().item()
    denominator = max(reference_rms, processed_rms, torch.finfo(reference.dtype).tiny)
    return {
        "mean_abs_diff": mean_abs_diff,
        "rmse": rmse,
        "reference_rms": reference_rms,
        "processed_rms": processed_rms,
        "normalized_rmse": rmse / denominator,
    }


def _infer_tolerance(reference: torch.Tensor, processed: torch.Tensor):
    eps = max(torch.finfo(reference.dtype).eps, torch.finfo(processed.dtype).eps)
    reference_scale = reference.detach().abs().max().item() if reference.numel() else 1.0
    processed_scale = processed.detach().abs().max().item() if processed.numel() else 1.0
    scale = max(1.0, reference_scale, processed_scale)
    return max(1e-6, 4 * eps * scale), max(1e-5, 4 * eps)


def _infer_distribution_tolerance(reference: torch.Tensor, processed: torch.Tensor):
    eps = max(torch.finfo(reference.dtype).eps, torch.finfo(processed.dtype).eps)
    reference_scale = reference.detach().abs().max().item() if reference.numel() else 1.0
    processed_scale = processed.detach().abs().max().item() if processed.numel() else 1.0
    scale = max(1.0, reference_scale, processed_scale)
    if eps <= torch.finfo(torch.float32).eps:
        max_abs_factor = 128
        normalized_rmse_factor = 64
    elif torch.bfloat16 in (reference.dtype, processed.dtype):
        # Keep one extra BF16 ULP beyond the accumulated-rounding envelope.
        max_abs_factor = 9
        normalized_rmse_factor = 8
    else:
        max_abs_factor = 8
        normalized_rmse_factor = 8
    return (
        max(4e-6, max_abs_factor * eps * scale),
        max(4e-6, normalized_rmse_factor * eps),
    )


class NumericalEquivalenceMetricConfig(LosslessMetricConfig):
    """Configuration for the built-in numerical equivalence metric."""

    model_config = ConfigDict(extra="forbid")
    name: Literal["numerical_equivalence"] = "numerical_equivalence"


class NumericalEquivalenceMetric(BaseLosslessMetric):
    """Preserve the existing allclose-or-distribution decision."""

    name = "numerical_equivalence"

    def evaluate(self, reference: torch.Tensor, processed: torch.Tensor, _context: MetricContext) -> MetricResult:
        reference_float = reference.float()
        processed_float = processed.float()
        max_abs_diff = _max_abs_diff(reference_float, processed_float)
        max_relative_diff = _max_relative_diff(reference_float, processed_float)
        distribution = _distribution_metrics(reference_float, processed_float)
        atol, rtol = _infer_tolerance(reference, processed)
        max_abs_threshold, normalized_rmse_threshold = _infer_distribution_tolerance(reference, processed)
        elementwise_allclose = bool(torch.allclose(reference_float, processed_float, atol=atol, rtol=rtol))
        distribution_passed = (
            max_abs_diff <= max_abs_threshold and distribution["normalized_rmse"] <= normalized_rmse_threshold
        )
        passed = elementwise_allclose or distribution_passed
        if elementwise_allclose:
            comparison_mode = "elementwise_allclose"
        elif distribution_passed:
            comparison_mode = "distribution"
        else:
            comparison_mode = "failed"
        violations = []
        if not passed and max_abs_diff > max_abs_threshold:
            violations.append(
                MetricViolation(
                    name="max_abs_diff",
                    value=max_abs_diff,
                    threshold=max_abs_threshold,
                    comparison="<=",
                )
            )
        if not passed and distribution["normalized_rmse"] > normalized_rmse_threshold:
            violations.append(
                MetricViolation(
                    name="normalized_rmse",
                    value=distribution["normalized_rmse"],
                    threshold=normalized_rmse_threshold,
                    comparison="<=",
                )
            )
        return MetricResult(
            name=self.name,
            passed=passed,
            value=max_abs_diff,
            threshold={
                "atol": atol,
                "rtol": rtol,
                "max_abs_diff": max_abs_threshold,
                "normalized_rmse": normalized_rmse_threshold,
            },
            observations={
                "max_abs_diff": max_abs_diff,
                "max_relative_diff": max_relative_diff,
                **distribution,
                "atol": atol,
                "rtol": rtol,
                "elementwise_allclose": elementwise_allclose,
                "distribution_passed": distribution_passed,
                "comparison_mode": comparison_mode,
            },
            violations=tuple(violations),
        )


def _get_numerical_equivalence_metric_plugin():
    return NumericalEquivalenceMetricConfig, NumericalEquivalenceMetric


register_plugin(_get_numerical_equivalence_metric_plugin)
