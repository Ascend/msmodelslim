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

from decimal import Decimal

import pytest

from msmodelslim.core.tune_strategy import AccuracyExpectation, EvaluateAccuracy
from msmodelslim.infra.service_oriented_evaluate_service import is_demand_satisfied
from msmodelslim.utils.exception import SpecError


def _expectation(dataset: str, target: str, tolerance: str) -> AccuracyExpectation:
    return AccuracyExpectation(dataset=dataset, target=Decimal(target), tolerance=Decimal(tolerance))


def _result(dataset: str, accuracy: str) -> EvaluateAccuracy:
    return EvaluateAccuracy(dataset=dataset, accuracy=Decimal(accuracy))


@pytest.mark.parametrize(
    "target,accuracy,tolerance,expected",
    [
        ("0.95", "0.94", "0.01", True),  # 边界：target - accuracy == tolerance
        ("0.95", "0.9401", "0.01", True),  # target - accuracy < tolerance
        ("0.95", "0.9399", "0.01", False),  # target - accuracy > tolerance
    ],
)
def test_is_demand_satisfied_return_expected_when_boundary_cases(
    target: str, accuracy: str, tolerance: str, expected: bool
):
    demand = [_expectation(dataset="test_dataset", target=target, tolerance=tolerance)]
    evaluate_result = [_result(dataset="test_dataset", accuracy=accuracy)]
    assert is_demand_satisfied(demand=demand, evaluate_result=evaluate_result) is expected


def test_is_demand_satisfied_return_false_when_result_missing_dataset_in_demand():
    demand = [_expectation(dataset="dataset_a", target="0.95", tolerance="0.01")]
    evaluate_result = [_result(dataset="dataset_b", accuracy="0.95")]
    assert is_demand_satisfied(demand=demand, evaluate_result=evaluate_result) is False


def test_is_demand_satisfied_raise_spec_error_when_demand_has_duplicate_dataset():
    demand = [
        _expectation(dataset="dataset_a", target="0.95", tolerance="0.01"),
        _expectation(dataset="dataset_a", target="0.96", tolerance="0.01"),
    ]
    evaluate_result = [_result(dataset="dataset_a", accuracy="0.95")]
    with pytest.raises(SpecError, match="Duplicate dataset found in demand."):
        is_demand_satisfied(demand=demand, evaluate_result=evaluate_result)


def test_is_demand_satisfied_raise_spec_error_when_result_has_duplicate_dataset():
    demand = [_expectation(dataset="dataset_a", target="0.95", tolerance="0.01")]
    evaluate_result = [
        _result(dataset="dataset_a", accuracy="0.95"),
        _result(dataset="dataset_a", accuracy="0.96"),
    ]
    with pytest.raises(SpecError, match="Duplicate dataset found in result."):
        is_demand_satisfied(demand=demand, evaluate_result=evaluate_result)
