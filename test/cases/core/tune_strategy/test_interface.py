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

from msmodelslim.core.tune_strategy.interface import (
    AccuracyExpectation,
    EvaluateAccuracy,
    EvaluateResult,
    ITuningStrategy,
    ITuningStrategyFactory,
    StrategyConfig,
)
from msmodelslim.utils.exception import SchemaValidateError


class TestEvaluateAccuracy:
    """Tests for EvaluateAccuracy model."""

    def test_constructs_with_dataset_and_accuracy_when_valid(self):
        """场景：传入 dataset 与 accuracy。预期：字段解析成功。"""
        acc = EvaluateAccuracy(dataset="bench", accuracy=Decimal("0.91"))
        assert acc.dataset == "bench"
        assert acc.accuracy == Decimal("0.91")


class TestEvaluateResult:
    """Tests for EvaluateResult model."""

    def test_constructs_with_defaults_when_minimal_fields(self):
        """场景：仅设置 is_satisfied。预期：accuracies/expectations 默认为空列表。"""
        result = EvaluateResult(is_satisfied=True)
        assert result.is_satisfied is True
        assert result.accuracies == []
        assert result.expectations == []

    def test_accepts_nested_models_when_provided(self):
        """场景：填入 accuracies 与 expectations。预期：嵌套模型解析成功。"""
        result = EvaluateResult(
            is_satisfied=False,
            accuracies=[EvaluateAccuracy(dataset="ds1", accuracy=Decimal("0.9"))],
            expectations=[AccuracyExpectation(dataset="ds1", target=Decimal("0.95"), tolerance=Decimal("0.05"))],
        )
        assert result.accuracies[0].dataset == "ds1"
        assert result.expectations[0].target == Decimal("0.95")


class TestAccuracyExpectation:
    """Tests for AccuracyExpectation validation."""

    def test_accepts_valid_target_and_tolerance_when_positive(self):
        """场景：target>0 且 tolerance>=0。预期：构造成功。"""
        exp = AccuracyExpectation(
            dataset="bench",
            target=Decimal("0.9"),
            tolerance=Decimal("0.01"),
        )
        assert exp.dataset == "bench"

    def test_raises_validation_error_when_target_zero(self):
        """场景：target=0。预期：校验失败。"""
        with pytest.raises(SchemaValidateError):
            AccuracyExpectation(
                dataset="bench",
                target=Decimal("0"),
                tolerance=Decimal("0.01"),
            )

    def test_raises_validation_error_when_tolerance_negative(self):
        """场景：tolerance<0。预期：校验失败。"""
        with pytest.raises(SchemaValidateError):
            AccuracyExpectation(
                dataset="bench",
                target=Decimal("0.9"),
                tolerance=Decimal("-0.01"),
            )


class TestStrategyConfig:
    """Tests for StrategyConfig plugin entry model."""

    def test_is_typed_config_subclass_when_inspected(self):
        """场景：检查 StrategyConfig 类型。预期：为 TypedConfig 子类且含 type 字段。"""
        from msmodelslim.utils.plugin import TypedConfig

        assert issubclass(StrategyConfig, TypedConfig)
        assert "type" in StrategyConfig.model_fields


class TestITuningStrategy:
    """Tests for ITuningStrategy abstract contract."""

    def test_cannot_instantiate_without_generate_practice(self):
        """场景：未实现 generate_practice。预期：TypeError。"""

        class IncompleteStrategy(ITuningStrategy):
            pass

        with pytest.raises(TypeError):
            IncompleteStrategy()  # pylint: disable=abstract-class-instantiated


class TestITuningStrategyFactory:
    """Tests for ITuningStrategyFactory abstract contract."""

    def test_cannot_instantiate_without_create_strategy(self):
        """场景：未实现 create_strategy。预期：TypeError。"""

        class IncompleteFactory(ITuningStrategyFactory):
            pass

        with pytest.raises(TypeError):
            IncompleteFactory()  # pylint: disable=abstract-class-instantiated
