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

from typing import Literal

import pytest
import torch
from pydantic import ConfigDict, field_validator

from msmodelslim.processor.lossless.metrics import (
    BaseLosslessMetric,
    LosslessMetricConfig,
    MetricResult,
    MetricViolation,
)
from msmodelslim.utils.exception import SchemaValidateError, UnexpectedError, UnsupportedError
from msmodelslim.utils.plugin import register_plugin
from test.cases.processor.lossless.test_lossless_check_processor import (
    _FakeConfig,
    _RepeatedLinearModule,
    _processor,
)


def _metric_processor(*args, **kwargs):
    processor = _processor(*args, **kwargs)
    processor.pre_run()
    return processor


def test_metric_failure_log_names_metric_threshold_and_side(mocker):
    logger = mocker.patch("msmodelslim.processor.lossless.processor.get_logger").return_value
    processor = _metric_processor(_RepeatedLinearModule(), _FakeConfig())

    with pytest.raises(UnexpectedError, match="metric_threshold_exceeded"):
        processor._compare_tree(
            torch.zeros(4, dtype=torch.float32),
            torch.ones(4, dtype=torch.float32),
            request_name="block",
            structure_name="block.linear",
            side="input",
            invocation=0,
        )

    messages = [call.args[0] % tuple(call.args[1:]) for call in logger.error.call_args_list]
    assert messages[0].startswith("Lossless check failed: metric=numerical_equivalence did not meet 2 threshold(s)")
    assert messages[1].startswith("Lossless check location: reason=metric_threshold_exceeded")
    assert "processor=_FakeProcessor" in messages[1]
    assert "decoder=block" in messages[1]
    assert "structure=block.linear" in messages[1]
    assert "side=input" in messages[1]
    assert "failed_metrics: max_abs_diff(value=1.0, expected<=" in messages[2]
    assert "normalized_rmse(value=1.0, expected<=" in messages[2]
    assert "key_metrics: max_abs_diff=1.0" in messages[3]


class _MeanAbsoluteErrorMetricConfig(LosslessMetricConfig):
    model_config = ConfigDict(extra="forbid")
    name: Literal["test_mean_absolute_error"] = "test_mean_absolute_error"
    threshold: float

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, value):
        if value < 0:
            raise SchemaValidateError("test_mean_absolute_error requires a non-negative threshold")
        return value


class _MeanAbsoluteErrorMetric(BaseLosslessMetric):
    def evaluate(self, reference, processed, context):
        value = (reference.float() - processed.float()).abs().mean().item()
        return MetricResult(
            name=self.name,
            passed=value <= self.config.threshold,
            value=value,
            threshold=self.config.threshold,
            observations={"decoder_seen": context.decoder},
        )


def _get_mean_absolute_error_metric_plugin():
    return _MeanAbsoluteErrorMetricConfig, _MeanAbsoluteErrorMetric


register_plugin(_get_mean_absolute_error_metric_plugin)


class _AfterFailureMetricConfig(LosslessMetricConfig):
    model_config = ConfigDict(extra="forbid")
    name: Literal["test_after_failure_metric"] = "test_after_failure_metric"


class _AfterFailureMetric(BaseLosslessMetric):
    calls = 0

    def evaluate(self, _reference, _processed, _context):
        type(self).calls += 1
        return MetricResult(
            name=self.name,
            passed=True,
            value=None,
            threshold=None,
        )


def _get_after_failure_metric_plugin():
    return _AfterFailureMetricConfig, _AfterFailureMetric


register_plugin(_get_after_failure_metric_plugin)


def test_third_party_metric_failure_log_falls_back_to_result_value_and_threshold(
    mocker,
):
    logger = mocker.patch("msmodelslim.processor.lossless.processor.get_logger").return_value
    processor = _metric_processor(
        _RepeatedLinearModule(),
        _FakeConfig(),
        metrics=[{"name": "test_mean_absolute_error", "threshold": 0.1}],
    )

    with pytest.raises(UnexpectedError, match="test_mean_absolute_error"):
        processor._compare_tree(
            torch.zeros(4),
            torch.ones(4),
            request_name="block",
            structure_name="block.linear",
            side="output",
            invocation=0,
        )

    messages = [call.args[0] % tuple(call.args[1:]) for call in logger.error.call_args_list]
    assert "metric=test_mean_absolute_error" in messages[0]
    assert "failed_metrics: test_mean_absolute_error(value=1.0, threshold=0.1)" in messages[2]


def test_third_party_metric_can_replace_default_metric_without_container_changes():
    processor = _metric_processor(
        _RepeatedLinearModule(),
        _FakeConfig(),
        metrics=[{"name": "test_mean_absolute_error", "threshold": 0.1}],
    )

    processor._compare_tree(
        torch.zeros(4),
        torch.full((4,), 0.05),
        request_name="block",
        structure_name="block.linear",
        side="output",
        invocation=0,
    )

    assert len(processor.result["checks"]) == 1
    check = processor.result["checks"][0]
    assert check["metric_name"] == "test_mean_absolute_error"
    assert check["decoder_seen"] == "block"
    assert "gating" not in check


def test_only_explicit_metrics_are_resolved_in_configuration_order():
    processor = _metric_processor(
        _RepeatedLinearModule(),
        _FakeConfig(),
        metrics=[
            {"name": "test_mean_absolute_error", "threshold": 0.1},
            "numerical_equivalence",
        ],
    )

    resolved = [metric.name for metric in processor._resolved_metrics]

    assert resolved == ["test_mean_absolute_error", "numerical_equivalence"]


def test_metrics_execute_in_config_order_and_stop_after_first_failure():
    _AfterFailureMetric.calls = 0
    processor = _metric_processor(
        _RepeatedLinearModule(),
        _FakeConfig(),
        metrics=[
            {"name": "test_mean_absolute_error", "threshold": 0.1},
            "test_after_failure_metric",
        ],
    )

    assert [metric.name for metric in processor._resolved_metrics] == [
        "test_mean_absolute_error",
        "test_after_failure_metric",
    ]

    with pytest.raises(UnexpectedError, match="metric_threshold_exceeded"):
        processor._compare_tree(
            torch.zeros(4),
            torch.ones(4),
            request_name="block",
            structure_name="block.linear",
            side="output",
            invocation=0,
        )

    assert _AfterFailureMetric.calls == 0
    assert [check["metric_name"] for check in processor.result["checks"]] == [
        "test_mean_absolute_error",
    ]


def test_removed_gating_field_and_builtin_cosine_are_rejected():
    with pytest.raises(SchemaValidateError, match="Extra inputs are not permitted"):
        LosslessMetricConfig.model_validate({"name": "numerical_equivalence", "gating": True})
    with pytest.raises(UnsupportedError, match="cosine_similarity"):
        LosslessMetricConfig.model_validate({"name": "cosine_similarity"})


def test_successful_metric_uses_info_log_with_key_fields(mocker):
    logger = mocker.patch("msmodelslim.processor.lossless.processor.get_logger").return_value
    processor = _metric_processor(_RepeatedLinearModule(), _FakeConfig())

    processor._compare_tree(
        torch.ones(2),
        torch.ones(2),
        request_name="block",
        decoder="decoder.0",
        structure_name="block.linear",
        structure_type="Linear",
        side="output",
        invocation=0,
    )

    logger.info.assert_called_once()
    message_template, *message_args = logger.info.call_args.args
    message = message_template % tuple(message_args)
    assert "processor=_FakeProcessor" in message
    assert "decoder=decoder.0" in message
    assert "structure=block.linear" in message
    assert "metric=numerical_equivalence" in message


class _InvalidResultMetric:
    name = "invalid_result_metric"

    def __init__(self, result):
        self.result = result

    def evaluate(self, _reference, _processed, _context):
        return self.result


@pytest.mark.parametrize(
    ("invalid_result", "expected_error"),
    [
        (True, "expected MetricResult"),
        (
            MetricResult(
                name="different_metric",
                passed=True,
                value=1.0,
                threshold=None,
            ),
            "returned result name",
        ),
        (MetricResult(name="invalid_result_metric", passed=0, value=None, threshold=None), "passed with type"),
        (
            MetricResult(name="invalid_result_metric", passed="true", value=None, threshold=None),
            "passed with type",
        ),
        (
            MetricResult(name="invalid_result_metric", passed=torch.tensor(True), value=None, threshold=None),
            "passed with type",
        ),
        (
            MetricResult(
                name="invalid_result_metric",
                passed=True,
                value=None,
                threshold=None,
                observations=[],
            ),
            "observations with type",
        ),
        (
            MetricResult(
                name="invalid_result_metric",
                passed=True,
                value=None,
                threshold=None,
                violations=[],
            ),
            "violations with type",
        ),
        (
            MetricResult(
                name="invalid_result_metric",
                passed=True,
                value=None,
                threshold=None,
                violations=(object(),),
            ),
            "violations[0]",
        ),
        (
            MetricResult(
                name="invalid_result_metric",
                passed=True,
                value=None,
                threshold=None,
                observations={"processor": "shadowed"},
            ),
            "reserved key",
        ),
    ],
)
def test_invalid_third_party_metric_result_fails_fast(invalid_result, expected_error):
    processor = _metric_processor(_RepeatedLinearModule(), _FakeConfig())
    processor._resolved_metrics = (_InvalidResultMetric(invalid_result),)

    with pytest.raises(UnexpectedError, match="metric_execution_error"):
        processor._compare_tree(
            torch.ones(2),
            torch.ones(2),
            request_name="block",
            decoder="decoder.0",
            structure_name="block.linear",
            structure_type="Linear",
            side="output",
            invocation=0,
        )

    anomaly = processor.result["anomalies"][0]
    assert anomaly["metric_name"] == "invalid_result_metric"
    assert expected_error in anomaly["error"]


def test_metric_result_allows_non_float_recordable_values():
    processor = _metric_processor(_RepeatedLinearModule(), _FakeConfig())
    processor._resolved_metrics = (
        _InvalidResultMetric(
            MetricResult(
                name="invalid_result_metric",
                passed=True,
                value={"score": [1, None]},
                threshold=None,
                observations={"details": {"items": [None]}},
                violations=(
                    MetricViolation(
                        name="informational",
                        value={"raw": "value"},
                        threshold=None,
                    ),
                ),
            )
        ),
    )

    processor._compare_tree(
        torch.ones(2),
        torch.ones(2),
        request_name="block",
        structure_name="block.linear",
        side="output",
        invocation=0,
    )

    check = processor.result["checks"][0]
    assert check["metric_value"] == {"score": [1, None]}
    assert check["details"] == {"items": [None]}


def test_floating_comparison_reports_named_distribution_metrics():
    processor = _metric_processor(_RepeatedLinearModule(), _FakeConfig())
    reference = torch.tensor([1.0, 2.0, 3.0])
    processed = reference + torch.tensor([1e-7, -1e-7, 0.0])

    processor._compare_tree(
        reference,
        processed,
        request_name="block",
        structure_name="block.linear",
        side="output",
        invocation=0,
    )

    check = processor.result["checks"][0]
    assert check["mean_abs_diff"] > 0
    assert check["rmse"] > 0
    assert check["reference_rms"] > 0
    assert check["processed_rms"] > 0
    assert 0 < check["normalized_rmse"] < 1


def test_distribution_tolerance_accepts_accumulated_rounding_without_hiding_the_metric():
    processor = _metric_processor(_RepeatedLinearModule(), _FakeConfig())
    reference = torch.linspace(-1.0, 1.0, 4096)
    reference[0] = 0.0
    processed = reference + 5e-7
    processed[0] = 2e-6

    processor._compare_tree(
        reference,
        processed,
        request_name="block",
        structure_name="block.linear",
        side="output",
        invocation=0,
    )

    check = processor.result["checks"][0]
    assert check["passed"]
    assert not check["elementwise_allclose"]
    assert check["distribution_passed"]
    assert check["comparison_mode"] == "distribution"


def test_distribution_tolerance_accepts_bfloat16_accumulated_rounding():
    processor = _metric_processor(_RepeatedLinearModule(), _FakeConfig())
    reference = torch.zeros(4096, dtype=torch.bfloat16)
    reference[1::2] = 1
    processed = reference.clone()
    processed[::2] = 0.05

    processor._compare_tree(
        reference,
        processed,
        request_name="block",
        structure_name="block.linear",
        side="output",
        invocation=0,
    )

    check = processor.result["checks"][0]
    assert check["passed"]
    assert not check["elementwise_allclose"]
    assert check["distribution_passed"]
    assert 0.04 < check["normalized_rmse"] < 0.06


def test_distribution_tolerance_accepts_one_bfloat16_ulp_guard():
    processor = _metric_processor(_RepeatedLinearModule(), _FakeConfig())
    reference = torch.full((1, 4, 4096), 0.3359375, dtype=torch.bfloat16)
    reference.reshape(-1)[0] = 3.390625
    processed = reference.clone()
    processed.reshape(-1)[1] += 0.21875
    processed.reshape(-1)[2:] += 0.0078125

    processor._compare_tree(
        reference,
        processed,
        request_name="block",
        structure_name="block.linear",
        side="output",
        invocation=0,
    )

    check = processor.result["checks"][0]
    assert check["passed"]
    assert not check["elementwise_allclose"]
    assert check["distribution_passed"]
    assert check["max_abs_diff"] == pytest.approx(0.21875)
    assert check["max_abs_diff"] < check["threshold"]["max_abs_diff"]
    assert check["normalized_rmse"] < check["threshold"]["normalized_rmse"]


def test_distribution_tolerance_rejects_bfloat16_error_beyond_one_ulp_guard():
    processor = _metric_processor(_RepeatedLinearModule(), _FakeConfig())
    reference = torch.full((1, 4, 4096), 0.3359375, dtype=torch.bfloat16)
    reference.reshape(-1)[0] = 3.390625
    processed = reference.clone()
    processed.reshape(-1)[1] += 0.25

    with pytest.raises(UnexpectedError, match="metric_threshold_exceeded"):
        processor._compare_tree(
            reference,
            processed,
            request_name="block",
            structure_name="block.linear",
            side="output",
            invocation=0,
        )

    anomaly = processor.result["anomalies"][0]
    assert anomaly["max_abs_diff"] == pytest.approx(0.25)
    assert anomaly["normalized_rmse"] < anomaly["threshold"]["normalized_rmse"]
    assert not anomaly["distribution_passed"]
    assert [item["name"] for item in anomaly["failed_metrics"]] == ["max_abs_diff"]


def test_distribution_tolerance_rejects_bfloat16_broad_distribution_drift():
    processor = _metric_processor(_RepeatedLinearModule(), _FakeConfig())
    reference = torch.full((1, 4, 4096), 0.3359375, dtype=torch.bfloat16)
    reference.reshape(-1)[0] = 3.390625
    processed = reference + torch.full_like(reference, 0.125)

    with pytest.raises(UnexpectedError, match="metric_threshold_exceeded"):
        processor._compare_tree(
            reference,
            processed,
            request_name="block",
            structure_name="block.linear",
            side="output",
            invocation=0,
        )

    anomaly = processor.result["anomalies"][0]
    assert anomaly["max_abs_diff"] < anomaly["threshold"]["max_abs_diff"]
    assert anomaly["normalized_rmse"] > anomaly["threshold"]["normalized_rmse"]
    assert not anomaly["distribution_passed"]
    assert [item["name"] for item in anomaly["failed_metrics"]] == ["normalized_rmse"]


def test_distribution_tolerance_accepts_float32_deep_chain_rounding():
    processor = _metric_processor(_RepeatedLinearModule(), _FakeConfig())
    reference = torch.zeros(4096)
    reference[1::2] = 1
    processed = reference.clone()
    processed[::2] = 7e-6

    processor._compare_tree(
        reference,
        processed,
        request_name="block",
        structure_name="block.linear",
        side="output",
        invocation=0,
    )

    check = processor.result["checks"][0]
    assert check["passed"]
    assert not check["elementwise_allclose"]
    assert check["distribution_passed"]
    assert check["normalized_rmse"] < check["threshold"]["normalized_rmse"]


def test_distribution_tolerance_rejects_one_localized_material_error():
    processor = _metric_processor(_RepeatedLinearModule(), _FakeConfig())
    reference = torch.linspace(-1.0, 1.0, 4096)
    reference[0] = 0.0
    processed = reference.clone()
    processed[0] = 0.1

    with pytest.raises(UnexpectedError, match="metric_threshold_exceeded"):
        processor._compare_tree(
            reference,
            processed,
            request_name="block",
            structure_name="block.linear",
            side="output",
            invocation=0,
        )

    anomaly = processor.result["anomalies"][0]
    assert not anomaly["distribution_passed"]
    assert anomaly["comparison_mode"] == "failed"


def test_empty_floating_tensor_is_a_valid_comparison():
    processor = _metric_processor(_RepeatedLinearModule(), _FakeConfig())

    processor._compare_tree(
        torch.empty(0, 3),
        torch.empty(0, 3),
        request_name="block",
        structure_name="block.linear",
        side="output",
        invocation=0,
    )

    check = processor.result["checks"][0]
    assert check["passed"]
    assert check["metric_value"] == 0.0
    assert check["max_relative_diff"] == 0.0
    assert check["normalized_rmse"] == 0.0
