#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from types import SimpleNamespace
from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
import torch
from pydantic import ConfigDict

from msmodelslim.processor.lossless.metrics import (
    BaseLosslessMetric,
    LosslessMetricConfig,
    MetricContext,
    MetricResult,
    NumericalEquivalenceMetric,
    NumericalEquivalenceMetricConfig,
    create_lossless_metric,
)
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.plugin import register_plugin
from msmodelslim.utils.plugin import plugin_utils as plugin_utils_module


def _context():
    return MetricContext(
        processor="test",
        decoder="decoder",
        structure="linear",
        structure_type="Linear",
        side="output",
        invocation=0,
        path="root",
    )


def test_builtin_metric_config_is_parsed_and_instantiated():
    config = LosslessMetricConfig.model_validate({"name": "numerical_equivalence"})

    assert isinstance(config, NumericalEquivalenceMetricConfig)
    metric = create_lossless_metric(config)
    assert isinstance(metric, NumericalEquivalenceMetric)

    result = metric.evaluate(torch.ones(2), torch.ones(2), _context())
    assert isinstance(result, MetricResult)
    assert result.name == "numerical_equivalence"
    assert result.passed


def test_builtin_metric_rejects_user_threshold_and_options():
    with pytest.raises(SchemaValidateError):
        LosslessMetricConfig.model_validate({"name": "numerical_equivalence", "threshold": 0.1})

    with pytest.raises(SchemaValidateError):
        LosslessMetricConfig.model_validate({"name": "numerical_equivalence", "options": {"atol": 0.1}})


class _MemoryMetricConfig(LosslessMetricConfig):
    model_config = ConfigDict(extra="forbid")
    name: Literal["memory_metric"] = "memory_metric"
    marker: str = "memory"


class _MemoryMetric(BaseLosslessMetric):
    def evaluate(self, reference, processed, context):
        return MetricResult(
            name=self.name,
            passed=True,
            value=self.config.marker,
            threshold=None,
            observations={"decoder": context.decoder},
        )


def _get_memory_metric_plugin():
    return _MemoryMetricConfig, _MemoryMetric


register_plugin(_get_memory_metric_plugin)


def test_memory_registered_plugin_is_parsed_and_created():
    config = LosslessMetricConfig.model_validate({"name": "memory_metric", "marker": "registered"})

    metric = create_lossless_metric(config)
    assert isinstance(metric, _MemoryMetric)
    assert metric.evaluate(None, None, _context()).observations["decoder"] == "decoder"
    assert metric.config.marker == "registered"


class _EntryPointMetricConfig(LosslessMetricConfig):
    model_config = ConfigDict(extra="forbid")
    name: Literal["entry_point_metric"] = "entry_point_metric"


class _EntryPointMetric(BaseLosslessMetric):
    def evaluate(self, reference, processed, context):
        return MetricResult(
            name=self.name,
            passed=True,
            value=context.path,
            threshold=None,
        )


def _get_entry_point_metric_plugin():
    return _EntryPointMetricConfig, _EntryPointMetric


def test_external_entry_point_plugin_is_discovered_without_preimport():
    fake_entry = SimpleNamespace(
        name="entry_point_metric",
        load=MagicMock(return_value=_get_entry_point_metric_plugin),
    )

    with patch.object(plugin_utils_module, "get_entry_points", return_value=[fake_entry]):
        config = LosslessMetricConfig.model_validate({"name": "entry_point_metric"})
        metric = create_lossless_metric(config)

    assert isinstance(config, _EntryPointMetricConfig)
    assert isinstance(metric, _EntryPointMetric)
    assert metric.evaluate(None, None, _context()).value == "root"
    fake_entry.load.assert_called_once()


def test_unknown_metric_plugin_is_rejected():
    with pytest.raises(UnsupportedError, match="No plugin found for type 'missing_metric'"):
        LosslessMetricConfig.model_validate({"name": "missing_metric"})


class _WrongComponentConfig(LosslessMetricConfig):
    model_config = ConfigDict(extra="forbid")
    name: Literal["wrong_component_metric"] = "wrong_component_metric"


class _WrongComponent:
    pass


def _get_wrong_component_plugin():
    return _WrongComponentConfig, _WrongComponent


register_plugin(_get_wrong_component_plugin)


def test_factory_rejects_component_outside_base_metric_contract():
    config = LosslessMetricConfig.model_validate({"name": "wrong_component_metric"})

    with pytest.raises(SchemaValidateError, match="must inherit from BaseLosslessMetric"):
        create_lossless_metric(config)
