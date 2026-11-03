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
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

# Exercise the production SmoothQuant transform through the checking container.

from copy import deepcopy

import pytest
import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.graph.adapter_types import AdapterConfig, MappingConfig
from msmodelslim.ir.norm_bias import RMSNormBias
from msmodelslim.processor.anti_outlier.smooth_quant import (
    SmoothQuantProcessorConfig,
)
from msmodelslim.processor.anti_outlier.smooth_quant.interface import SmoothQuantInterface
from msmodelslim.processor.lossless.processor import LosslessCheckProcessor, LosslessCheckProcessorConfig
from msmodelslim.utils.exception import UnsupportedError, UnexpectedError


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = RMSNormBias(16)
        self.first = nn.Linear(16, 16)
        self.second = nn.Linear(16, 16)
        self.dormant = nn.Linear(16, 16)

    def forward(self, hidden_states, execute=True):
        if not execute:
            raise RuntimeError("calibration forward failed")
        hidden_states = self.norm(hidden_states)
        return self.first(hidden_states) + self.second(hidden_states)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _Block()


class _Adapter(SmoothQuantInterface):
    def get_adapter_config_for_subgraph(self):
        return [
            AdapterConfig("norm-linear", MappingConfig(source="block.norm", targets=["block.first", "block.second"]))
        ]

    @staticmethod
    def generate_model_forward(_model, _inputs):
        yield


def _setup(*, symmetric=True, include=None, exclude=None, seed=31):
    torch.manual_seed(seed)
    model = _Model()
    data = torch.randn(2, 5, 16) * torch.arange(1, 17)
    config = dict(configs=[SmoothQuantProcessorConfig(symmetric=symmetric)], metrics=["numerical_equivalence"])
    if include is not None:
        config["include"] = include
    if exclude is not None:
        config["exclude"] = exclude
    processor = LosslessCheckProcessor(model, LosslessCheckProcessorConfig(**config), _Adapter())
    request = BatchProcessRequest(name="block", module=model.block, datas=[((data,), {})])
    return model, processor, request


@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("seed", [31, 37, 41])
def test_real_smoothing_changes_weights_and_checks_all_executed_linears(symmetric, seed):
    model, processor, request = _setup(symmetric=symmetric, seed=seed, exclude=["block.dormant"])
    before = deepcopy(model)
    processor.pre_run()
    processor.process(request)
    processor.post_run()
    assert processor.result["passed"] is True
    assert processor.result["total_checks"] == 4
    assert processor.result["checked_targets"] == ["block.first", "block.second"]
    assert not torch.allclose(before.block.first.weight, model.block.first.weight)
    assert torch.allclose(before.block(*request.datas[0][0]), request.outputs[0], atol=1e-5, rtol=1e-4)
    assert processor.active_capture_count == 0
    assert not processor.processor.hook_manager.hook_handles
    assert not processor.processor.stats_collector.act_stats


def test_dormant_linear_is_not_reported_as_passed():
    _, processor, request = _setup()
    processor.pre_run()
    processor.process(request)
    processor.post_run()
    assert processor.result["passed"] is None
    assert processor.result["skipped_targets"] == ["block.dormant"]
    assert processor.result["checked_targets"] == ["block.first", "block.second"]


def test_weight_fault_is_localized_with_independent_alignment(monkeypatch):
    _, processor, request = _setup(include=["block.first", "block.second"])
    child = processor.processor
    actual = child.apply_smooth_algorithm

    def corrupt(subgraph, linear_names):
        actual(subgraph, linear_names)
        with torch.no_grad():
            subgraph.linears[1].weight.mul_(1.5)

    monkeypatch.setattr(child, "apply_smooth_algorithm", corrupt)
    processor.pre_run()
    with pytest.raises(UnexpectedError):
        processor.process(request)
    assert processor.result["passed"] is False
    first = processor.result["anomalies"][0]
    assert first["structure"] == "block.second"
    assert first["side"] == "output"
    assert first["reason"] == "metric_threshold_exceeded"
    assert processor.active_capture_count == 0
    assert not child.hook_manager.hook_handles


def test_scope_and_exclusion_still_apply_to_production_smoothing():
    _, processor, request = _setup(include=["block.*"], exclude=["block.second", "block.dormant"])
    processor.pre_run()
    processor.process(request)
    processor.post_run()
    assert processor.result["passed"] is True
    assert processor.result["total_checks"] == 2
    assert processor.result["checked_targets"] == ["block.first"]


def test_unmatched_check_scope_still_runs_real_smoothing():
    model, processor, request = _setup(include=["does.not.exist"])
    before = model.block.first.weight.detach().clone()
    processor.pre_run()
    processor.process(request)
    processor.post_run()
    assert not torch.equal(before, model.block.first.weight)
    assert processor.result["passed"] is None
    assert processor.result["total_checks"] == 0
    assert processor.result["unmatched_include"] == ["does.not.exist"]


def test_calibration_forward_does_not_change_comparison_rng(monkeypatch):
    model, processor, request = _setup(exclude=["block.dormant"])
    before = deepcopy(model)
    original_forward = model.block.forward
    monkeypatch.setattr(model.block, "forward", lambda value: original_forward(value + torch.rand_like(value)))
    rng_before = torch.random.get_rng_state()
    value = request.datas[0][0][0]
    expected_output = before.block(value + torch.rand_like(value))
    expected_rng = torch.random.get_rng_state()
    torch.random.set_rng_state(rng_before)
    processor.pre_run()
    processor.process(request)
    processor.post_run()
    assert processor.result["passed"] is True
    assert torch.allclose(request.outputs[0], expected_output, atol=1e-5, rtol=1e-4)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)


def test_auto_detect_without_declared_plan_is_explicitly_unsupported():
    model, _, _ = _setup()

    class _Undeclared:
        @staticmethod
        def generate_model_forward(_model, _inputs):
            yield

    processor = LosslessCheckProcessor(
        model,
        LosslessCheckProcessorConfig(configs=[SmoothQuantProcessorConfig()], metrics=["numerical_equivalence"]),
        _Undeclared(),
    )
    with pytest.raises(UnsupportedError, match="declared norm-linear"):
        processor.pre_run()


def test_calibration_failure_releases_statistics_hooks(monkeypatch):
    _, processor, request = _setup(exclude=["block.dormant"])
    child = processor.processor

    def fail(_request):
        raise RuntimeError("calibration failed after hook registration")

    monkeypatch.setattr(child, "postprocess", fail)
    processor.pre_run()
    with pytest.raises(RuntimeError, match="after hook registration"):
        processor.process(request)
    assert not child.hook_manager.hook_handles
    assert not child.stats_collector.act_stats
    assert processor.active_capture_count == 0
    assert processor.result["passed"] is False


@pytest.mark.parametrize("phase", ["prepare_lossless_check", "pre_run", "process", "postprocess", "post_run"])
def test_container_aborts_temporary_state_on_lifecycle_failure(monkeypatch, phase):
    _, processor, request = _setup(exclude=["block.dormant"])
    child = processor.processor
    original = getattr(child, phase)

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError(f"injected {phase} failure")

    monkeypatch.setattr(child, phase, fail)
    if phase in {"prepare_lossless_check", "pre_run"}:
        with pytest.raises(RuntimeError, match=phase):
            processor.pre_run()
    else:
        processor.pre_run()
        if phase == "post_run":
            processor.process(request)
            with pytest.raises(RuntimeError, match=phase):
                processor.post_run()
        else:
            with pytest.raises(RuntimeError, match=phase):
                processor.process(request)
    assert not child._lossless_check_active
    assert not child._lossless_alignments
    assert not child.hook_manager.hook_handles
    assert not child.stats_collector.act_stats
    assert processor.active_capture_count == 0
