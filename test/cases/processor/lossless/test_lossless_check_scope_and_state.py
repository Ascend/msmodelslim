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

import random
from dataclasses import dataclass
from typing import Literal

import pytest
import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.processor.base import AutoProcessorConfig, AutoSessionProcessor
from msmodelslim.processor.lossless.processor import (
    LosslessCheckProcessor,
    LosslessCheckProcessorConfig,
)
from msmodelslim.processor.lossless.protocol import (
    LosslessCheckContext,
    LosslessCheckPlan,
    LosslessCheckTarget,
)
from msmodelslim.utils.exception import SchemaValidateError, UnexpectedError


class _TwoLinearModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(3, 3, bias=False)
        self.second = nn.Linear(3, 3, bias=False)
        self.lossless_targets = {"first": self.first, "second": self.second}
        self.forward_count = 0
        with torch.no_grad():
            self.first.weight.copy_(torch.eye(3))
            self.second.weight.copy_(torch.eye(3))

    def forward(self, hidden_states, *, scale=1.0, mask=None, execute=True):
        del mask
        self.forward_count += 1
        if not execute:
            return hidden_states
        return self.first(hidden_states), self.second(hidden_states) * scale


class _StochasticLinearModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(3, 3, bias=False)
        self.forward_count = 0
        self.forward_grad_enabled = []
        self.forward_training = []
        self.raise_on_call = None
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(3))

    def forward(self, hidden_states, *, scale=1.0, mask=None, execute=True):
        del mask
        self.forward_count += 1
        self.forward_grad_enabled.append(torch.is_grad_enabled())
        self.forward_training.append((self.training, self.dropout.training))
        if not execute:
            return hidden_states
        randomized = self.dropout(hidden_states) + torch.rand_like(hidden_states) + random.random()
        if self.forward_count == self.raise_on_call:
            raise RuntimeError("stochastic forward failed")
        return self.linear(randomized) * scale


class _ScopeConfig(AutoProcessorConfig):
    type: Literal["lossless_scope_test_fake"] = "lossless_scope_test_fake"
    fault_target: str = ""


@dataclass(frozen=True)
class _ScopeContext(LosslessCheckContext):
    pass


def _scope_targets(request):
    target_modules = getattr(request.module, "lossless_targets", None)
    if target_modules is None:
        target_modules = {"linear": request.module.linear}
    return tuple(
        LosslessCheckTarget(
            name=f"{request.name}.{name}",
            path=f"{request.name}.{name}",
            decoder=request.name,
            structure_type=module.__class__.__name__,
        )
        for name, module in target_modules.items()
    )


@QABCRegistry.register(dispatch_key=_ScopeConfig, abc_class=AutoSessionProcessor)
class _ScopeProcessor(AutoSessionProcessor):
    def __init__(self, model, config, adapter=None):
        super().__init__(model)
        self.config = config
        self.adapter = adapter
        self.events = getattr(model, "lossless_events", [])
        model.lossless_events = self.events

    def prepare_lossless_check(self):
        return LosslessCheckPlan(expected_targets=())

    def create_lossless_check_context(self, request, reference_datas):
        return _ScopeContext(
            targets=_scope_targets(request),
            reference_datas=reference_datas,
            covered_targets=(),
            unresolved_targets=(),
        )

    @staticmethod
    def align_lossless_check_io(_context, _target, reference, processed):
        return reference, processed

    def process(self, request):
        self.events.append("scope.process")
        if self.config.fault_target:
            with torch.no_grad():
                getattr(request.module, self.config.fault_target).weight.add_(0.5)


class _Adapter:
    @staticmethod
    def generate_model_forward(_model, _inputs):
        yield


def _request(module, *, execute=True, name="block"):
    return BatchProcessRequest(
        name=name,
        module=module,
        datas=[
            (
                (torch.tensor([[1.0, 2.0, 3.0]]),),
                {"scale": 2.0, "mask": torch.tensor([[1, 0, 1]]), "execute": execute},
            )
        ],
        outputs=None,
    )


def _processor(model, *, include=None, exclude=None, fault_target=""):
    config_kwargs = {
        "configs": [_ScopeConfig(fault_target=fault_target)],
        "metrics": ["numerical_equivalence"],
    }
    if include is not None:
        config_kwargs["include"] = include
    if exclude is not None:
        config_kwargs["exclude"] = exclude
    return LosslessCheckProcessor(
        model,
        LosslessCheckProcessorConfig(**config_kwargs),
        _Adapter(),
    )


def test_config_exposes_include_exclude_and_required_metrics():
    config = LosslessCheckProcessorConfig(
        configs=[_ScopeConfig()],
        metrics=["numerical_equivalence"],
    )
    assert config.include == ["*"]
    assert config.exclude == []
    assert set(config.model_dump()) == {
        "type",
        "configs",
        "include",
        "exclude",
        "metrics",
    }


@pytest.mark.parametrize(
    "include",
    [
        [],
        [""],
        ["   "],
        ["x" * 201],
    ],
)
def test_include_rejects_empty_or_oversized_values(include):
    with pytest.raises(SchemaValidateError):
        LosslessCheckProcessorConfig(
            configs=[_ScopeConfig()],
            include=include,
            metrics=["numerical_equivalence"],
        )


def test_include_limits_checked_targets_without_skipping_child_processing():
    model = _TwoLinearModule()
    processor = _processor(
        model,
        include=["block.first"],
    )

    processor.pre_run()
    processor.process(_request(model))
    processor.post_run()

    assert processor.result["checked_targets"] == ["block.first"]
    assert model.forward_count == 2
    assert model.lossless_events.count("scope.process") == 1


def test_unmatched_include_runs_processed_forward_once_and_is_reported():
    model = _TwoLinearModule()
    processor = _processor(
        model,
        include=["block.missing"],
    )

    processor.pre_run()
    processor.process(_request(model))
    processor.post_run()

    assert model.forward_count == 1
    assert model.lossless_events.count("scope.process") == 1
    assert processor.result["include"] == ["block.missing"]
    assert not processor.result["matched_targets"]
    assert not processor.result["checked_targets"]
    assert processor.result["unmatched_include"] == ["block.missing"]


def test_fault_in_unselected_target_is_processed_but_not_compared():
    model = _TwoLinearModule()
    processor = _processor(
        model,
        include=["block.first"],
        fault_target="second",
    )

    processor.pre_run()
    processor.process(_request(model))
    processor.post_run()

    assert processor.result["passed"]
    assert processor.result["checked_targets"] == ["block.first"]
    assert torch.allclose(model.second.weight, torch.eye(3) + 0.5)
    assert all(check["side"] != "request_output" for check in processor.result["checks"])


def test_fault_in_selected_target_fails_with_precise_structure():
    model = _TwoLinearModule()
    processor = _processor(
        model,
        include=["block.first"],
        fault_target="first",
    )

    processor.pre_run()
    with pytest.raises(UnexpectedError, match="block.first.*side=output"):
        processor.process(_request(model))

    anomaly = processor.result["anomalies"][0]
    assert anomaly["structure"] == "block.first"
    assert anomaly["side"] == "output"
    assert anomaly["reason"] == "metric_threshold_exceeded"


def test_dual_forward_replays_rng_uses_eval_no_grad_and_restores_mixed_training_state():
    model = _StochasticLinearModule()
    model.train()
    model.dropout.eval()
    original_training = {name: module.training for name, module in model.named_modules()}

    torch.manual_seed(1234)
    random.seed(1234)
    model.eval()
    with torch.no_grad():
        model(torch.tensor([[1.0, 2.0, 3.0]]), scale=2.0)
    expected_rng_state = torch.get_rng_state().clone()
    expected_python_rng_state = random.getstate()
    model.train()
    model.dropout.eval()
    model.forward_count = 0
    model.forward_grad_enabled.clear()

    torch.manual_seed(1234)
    random.seed(1234)
    processor = _processor(model)
    request = _request(model)
    processor.pre_run()
    processor.process(request)
    processor.post_run()

    assert processor.result["passed"]
    assert model.forward_count == 2
    assert model.forward_grad_enabled == [False, False]
    assert torch.equal(torch.get_rng_state(), expected_rng_state)
    assert random.getstate() == expected_python_rng_state
    assert {name: module.training for name, module in model.named_modules()} == original_training
    assert all(not output.requires_grad for output in request.outputs)


def test_failed_dual_forward_restores_rng_and_mixed_training_state():
    model = _StochasticLinearModule()
    model.train()
    model.dropout.eval()
    original_training = {name: module.training for name, module in model.named_modules()}
    torch.manual_seed(4321)
    random.seed(4321)
    original_rng_state = torch.get_rng_state().clone()
    original_python_rng_state = random.getstate()
    processor = _processor(model, fault_target="linear")

    processor.pre_run()
    with pytest.raises(UnexpectedError, match="metric_threshold_exceeded"):
        processor.process(_request(model))

    assert torch.equal(torch.get_rng_state(), original_rng_state)
    assert random.getstate() == original_python_rng_state
    assert {name: module.training for name, module in model.named_modules()} == original_training


def test_processed_forward_exception_restores_rng_training_state_and_hooks():
    model = _StochasticLinearModule()
    model.train()
    model.dropout.eval()
    model.raise_on_call = 2
    original_training = {name: module.training for name, module in model.named_modules()}
    original_pre_hooks = len(model.linear._forward_pre_hooks)
    original_forward_hooks = len(model.linear._forward_hooks)
    torch.manual_seed(2468)
    random.seed(2468)
    original_rng_state = torch.get_rng_state().clone()
    original_python_rng_state = random.getstate()
    processor = _processor(model)

    processor.pre_run()
    with pytest.raises(RuntimeError, match="stochastic forward failed"):
        processor.process(_request(model))

    assert torch.equal(torch.get_rng_state(), original_rng_state)
    assert random.getstate() == original_python_rng_state
    assert {name: module.training for name, module in model.named_modules()} == original_training
    assert len(model.linear._forward_pre_hooks) == original_pre_hooks
    assert len(model.linear._forward_hooks) == original_forward_hooks
    assert processor.active_capture_count == 0


def test_uninitialized_accelerator_rng_is_not_read(monkeypatch):
    model = _StochasticLinearModule()

    def fail_if_called():
        raise AssertionError("uninitialized accelerator RNG must not be read")

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", fail_if_called)
    processor = _processor(model)

    processor.pre_run()
    processor.process(_request(model))
    processor.post_run()

    assert processor.result["passed"]


def test_overlapping_include_patterns_are_deduplicated_and_case_sensitive():
    model = _TwoLinearModule()
    processor = _processor(
        model,
        include=["block.second", "block.*", "BLOCK.*"],
    )

    processor.pre_run()
    processor.process(_request(model))
    processor.post_run()

    assert processor.result["matched_targets"] == ["block.first", "block.second"]
    assert processor.result["checked_targets"] == ["block.first", "block.second"]
    assert processor.result["unmatched_include"] == ["BLOCK.*"]
    assert len(processor.result["checked_targets"]) == len(set(processor.result["checked_targets"]))


def test_unmatched_include_warns_once_per_run_after_all_requests(mocker):
    logger = mocker.patch("msmodelslim.processor.lossless.processor.get_logger").return_value
    model = _TwoLinearModule()
    processor = _processor(
        model,
        include=["missing.*"],
    )

    processor.pre_run()
    processor.process(_request(model, name="block0"))
    processor.process(_request(model, name="block1"))
    processor.post_run()

    logger.warning.assert_called_once_with(
        "Lossless check include patterns did not match any target: %s",
        ["missing.*"],
    )


def test_exclude_wins_when_include_and_exclude_overlap():
    model = _TwoLinearModule()
    processor = _processor(
        model,
        include=["block.*"],
        exclude=["block.second"],
    )

    processor.pre_run()
    processor.process(_request(model))
    processor.post_run()

    assert processor.result["checked_targets"] == ["block.first"]
    assert not processor.result["unmatched_exclude"]


def test_include_and_exclude_support_repository_brace_syntax():
    model = _TwoLinearModule()
    processor = _processor(
        model,
        include=["block.{first,second}"],
        exclude=["block.{second,third}"],
    )

    processor.pre_run()
    processor.process(_request(model))
    processor.post_run()

    assert processor.result["checked_targets"] == ["block.first"]
    assert not processor.result["unmatched_include"]
    assert not processor.result["unmatched_exclude"]
