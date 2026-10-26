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

# pylint: disable=abstract-method,redefined-outer-name
from typing import List

import pytest
import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.svd_residual.processor import (
    DEFAULT_SEED,
    SVDResidualProcessor,
    SVDResidualProcessorConfig,
    SVD_LOWRANK_L1_PARAM_NAME,
    SVD_LOWRANK_L2_PARAM_NAME,
)
from msmodelslim.utils.exception import SchemaValidateError


class DummyLogger:
    def __init__(self):
        self.messages: List[str] = []

    def warning(self, msg: str, *args) -> None:
        self.messages.append(msg % args if args else msg)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 6, bias=False)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(3, 4, bias=False)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


@pytest.fixture
def simple_model():
    torch.manual_seed(DEFAULT_SEED)
    return SimpleModel()


@pytest.fixture
def processor_all(simple_model):
    config = SVDResidualProcessorConfig(rank=2, include=["*"], exclude=[])
    return SVDResidualProcessor(model=simple_model, config=config, adapter=None)


def _request(module: nn.Module, name: str = "root") -> BatchProcessRequest:
    return BatchProcessRequest(name=name, module=module, datas=[])


def _lowrank_reconstruction(linear: nn.Linear) -> torch.Tensor:
    l1 = getattr(linear, SVD_LOWRANK_L1_PARAM_NAME)
    l2 = getattr(linear, SVD_LOWRANK_L2_PARAM_NAME)
    return l2 @ l1


def test_svd_residual_processor_config_returns_expected_defaults_when_created():
    cfg = SVDResidualProcessorConfig()
    assert (cfg.type, cfg.rank, cfg.include, cfg.exclude) == ("svd_res", 32, ["*"], [])


def test_svd_residual_processor_config_raises_validation_error_when_rank_is_zero():
    with pytest.raises(SchemaValidateError, match=r"Input should be greater than 0"):
        SVDResidualProcessorConfig(rank=0)


def test_svd_residual_processor_returns_true_when_is_data_free_called(simple_model):
    proc = SVDResidualProcessor(
        model=simple_model,
        config=SVDResidualProcessorConfig(rank=2),
        adapter=None,
    )
    assert proc.is_data_free() is True


def test_svd_residual_processor_calls_decompose_when_process_called(monkeypatch, simple_model):
    proc = SVDResidualProcessor(
        model=simple_model,
        config=SVDResidualProcessorConfig(rank=2),
        adapter=None,
    )
    calls = []

    def _record_decompose(prefix, module):
        calls.append((prefix, module))

    monkeypatch.setattr(proc, "decompose", _record_decompose)
    request = _request(simple_model)
    proc.process(request)
    assert calls == [("root", simple_model)]


def test_svd_residual_processor_calls_set_hook_ir_when_postprocess_called(monkeypatch, simple_model):
    proc = SVDResidualProcessor(
        model=simple_model,
        config=SVDResidualProcessorConfig(rank=2),
        adapter=None,
    )
    calls = []

    def _record_set_hook_ir(module):
        calls.append(module)

    monkeypatch.setattr(proc, "set_hook_ir", _record_set_hook_ir)
    request = _request(simple_model)
    proc.postprocess(request)
    assert calls == [simple_model]


def test_svd_residual_processor_writes_residual_weight_when_process_matches_linear(simple_model):
    proc = SVDResidualProcessor(
        model=simple_model,
        config=SVDResidualProcessorConfig(rank=2, include=["root.linear1"], exclude=[]),
        adapter=None,
    )
    original_weight = simple_model.linear1.weight.detach().clone()
    proc.process(_request(simple_model))
    reconstructed = _lowrank_reconstruction(simple_model.linear1)
    assert torch.allclose(simple_model.linear1.weight.detach(), original_weight - reconstructed, atol=1e-6)


def test_svd_residual_processor_registers_lowrank_parameters_when_process_matches_linear(simple_model):
    proc = SVDResidualProcessor(
        model=simple_model,
        config=SVDResidualProcessorConfig(rank=2, include=["root.linear1"], exclude=[]),
        adapter=None,
    )
    proc.process(_request(simple_model))
    l1 = getattr(simple_model.linear1, SVD_LOWRANK_L1_PARAM_NAME)
    l2 = getattr(simple_model.linear1, SVD_LOWRANK_L2_PARAM_NAME)
    assert l1.shape == (2, 4)
    assert l2.shape == (6, 2)
    assert l1.requires_grad is False
    assert l2.requires_grad is False


def test_svd_residual_processor_keeps_excluded_linear_unchanged_when_process_matches_include(simple_model):
    proc = SVDResidualProcessor(
        model=simple_model,
        config=SVDResidualProcessorConfig(
            rank=2,
            include=["root.linear1", "root.linear2"],
            exclude=["root.linear2"],
        ),
        adapter=None,
    )
    original_weight = simple_model.linear2.weight.detach().clone()
    proc.process(_request(simple_model))
    assert torch.equal(simple_model.linear2.weight.detach(), original_weight)
    assert not hasattr(simple_model.linear2, SVD_LOWRANK_L1_PARAM_NAME)
    assert not hasattr(simple_model.linear2, SVD_LOWRANK_L2_PARAM_NAME)


def test_svd_residual_processor_keeps_all_linears_unchanged_when_include_does_not_match(simple_model):
    proc = SVDResidualProcessor(
        model=simple_model,
        config=SVDResidualProcessorConfig(rank=2, include=["root.missing"], exclude=[]),
        adapter=None,
    )
    weights = [module.weight.detach().clone() for module in (simple_model.linear1, simple_model.linear2)]
    proc.process(_request(simple_model))
    assert torch.equal(simple_model.linear1.weight.detach(), weights[0])
    assert torch.equal(simple_model.linear2.weight.detach(), weights[1])
    assert not hasattr(simple_model.linear1, SVD_LOWRANK_L1_PARAM_NAME)
    assert not hasattr(simple_model.linear2, SVD_LOWRANK_L1_PARAM_NAME)


def test_svd_residual_processor_raises_schema_validate_error_when_rank_exceeds_weight_dimension(simple_model):
    proc = SVDResidualProcessor(
        model=simple_model,
        config=SVDResidualProcessorConfig(rank=5, include=["root.linear1"], exclude=[]),
        adapter=None,
    )
    with pytest.raises(
        SchemaValidateError,
        match=r"rank \(5\) must not exceed min\(out_features, in_features\) = 4",
    ):
        proc.process(_request(simple_model))


def test_svd_residual_processor_registers_forward_hook_when_postprocess_follows_process(simple_model):
    proc = SVDResidualProcessor(
        model=simple_model,
        config=SVDResidualProcessorConfig(rank=2, include=["root.linear1"], exclude=[]),
        adapter=None,
    )
    request = _request(simple_model)
    proc.process(request)
    proc.postprocess(request)
    assert len(simple_model.linear1._forward_pre_hooks) == 1
    assert len(simple_model.linear2._forward_pre_hooks) == 0


def test_svd_residual_processor_skips_hook_when_postprocess_precedes_process(simple_model):
    proc = SVDResidualProcessor(
        model=simple_model,
        config=SVDResidualProcessorConfig(rank=2, include=["root.linear1"], exclude=[]),
        adapter=None,
    )
    proc.postprocess(_request(simple_model))
    assert len(simple_model.linear1._forward_pre_hooks) == 0
    assert len(simple_model.linear2._forward_pre_hooks) == 0


def test_svd_residual_processor_registers_nested_forward_hooks_when_postprocess_follows_process():
    class Nested(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = SimpleModel()

    torch.manual_seed(DEFAULT_SEED)
    nested = Nested()
    proc = SVDResidualProcessor(
        model=nested,
        config=SVDResidualProcessorConfig(rank=2, include=["root.block.linear1", "root.block.linear2"]),
        adapter=None,
    )
    request = _request(nested)
    proc.process(request)
    proc.postprocess(request)
    assert len(nested.block.linear1._forward_pre_hooks) == 1
    assert len(nested.block.linear2._forward_pre_hooks) == 1


def test_svd_residual_processor_logs_no_warning_when_post_run_patterns_all_match(monkeypatch, processor_all):
    from msmodelslim.processor.svd_residual import processor as processor_mod

    logger = DummyLogger()
    monkeypatch.setattr(processor_mod, "get_logger", lambda: logger)
    processor_all.process(_request(processor_all.model))
    processor_all.post_run()
    assert not logger.messages


def test_svd_residual_processor_logs_unmatched_patterns_when_post_run_has_missing_names(monkeypatch, simple_model):
    from msmodelslim.processor.svd_residual import processor as processor_mod

    logger = DummyLogger()
    monkeypatch.setattr(processor_mod, "get_logger", lambda: logger)
    proc = SVDResidualProcessor(
        model=simple_model,
        config=SVDResidualProcessorConfig(
            rank=2,
            include=["root.linear1", "missing_include"],
            exclude=["missing_exclude"],
        ),
        adapter=None,
    )
    proc.process(_request(simple_model))
    proc.post_run()
    assert logger.messages == [
        "These include patterns are not matched any module, please ensure this is as expected: ['missing_include']",
        "These exclude patterns are not matched any module, please ensure this is as expected: ['missing_exclude']",
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
