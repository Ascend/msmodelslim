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

# pylint: disable=redefined-outer-name
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
    _warning_unmatched_pattern,
)


class DummyLogger:
    def __init__(self):
        self.messages: List[str] = []

    def warning(self, msg: str, *args) -> None:
        self.messages.append(msg % args if args else msg)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # out_features, in_features
        self.linear1 = nn.Linear(4, 6, bias=False)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(3, 4, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


@pytest.fixture(scope="module")
def simple_model():
    torch.manual_seed(DEFAULT_SEED)
    return SimpleModel()


@pytest.fixture(scope="module")
def processor_all(simple_model):
    config = SVDResidualProcessorConfig(rank=2, include=["*"], exclude=[])
    return SVDResidualProcessor(model=simple_model, config=config, adapter=None)


@pytest.fixture(scope="module")
def processor_custom(simple_model):
    include = ["root.linear1", "root.linear2"]
    exclude = ["root.linear2"]
    config = SVDResidualProcessorConfig(rank=2, include=include, exclude=exclude)
    return SVDResidualProcessor(model=simple_model, config=config, adapter=None)


def test_SVDResidualProcessorConfig_given_default_when_created_then_has_expected_defaults():
    cfg = SVDResidualProcessorConfig()
    assert cfg.type == "svd_res"
    assert cfg.rank == 32
    assert cfg.include == ["*"]
    assert cfg.exclude == []


def test__warning_unmatched_pattern_given_config_set_with_extra_keys_when_not_only_wildcard_then_log_warning(
    monkeypatch,
):
    class DummyConfigSet:
        def unmatched_keys(self):
            return {"*", "foo", "bar"}

    dummy_logger = DummyLogger()
    from msmodelslim.processor.svd_residual import processor as processor_mod

    monkeypatch.setattr(processor_mod, "get_logger", lambda: dummy_logger)
    _warning_unmatched_pattern("include", DummyConfigSet())
    assert len(dummy_logger.messages) == 1
    msg = dummy_logger.messages[0]
    assert "foo" in msg and "bar" in msg
    assert "*" not in msg


def test__warning_unmatched_pattern_given_only_wildcard_unmatched_when_called_then_not_log_warning(monkeypatch):
    class DummyConfigSet:
        def unmatched_keys(self):
            return {"*"}

    dummy_logger = DummyLogger()
    from msmodelslim.processor.svd_residual import processor as processor_mod

    monkeypatch.setattr(processor_mod, "get_logger", lambda: dummy_logger)
    _warning_unmatched_pattern("exclude", DummyConfigSet())
    assert not dummy_logger.messages


def test_SVDResidualProcessor_is_data_free_given_any_model_when_called_then_return_true(simple_model):
    cfg = SVDResidualProcessorConfig(rank=4)
    proc = SVDResidualProcessor(model=simple_model, config=cfg, adapter=None)
    assert proc.is_data_free() is True


def test_SVDResidualProcessor_post_run_given_processor_when_called_then_delegate_to_warning_helper(
    monkeypatch, processor_all
):
    called_args = []

    from msmodelslim.processor.svd_residual import processor as processor_mod

    def fake_warning(name, config_set):
        called_args.append((name, config_set))

    monkeypatch.setattr(processor_mod, "_warning_unmatched_pattern", fake_warning)
    processor_all.post_run()
    names = [name for name, _ in called_args]
    assert "include" in names
    assert "exclude" in names
    assert any(config_set is processor_all.include for _, config_set in called_args)
    assert any(config_set is processor_all.exclude for _, config_set in called_args)


def test_process_given_batch_request_when_valid_then_decompose_called_once(monkeypatch, processor_all, simple_model):
    called = {}

    def fake_decompose(prefix, module):
        called["prefix"] = prefix
        called["module"] = module

    monkeypatch.setattr(processor_all, "decompose", fake_decompose)
    request = BatchProcessRequest(name="root", module=simple_model)
    processor_all.process(request)
    assert called["prefix"] == "root"
    assert called["module"] is simple_model


def test_process_given_batch_request_when_invalid_name_then_decompose_still_called(
    monkeypatch, processor_all, simple_model
):
    called = {}

    def fake_decompose(prefix, module):
        called["prefix"] = prefix
        called["module"] = module

    monkeypatch.setattr(processor_all, "decompose", fake_decompose)
    request = BatchProcessRequest(name="", module=simple_model)
    processor_all.process(request)
    assert called["prefix"] == ""
    assert called["module"] is simple_model


def test_postprocess_given_batch_request_when_called_then_set_hook_ir_invoked(monkeypatch, processor_all, simple_model):
    called = {}

    def fake_set_hook_ir(block):
        called["block"] = block

    monkeypatch.setattr(processor_all, "set_hook_ir", fake_set_hook_ir)
    request = BatchProcessRequest(name="root", module=simple_model)
    processor_all.postprocess(request)
    assert called["block"] is simple_model


def test_postprocess_given_batch_request_when_module_is_submodule_then_set_hook_ir_receives_submodule(
    monkeypatch, processor_all, simple_model
):
    called = {}

    def fake_set_hook_ir(block):
        called["block"] = block

    monkeypatch.setattr(processor_all, "set_hook_ir", fake_set_hook_ir)
    request = BatchProcessRequest(name="root.linear1", module=simple_model.linear1)
    processor_all.postprocess(request)
    assert called["block"] is simple_model.linear1


def test__should_process_module_given_non_linear_module_when_checked_then_return_false(processor_all):
    non_linear = nn.ReLU()
    result = processor_all._should_process_module("root.relu", non_linear)
    assert result is False


def test__should_process_module_given_linear_not_in_include_when_checked_then_return_false(simple_model):
    cfg = SVDResidualProcessorConfig(rank=2, include=["some_other_pattern"], exclude=[])
    proc = SVDResidualProcessor(model=simple_model, config=cfg, adapter=None)
    linear = nn.Linear(2, 2, bias=False)
    result = proc._should_process_module("target.linear", linear)
    assert result is False


def test__should_process_module_given_linear_in_exclude_when_checked_then_return_false(simple_model):
    cfg = SVDResidualProcessorConfig(rank=2, include=["target.linear"], exclude=["target.linear"])
    proc = SVDResidualProcessor(model=simple_model, config=cfg, adapter=None)
    linear = nn.Linear(2, 2, bias=False)
    result = proc._should_process_module("target.linear", linear)
    assert result is False


def test__should_process_module_given_linear_in_include_and_not_in_exclude_when_checked_then_return_true(simple_model):
    cfg = SVDResidualProcessorConfig(rank=2, include=["*"], exclude=["other.linear"])
    proc = SVDResidualProcessor(model=simple_model, config=cfg, adapter=None)
    linear = nn.Linear(2, 2, bias=False)
    result = proc._should_process_module("myblock.linear", linear)
    assert result is True


def test_decompose_given_matching_include_when_linear_layers_then_residual_and_svd_params_registered(
    processor_all, simple_model
):
    torch.manual_seed(DEFAULT_SEED)
    model_copy = SimpleModel()
    original_weights = {
        "root.linear1": model_copy.linear1.weight.detach().clone(),
        "root.linear2": model_copy.linear2.weight.detach().clone(),
    }
    # override model in processor to avoid mutating fixture
    processor_all.model = model_copy
    processor_all.decompose("root", model_copy)

    for name, linear in [
        ("root.linear1", model_copy.linear1),
        ("root.linear2", model_copy.linear2),
    ]:
        assert isinstance(getattr(linear, SVD_LOWRANK_L1_PARAM_NAME), nn.Parameter)
        assert isinstance(getattr(linear, SVD_LOWRANK_L2_PARAM_NAME), nn.Parameter)
        left = getattr(linear, SVD_LOWRANK_L1_PARAM_NAME).data
        right = getattr(linear, SVD_LOWRANK_L2_PARAM_NAME).data
        reconstructed = right @ left
        residual = linear.weight.data
        restored = reconstructed + residual
        assert torch.allclose(restored, original_weights[name], atol=1e-4, rtol=1e-4)
        assert reconstructed.shape[0] == linear.weight.shape[0]
        assert reconstructed.shape[1] == linear.weight.shape[1]
        assert right.shape[1] == processor_all.config.rank
        assert left.shape[0] == processor_all.config.rank
        assert left.shape[1] == linear.weight.shape[1]


def test_decompose_given_exclude_pattern_when_second_linear_excluded_then_only_first_is_modified(
    processor_custom, simple_model
):
    model_copy = SimpleModel()
    original_l1 = model_copy.linear1.weight.detach().clone()
    original_l2 = model_copy.linear2.weight.detach().clone()
    processor_custom.model = model_copy
    processor_custom.decompose("root", model_copy)

    l1 = model_copy.linear1
    l2 = model_copy.linear2

    assert not torch.allclose(l1.weight.data, original_l1)
    assert isinstance(getattr(l1, SVD_LOWRANK_L1_PARAM_NAME), nn.Parameter)
    assert isinstance(getattr(l1, SVD_LOWRANK_L2_PARAM_NAME), nn.Parameter)

    assert torch.allclose(l2.weight.data, original_l2)
    assert not hasattr(l2, SVD_LOWRANK_L1_PARAM_NAME)
    assert not hasattr(l2, SVD_LOWRANK_L2_PARAM_NAME)


def test__decompose_linear_layer_given_first_call_when_no_params_then_register_new_parameters(processor_all):
    linear = nn.Linear(4, 5, bias=False)
    original = linear.weight.detach().clone()
    processor_all._decompose_linear_layer(linear)
    assert hasattr(linear, SVD_LOWRANK_L1_PARAM_NAME)
    assert hasattr(linear, SVD_LOWRANK_L2_PARAM_NAME)
    left = getattr(linear, SVD_LOWRANK_L1_PARAM_NAME)
    right = getattr(linear, SVD_LOWRANK_L2_PARAM_NAME)
    assert isinstance(left, nn.Parameter)
    assert isinstance(right, nn.Parameter)
    reconstructed = right.data @ left.data
    residual = linear.weight.data
    restored = reconstructed + residual
    assert torch.allclose(restored, original, atol=1e-4, rtol=1e-4)


def test__decompose_linear_layer_given_second_call_when_params_exist_then_update_parameter_data_only(processor_all):
    linear = nn.Linear(4, 5, bias=False)
    processor_all._decompose_linear_layer(linear)
    left1 = getattr(linear, SVD_LOWRANK_L1_PARAM_NAME)
    right1 = getattr(linear, SVD_LOWRANK_L2_PARAM_NAME)
    id_left1 = id(left1)
    id_right1 = id(right1)
    data_left1 = left1.data.clone()
    data_right1 = right1.data.clone()

    processor_all._decompose_linear_layer(linear)
    left2 = getattr(linear, SVD_LOWRANK_L1_PARAM_NAME)
    right2 = getattr(linear, SVD_LOWRANK_L2_PARAM_NAME)
    assert id(left2) == id_left1
    assert id(right2) == id_right1
    assert not torch.allclose(left2.data, data_left1)
    assert not torch.allclose(right2.data, data_right1)


def test__perform_svd_decomposition_given_valid_rank_when_called_then_shapes_and_reconstruction_are_reasonable(
    processor_all,
):
    weight = torch.randn(4, 6, dtype=torch.float32)
    processor_all.config.rank = 3
    left, right = processor_all._perform_svd_decomposition(weight)
    assert left.shape == (3, 6)
    assert right.shape == (4, 3)
    approx = right @ left
    assert approx.shape == weight.shape
    assert torch.norm(weight - approx) <= torch.norm(weight)


def test__perform_svd_decomposition_given_smaller_rank_when_called_then_higher_error_than_full_rank(processor_all):
    weight = torch.randn(4, 6, dtype=torch.float32)
    full_rank = min(weight.shape)
    processor_all.config.rank = full_rank
    left_full, right_full = processor_all._perform_svd_decomposition(weight)
    approx_full = right_full @ left_full
    err_full = torch.norm(weight - approx_full)

    processor_all.config.rank = 1
    try:
        left_low, right_low = processor_all._perform_svd_decomposition(weight)
    except RuntimeError:
        # CPU平台对 rank=1 的 svd_lowrank 有后端限制，跳过这条测试
        pytest.skip("svd_lowrank with rank=1 is not supported on this backend, skip error comparison test")

    approx_low = right_low @ left_low
    err_low = torch.norm(weight - approx_low)

    assert err_low >= err_full - 1e-5


def test_set_hook_ir_given_linear_with_svd_params_when_called_then_hook_registered(monkeypatch, processor_all):
    class FakeHookIR:
        def __init__(self, left, right):
            self.left = left
            self.right = right
            self.handle = None

        def set_hook_handle(self, handle):
            self.handle = handle

        def __call__(self, module, inputs):
            return inputs

    from msmodelslim.processor.svd_residual import processor as processor_mod

    monkeypatch.setattr(processor_mod.qir, "SVDResidualHookIR", FakeHookIR)

    model = nn.Module()
    linear = nn.Linear(3, 4, bias=False)
    setattr(linear, SVD_LOWRANK_L1_PARAM_NAME, nn.Parameter(torch.randn(2, 3)))
    setattr(linear, SVD_LOWRANK_L2_PARAM_NAME, nn.Parameter(torch.randn(4, 2)))
    model.add_module("proj", linear)

    processor_all.set_hook_ir(model)

    assert len(linear._forward_pre_hooks) == 1
    hook = list(linear._forward_pre_hooks.values())[0]
    assert isinstance(hook, FakeHookIR)
    assert hook.left is getattr(linear, SVD_LOWRANK_L1_PARAM_NAME)
    assert hook.right is getattr(linear, SVD_LOWRANK_L2_PARAM_NAME)
    assert hook.handle is not None


def test_set_hook_ir_given_linear_without_svd_params_when_called_then_hook_not_registered(monkeypatch, processor_all):
    class FakeHookIR:
        def __init__(self, left, right):
            self.created = True

        def set_hook_handle(self, handle):
            pass

        def __call__(self, module, inputs):
            return inputs

    from msmodelslim.processor.svd_residual import processor as processor_mod

    monkeypatch.setattr(processor_mod.qir, "SVDResidualHookIR", FakeHookIR)

    model = nn.Module()
    linear = nn.Linear(3, 4, bias=False)
    model.add_module("proj", linear)

    processor_all.set_hook_ir(model)
    assert len(linear._forward_pre_hooks) == 0


def test__register_hook_for_linear_given_svd_params_present_when_called_then_register_pre_hook(
    monkeypatch, processor_all
):
    class FakeHookIR:
        def __init__(self, left, right):
            self.left = left
            self.right = right
            self.handle = None

        def set_hook_handle(self, handle):
            self.handle = handle

        def __call__(self, module, inputs):
            return inputs

    from msmodelslim.processor.svd_residual import processor as processor_mod

    monkeypatch.setattr(processor_mod.qir, "SVDResidualHookIR", FakeHookIR)

    linear = nn.Linear(3, 4, bias=False)
    setattr(linear, SVD_LOWRANK_L1_PARAM_NAME, nn.Parameter(torch.randn(2, 3)))
    setattr(linear, SVD_LOWRANK_L2_PARAM_NAME, nn.Parameter(torch.randn(4, 2)))

    processor_all._register_hook_for_linear(linear)
    assert len(linear._forward_pre_hooks) == 1
    hook = list(linear._forward_pre_hooks.values())[0]
    assert isinstance(hook, FakeHookIR)
    assert hook.handle is not None


def test__register_hook_for_linear_given_no_svd_params_when_called_then_not_register_hook(monkeypatch, processor_all):
    class FakeHookIR:
        def __init__(self, left, right):
            raise AssertionError("HookIR should not be constructed when params are missing")

        def set_hook_handle(self, handle):
            pass

        def __call__(self, module, inputs):
            return inputs

    from msmodelslim.processor.svd_residual import processor as processor_mod

    monkeypatch.setattr(processor_mod.qir, "SVDResidualHookIR", FakeHookIR)

    linear = nn.Linear(3, 4, bias=False)
    processor_all._register_hook_for_linear(linear)
    assert len(linear._forward_pre_hooks) == 0


def test_SVDResidualProcessor_is_data_free_given_multiple_calls_when_invoked_repeatedly_then_consistently_true(
    simple_model,
):
    cfg = SVDResidualProcessorConfig(rank=1)
    proc = SVDResidualProcessor(model=simple_model, config=cfg, adapter=None)
    for _ in range(5):
        assert proc.is_data_free() is True


def test_decompose_given_no_matching_include_when_pattern_mismatch_then_no_linear_is_processed(simple_model):
    cfg = SVDResidualProcessorConfig(rank=1, include=["non_match_pattern"], exclude=[])
    proc = SVDResidualProcessor(model=simple_model, config=cfg, adapter=None)
    model_copy = SimpleModel()
    original_l1 = model_copy.linear1.weight.detach().clone()
    original_l2 = model_copy.linear2.weight.detach().clone()
    proc.decompose("root", model_copy)
    assert torch.allclose(model_copy.linear1.weight.data, original_l1)
    assert torch.allclose(model_copy.linear2.weight.data, original_l2)
    assert not hasattr(model_copy.linear1, SVD_LOWRANK_L1_PARAM_NAME)
    assert not hasattr(model_copy.linear2, SVD_LOWRANK_L1_PARAM_NAME)


def test_set_hook_ir_given_nested_modules_when_called_then_recursively_register_on_all_linear(
    monkeypatch, processor_all
):
    class FakeHookIR:
        instances: List["FakeHookIR"] = []

        def __init__(self, left, right):
            self.left = left
            self.right = right
            self.handle = None
            FakeHookIR.instances.append(self)

        def set_hook_handle(self, handle):
            self.handle = handle

        def __call__(self, module, inputs):
            return inputs

    from msmodelslim.processor.svd_residual import processor as processor_mod

    monkeypatch.setattr(processor_mod.qir, "SVDResidualHookIR", FakeHookIR)

    class Nested(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = SimpleModel()

    nested = Nested()
    for m in nested.modules():
        if isinstance(m, nn.Linear):
            setattr(m, SVD_LOWRANK_L1_PARAM_NAME, nn.Parameter(torch.randn(1, m.in_features)))
            setattr(m, SVD_LOWRANK_L2_PARAM_NAME, nn.Parameter(torch.randn(m.out_features, 1)))

    processor_all.set_hook_ir(nested)
    linear_count = sum(1 for m in nested.modules() if isinstance(m, nn.Linear))
    assert len(FakeHookIR.instances) == linear_count
    for m in nested.modules():
        if isinstance(m, nn.Linear):
            assert len(m._forward_pre_hooks) == 1
