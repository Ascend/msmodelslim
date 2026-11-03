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

import copy
import sys
import types

import pytest
import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.ir.quarot import (
    OnlineRotationInfo,
    OnlineRotationInputHookIR,
    OnlineRotationOutputHookIR,
    RotationType,
)
from msmodelslim.model.wan2_2.base_model_adapter import Wan2_2BaseModelAdapter
from msmodelslim.processor.lossless.metrics.base import MetricContext
from msmodelslim.processor.lossless.metrics.builtins import (
    NumericalEquivalenceMetric,
    NumericalEquivalenceMetricConfig,
)
from msmodelslim.processor.quarot.common.quarot_utils import create_rot
from msmodelslim.processor.quarot.online_quarot.online_quarot import (
    OnlineQuaRotProcessor,
    OnlineQuaRotProcessorConfig,
)


def _attention_backend(q, k, v, **_kwargs):
    return torch.zeros_like(q)


def _install_legacy_wan_modules(monkeypatch, events=None, rope_transform=None, attention_impl=None):
    wan_package = types.ModuleType("wan")
    wan_package.__path__ = []
    modules_package = types.ModuleType("wan.modules")
    modules_package.__path__ = []
    model_module = types.ModuleType("wan.modules.model")

    def rope_apply(tensor, _grid_sizes, _freqs):
        if events is not None:
            events.append("rope")
        return tensor if rope_transform is None else rope_transform(tensor)

    def attention(q, k, v, **kwargs):
        if events is not None:
            events.append("attention")
        if attention_impl is not None:
            return attention_impl(q, k, v, **kwargs)
        return _attention_backend(q, k, v, **kwargs)

    model_module.rope_apply = rope_apply
    attention_module = types.ModuleType("wan.modules.attention")
    attention_module.attention = attention
    attention_module.flash_attention = attention

    monkeypatch.setitem(sys.modules, "wan", wan_package)
    monkeypatch.setitem(sys.modules, "wan.modules", modules_package)
    monkeypatch.setitem(sys.modules, "wan.modules.model", model_module)
    monkeypatch.setitem(sys.modules, "wan.modules.attention", attention_module)


class _TraceIdentity(nn.Module):
    def __init__(self, events, name):
        super().__init__()
        self.events = events
        self.name = name

    def forward(self, value):
        self.events.append(self.name)
        return value


class _QuantizeToHalfStep(nn.Module):
    def forward(self, value):
        return torch.round(value * 2) / 2


class WanSelfAttention(nn.Module):
    def __init__(self, dim=4, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = (-1, -1)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.norm_q = nn.Identity()
        self.norm_k = nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        raise AssertionError("The adapter must install its legacy forward wrapper")


class WanCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens):  # pylint: disable=arguments-differ
        raise AssertionError("The adapter must install its legacy forward wrapper")


WanSelfAttention.__module__ = "wan.modules.model"
WanSelfAttention.forward.__module__ = "wan.modules.model"
WanCrossAttention.__module__ = "wan.modules.model"
WanCrossAttention.forward.__module__ = "wan.modules.model"


def test_inject_fa3_placeholders_resolves_legacy_self_and_cross_attention(monkeypatch):
    _install_legacy_wan_modules(monkeypatch)
    model = nn.Module()
    model.self_attn = WanSelfAttention()
    model.cross_attn = WanCrossAttention()
    adapter = object.__new__(Wan2_2BaseModelAdapter)

    adapter.inject_fa3_placeholders("", model, lambda _name: True)

    self_output = model.self_attn(
        torch.randn(1, 2, 4),
        torch.tensor([2]),
        torch.tensor([[1, 1, 2]]),
        torch.ones(2, 2, dtype=torch.complex64),
    )
    cross_output = model.cross_attn(
        torch.randn(1, 2, 4),
        torch.randn(1, 3, 4),
        torch.tensor([3]),
    )

    assert self_output.shape == (1, 2, 4)
    assert cross_output.shape == (1, 2, 4)
    assert isinstance(model.self_attn.fa3_q, nn.Module)
    assert isinstance(model.cross_attn.fa3_k, nn.Module)


def test_fa3_only_self_attention_keeps_legacy_placeholder_before_rope_order(monkeypatch):
    events = []
    _install_legacy_wan_modules(monkeypatch, events)
    model = nn.Module()
    model.self_attn = WanSelfAttention()
    adapter = object.__new__(Wan2_2BaseModelAdapter)

    adapter.inject_fa3_placeholders("", model, lambda _name: True)
    for name in ("fa3_q", "fa3_k", "fa3_v"):
        model.self_attn.set_submodule(name, _TraceIdentity(events, name))

    model.self_attn(
        torch.randn(1, 2, 4),
        torch.tensor([2]),
        torch.tensor([[1, 1, 2]]),
        torch.ones(2, 2, dtype=torch.complex64),
    )

    assert events == ["fa3_q", "fa3_k", "fa3_v", "rope", "rope", "attention"]


def test_online_quarot_all_excluded_keeps_fa3_only_order(monkeypatch):
    events = []
    _install_legacy_wan_modules(monkeypatch, events)
    model = nn.Module()
    model.self_attn = WanSelfAttention()
    adapter = object.__new__(Wan2_2BaseModelAdapter)
    processor = OnlineQuaRotProcessor(
        model,
        OnlineQuaRotProcessorConfig(exclude=["*"]),
        adapter,
    )
    processor.pre_run()
    processor.preprocess(BatchProcessRequest(name="", module=model, datas=None, outputs=None))
    adapter.inject_fa3_placeholders("", model, lambda _name: True)
    for name in ("fa3_q", "fa3_k", "fa3_v"):
        model.self_attn.set_submodule(name, _TraceIdentity(events, name))

    assert not processor.rotation_infos
    assert isinstance(model.self_attn.q_rot, nn.Identity)
    assert isinstance(model.self_attn.k_rot, nn.Identity)
    model.self_attn(
        torch.randn(1, 2, 4),
        torch.tensor([2]),
        torch.tensor([[1, 1, 2]]),
        torch.ones(2, 2, dtype=torch.complex64),
    )

    assert events == ["fa3_q", "fa3_k", "fa3_v", "rope", "rope", "attention"]


@pytest.mark.parametrize("online_before_fa3", [True, False])
def test_active_online_rotation_keeps_rope_before_fa3_for_both_injection_orders(monkeypatch, online_before_fa3):
    events = []
    _install_legacy_wan_modules(monkeypatch, events)
    model = nn.Module()
    model.self_attn = WanSelfAttention()
    adapter = object.__new__(Wan2_2BaseModelAdapter)

    def install_online_rotation():
        processor = OnlineQuaRotProcessor(model, OnlineQuaRotProcessorConfig(), adapter)
        processor.pre_run()
        processor.preprocess(BatchProcessRequest(name="", module=model, datas=None, outputs=None))
        for name in ("q_rot", "k_rot"):
            model.self_attn.get_submodule(name).register_forward_hook(
                lambda _module, _args, _output, name=name: events.append(name)
            )
        return processor

    processor = None
    if online_before_fa3:
        processor = install_online_rotation()
    adapter.inject_fa3_placeholders("", model, lambda _name: True)
    for name in ("fa3_q", "fa3_k", "fa3_v"):
        model.self_attn.set_submodule(name, _TraceIdentity(events, name))
    if not online_before_fa3:
        processor = install_online_rotation()

    assert processor is not None
    assert processor.rotation_infos
    assert isinstance(model.self_attn.q_rot, nn.Module)
    assert isinstance(model.self_attn.k_rot, nn.Module)
    model.self_attn(
        torch.randn(1, 2, 4),
        torch.tensor([2]),
        torch.tensor([[1, 1, 2]]),
        torch.ones(2, 2, dtype=torch.complex64),
    )

    assert events == ["rope", "rope", "q_rot", "k_rot", "fa3_q", "fa3_k", "fa3_v", "attention"]


@pytest.mark.parametrize(
    ("rotation_type", "hook_type"),
    [
        (RotationType.INPUT, OnlineRotationInputHookIR),
        (RotationType.OUTPUT, OnlineRotationOutputHookIR),
    ],
)
def test_online_rotation_hook_on_identity_keeps_rope_before_fa3(monkeypatch, rotation_type, hook_type):
    events = []
    _install_legacy_wan_modules(monkeypatch, events)
    model = nn.Module()
    model.self_attn = WanSelfAttention()
    matrix = torch.tensor(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    for name in ("q_rot", "k_rot"):
        module = nn.Identity()
        full_name = f"self_attn.{name}"
        info = OnlineRotationInfo(matrix.clone(), rotation_type, full_name)
        hook = hook_type(full_name, info)
        handle = (
            module.register_forward_pre_hook(hook)
            if rotation_type == RotationType.INPUT
            else module.register_forward_hook(hook)
        )
        hook.set_hook_handle(handle)
        model.self_attn.set_submodule(name, module)
    adapter = object.__new__(Wan2_2BaseModelAdapter)

    adapter.inject_fa3_placeholders("", model, lambda _name: True)
    for name in ("fa3_q", "fa3_k", "fa3_v"):
        model.self_attn.set_submodule(name, _TraceIdentity(events, name))
    model.self_attn(
        torch.randn(1, 2, 4),
        torch.tensor([2]),
        torch.tensor([[1, 1, 2]]),
        torch.ones(2, 2, dtype=torch.complex64),
    )

    assert events == ["rope", "rope", "fa3_q", "fa3_k", "fa3_v", "attention"]


def test_online_observer_hooks_on_identity_do_not_change_fa3_order(monkeypatch):
    events = []
    _install_legacy_wan_modules(monkeypatch, events)
    model = nn.Module()
    model.self_attn = WanSelfAttention()
    for name in ("q_rot", "k_rot"):
        module = nn.Identity()
        module.register_forward_hook(lambda _module, _args, _output, name=name: events.append(f"observe_{name}"))
        model.self_attn.set_submodule(name, module)
    adapter = object.__new__(Wan2_2BaseModelAdapter)

    adapter.inject_fa3_placeholders("", model, lambda _name: True)
    for name in ("fa3_q", "fa3_k", "fa3_v"):
        model.self_attn.set_submodule(name, _TraceIdentity(events, name))
    model.self_attn(
        torch.randn(1, 2, 4),
        torch.tensor([2]),
        torch.tensor([[1, 1, 2]]),
        torch.ones(2, 2, dtype=torch.complex64),
    )

    assert events == [
        "observe_q_rot",
        "observe_k_rot",
        "fa3_q",
        "fa3_k",
        "fa3_v",
        "rope",
        "rope",
        "attention",
    ]
    assert events.count("observe_q_rot") == 1
    assert events.count("observe_k_rot") == 1


def test_pending_lossless_rotation_preserves_reference_order_and_cleans_up(monkeypatch):
    events = []
    _install_legacy_wan_modules(monkeypatch, events)
    model = nn.Module()
    model.self_attn = WanSelfAttention()
    adapter = object.__new__(Wan2_2BaseModelAdapter)
    first = OnlineQuaRotProcessor(model, OnlineQuaRotProcessorConfig(), adapter)
    second = OnlineQuaRotProcessor(model, OnlineQuaRotProcessorConfig(), adapter)
    adapter.inject_fa3_placeholders("", model, lambda _name: True)
    for name in ("fa3_q", "fa3_k", "fa3_v"):
        model.self_attn.set_submodule(name, _TraceIdentity(events, name))
    inputs = (
        torch.randn(1, 2, 4),
        torch.tensor([2]),
        torch.tensor([[1, 1, 2]]),
        torch.ones(2, 2, dtype=torch.complex64),
    )

    first.prepare_lossless_check()
    second.prepare_lossless_check()
    first.abort_lossless_check()
    model.self_attn(*inputs)
    assert events == ["rope", "rope", "fa3_q", "fa3_k", "fa3_v", "attention"]

    events.clear()
    second.post_run()
    model.self_attn(*inputs)
    assert events == ["fa3_q", "fa3_k", "fa3_v", "rope", "rope", "attention"]
    assert not any(hasattr(module, "_lossless_online_rotation_pending") for module in model.modules())


def test_partially_excluded_attention_keeps_rope_before_fa3_only_for_active_sites(monkeypatch):
    events = []
    _install_legacy_wan_modules(monkeypatch, events)
    model = nn.Module()
    model.active_attn = WanSelfAttention()
    model.excluded_attn = WanSelfAttention()
    adapter = object.__new__(Wan2_2BaseModelAdapter)
    processor = OnlineQuaRotProcessor(
        model,
        OnlineQuaRotProcessorConfig(exclude=["excluded_attn.q_rot", "excluded_attn.k_rot"]),
        adapter,
    )
    processor.pre_run()
    processor.preprocess(BatchProcessRequest(name="", module=model, datas=None, outputs=None))
    adapter.inject_fa3_placeholders("", model, lambda _name: True)
    for attention_name in ("active_attn", "excluded_attn"):
        attention = getattr(model, attention_name)
        for name in ("fa3_q", "fa3_k", "fa3_v"):
            attention.set_submodule(name, _TraceIdentity(events, f"{attention_name}.{name}"))
    for name in ("q_rot", "k_rot"):
        getattr(model.active_attn, name).register_forward_hook(
            lambda _module, _args, _output, name=name: events.append(f"active_attn.{name}")
        )

    model.active_attn(
        torch.randn(1, 2, 4),
        torch.tensor([2]),
        torch.tensor([[1, 1, 2]]),
        torch.ones(2, 2, dtype=torch.complex64),
    )
    assert events == [
        "rope",
        "rope",
        "active_attn.q_rot",
        "active_attn.k_rot",
        "active_attn.fa3_q",
        "active_attn.fa3_k",
        "active_attn.fa3_v",
        "attention",
    ]
    events.clear()
    model.excluded_attn(
        torch.randn(1, 2, 4),
        torch.tensor([2]),
        torch.tensor([[1, 1, 2]]),
        torch.ones(2, 2, dtype=torch.complex64),
    )
    assert events == [
        "excluded_attn.fa3_q",
        "excluded_attn.fa3_k",
        "excluded_attn.fa3_v",
        "rope",
        "rope",
        "attention",
    ]


def test_all_excluded_online_rotation_matches_fa3_only_numerically(monkeypatch):
    def rope_transform(tensor):
        return torch.stack(
            (
                0.8 * tensor[..., 0] - 0.6 * tensor[..., 1],
                0.6 * tensor[..., 0] + 0.8 * tensor[..., 1],
                0.6 * tensor[..., 2] - 0.8 * tensor[..., 3],
                0.8 * tensor[..., 2] + 0.6 * tensor[..., 3],
            ),
            dim=-1,
        )

    def attention_impl(q, k, v, **_kwargs):
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        weights = torch.softmax(q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5, dim=-1)
        return (weights @ v).transpose(1, 2)

    torch.manual_seed(11)
    baseline_model = nn.Module()
    baseline_model.self_attn = WanSelfAttention()
    excluded_model = copy.deepcopy(baseline_model)
    inputs = (
        torch.randn(1, 2, 4),
        torch.tensor([2]),
        torch.tensor([[1, 1, 2]]),
        torch.ones(2, 2, dtype=torch.complex64),
    )
    adapter = object.__new__(Wan2_2BaseModelAdapter)

    _install_legacy_wan_modules(monkeypatch, rope_transform=rope_transform, attention_impl=attention_impl)
    adapter.inject_fa3_placeholders("", baseline_model, lambda _name: True)
    for name in ("fa3_q", "fa3_k", "fa3_v"):
        baseline_model.self_attn.set_submodule(name, _QuantizeToHalfStep())
    baseline_output = baseline_model.self_attn(*inputs)

    processor = OnlineQuaRotProcessor(
        excluded_model,
        OnlineQuaRotProcessorConfig(exclude=["*"]),
        adapter,
    )
    processor.pre_run()
    processor.preprocess(BatchProcessRequest(name="", module=excluded_model, datas=None, outputs=None))
    adapter.inject_fa3_placeholders("", excluded_model, lambda _name: True)
    for name in ("fa3_q", "fa3_k", "fa3_v"):
        excluded_model.self_attn.set_submodule(name, _QuantizeToHalfStep())
    excluded_output = excluded_model.self_attn(*inputs)

    assert not processor.rotation_infos
    assert torch.equal(excluded_output, baseline_output)


@pytest.mark.parametrize("method_attention", [False, True])
def test_self_attention_rotates_after_rope_before_fa3(monkeypatch, method_attention):
    events = []
    _install_legacy_wan_modules(monkeypatch, events)
    model = nn.Module()
    model.self_attn = WanSelfAttention()
    model.self_attn.q_rot = _TraceIdentity(events, "q_rot")
    model.self_attn.k_rot = _TraceIdentity(events, "k_rot")
    if method_attention:

        def attention(q, k, v, **_kwargs):
            events.append("attention")
            return _attention_backend(q, k, v)

        model.self_attn.attention = attention
    adapter = object.__new__(Wan2_2BaseModelAdapter)

    adapter.inject_fa3_placeholders("", model, lambda _name: True)
    for name in ("fa3_q", "fa3_k", "fa3_v"):
        model.self_attn.set_submodule(name, _TraceIdentity(events, name))

    model.self_attn(
        torch.randn(1, 2, 4),
        torch.tensor([2]),
        torch.tensor([[1, 1, 2]]),
        torch.ones(2, 2, dtype=torch.complex64),
    )

    assert events == [
        "rope",
        "rope",
        "q_rot",
        "k_rot",
        "fa3_q",
        "fa3_k",
        "fa3_v",
        "attention",
    ]


def test_head_dim_128_fp32_online_rotation_preserves_numerical_equivalence():
    torch.manual_seed(7)
    model = nn.Module()
    model.self_attn = WanSelfAttention(dim=128, num_heads=1)
    adapter = object.__new__(Wan2_2BaseModelAdapter)

    rotation_config = adapter.get_online_rotation_configs(model)["self_attn.q_rot"]
    rotation = create_rot(
        mode=rotation_config.rotation_mode,
        size=rotation_config.rotation_size,
        block_size=rotation_config.block_size,
        rot_step=rotation_config.rot_step,
        eye_step=rotation_config.eye_step,
        seed=rotation_config.seed,
        dtype=rotation_config.dtype,
    )

    reference = torch.randn(2, 128, dtype=model.self_attn.q.weight.dtype)
    applied_rotation = rotation.to(dtype=reference.dtype)
    rotated = reference @ applied_rotation.T
    aligned = rotated @ applied_rotation
    result = NumericalEquivalenceMetric(NumericalEquivalenceMetricConfig()).evaluate(
        reference,
        aligned,
        MetricContext(
            processor="Wan2.2 online QuaRot",
            decoder="HADAMARD rotation round trip",
            structure="self_attn.q_rot",
            structure_type="OnlineRotationWrapper",
            side="output",
            invocation=0,
            path="head_dim=128, parameter dtype=float32",
        ),
    )

    assert result.passed, result.observations
    assert rotation_config.dtype is torch.float32


def test_online_rotation_uses_bfloat16_attention_parameter_dtype():
    model = nn.Module()
    model.self_attn = WanSelfAttention(dim=128, num_heads=1).to(dtype=torch.bfloat16)
    adapter = object.__new__(Wan2_2BaseModelAdapter)

    configs = adapter.get_online_rotation_configs(model)

    assert configs["self_attn.q_rot"].dtype is torch.bfloat16
    assert configs["self_attn.k_rot"].dtype is torch.bfloat16


def test_online_rotation_configs_without_model_remain_empty():
    adapter = object.__new__(Wan2_2BaseModelAdapter)

    assert not adapter.get_online_rotation_configs()
