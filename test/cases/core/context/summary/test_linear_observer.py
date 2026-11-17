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

import pytest
import torch
from torch import nn

from msmodelslim.core.context.summary import SummaryContext, LinearQuantStatisticsObserver, LinearStats
from msmodelslim.core.context.summary.observers.linear_observer import ir_bits, module_bits
from msmodelslim.core.quantization_schema import WrapperIR


def _make_observer():
    return LinearQuantStatisticsObserver()


def _make_layer():
    layer = nn.Module()
    attention = nn.Module()
    attention.add_module("q_proj", nn.Linear(16, 8, bias=False))
    attention.add_module("k_proj", nn.Linear(16, 4, bias=False))
    layer.add_module("self_attn", attention)
    layer.add_module("mlp", nn.Linear(16, 32, bias=False))
    return layer


def _record(context, observer, layer_name, layer):
    observer.record(context, layer_name, layer, is_before_quant=True)
    observer.record(context, layer_name, layer, is_before_quant=False)


def test_linear_observer_pairs_pre_and_post_by_module_path_when_quantizing_layer(summary_state, fake_linear_factory):
    context = SummaryContext(summary_state)
    observer = _make_observer()
    layer = _make_layer()

    observer.record(context, "model.layers.0", layer, is_before_quant=True)
    layer.self_attn.q_proj = fake_linear_factory("W8A8DynamicPerChannelFakeQuantLinear", 8, 16)
    observer.record(context, "model.layers.0", layer, is_before_quant=False)

    entries = summary_state["linear@rank_0"]["model.layers.0"]
    assert entries["self_attn.q_proj"] == {
        "numel": 16 * 8,
        "original_bits": 32,
        "quantized_bits": 8,
        "quantized": True,
    }
    assert entries["self_attn.k_proj"] == {
        "numel": 16 * 4,
        "original_bits": 32,
        "quantized_bits": 32,
        "quantized": False,
    }
    assert entries["mlp"] == {
        "numel": 16 * 32,
        "original_bits": 32,
        "quantized_bits": 32,
        "quantized": False,
    }


def test_linear_observer_takes_numel_from_pre_and_width_from_post_class_name_when_recording(
    summary_state, fake_linear_factory
):
    context = SummaryContext(summary_state)
    observer = _make_observer()
    layer = nn.Module()
    layer.add_module("proj", nn.Linear(8, 8, bias=False))

    observer.record(context, "layer0", layer, is_before_quant=True)
    layer.proj = fake_linear_factory("W4A8DynamicFakeQuantLinear", 8, 8, dtype=torch.int8)
    observer.record(context, "layer0", layer, is_before_quant=False)

    entry = summary_state["linear@rank_0"]["layer0"]["proj"]
    # 4bit 权重以 int8 存储；若读 post 的 element_size 会变成 8bit。
    assert entry == {"numel": 64, "original_bits": 32, "quantized_bits": 4, "quantized": True}


def test_linear_observer_writes_nothing_when_after_has_no_before(summary_state):
    context = SummaryContext(summary_state)
    observer = _make_observer()
    layer = _make_layer()

    observer.record(context, "model.layers.0", layer, is_before_quant=False)

    assert summary_state == {}


def test_linear_observer_keeps_layers_separate_when_visited_back_to_back(summary_state, fake_linear_factory):
    context = SummaryContext(summary_state)
    observer = _make_observer()
    first = nn.Module()
    first.add_module("proj", nn.Linear(4, 4, bias=False))
    second = nn.Module()
    second.add_module("proj", nn.Linear(4, 4, bias=False))

    observer.record(context, "layer0", first, is_before_quant=True)
    first.proj = fake_linear_factory("W8A8DynamicPerChannelFakeQuantLinear", 4, 4)
    observer.record(context, "layer0", first, is_before_quant=False)
    _record(context, observer, "layer1", second)

    linear = summary_state["linear@rank_0"]
    assert linear["layer0"]["proj"]["quantized_bits"] == 8
    assert linear["layer1"]["proj"]["quantized_bits"] == 32


def test_linear_observer_sees_through_wrapper_ir_when_snapshotting(summary_state, fake_linear_factory):
    context = SummaryContext(summary_state)
    observer = _make_observer()
    layer = nn.Module()
    layer.add_module("ffn", WrapperIR(nn.Linear(8, 4, bias=False)))

    observer.record(context, "layer0", layer, is_before_quant=True)
    layer.ffn = WrapperIR(fake_linear_factory("W8A8DynamicPerChannelFakeQuantLinear", 4, 8))
    observer.record(context, "layer0", layer, is_before_quant=False)

    assert summary_state["linear@rank_0"]["layer0"] == {
        "ffn": {"numel": 32, "original_bits": 32, "quantized_bits": 8, "quantized": True}
    }


def test_linear_observer_does_not_descend_atomic_ir_when_snapshotting(summary_state, fake_linear_factory):
    context = SummaryContext(summary_state)
    observer = _make_observer()
    layer = nn.Module()
    layer.add_module("proj", nn.Linear(8, 4, bias=False))

    observer.record(context, "layer0", layer, is_before_quant=True)
    quantized = fake_linear_factory("W8A8DynamicPerChannelFakeQuantLinear", 4, 8)
    quantized.add_module("scale", nn.Linear(2, 2, bias=False))
    layer.proj = quantized
    observer.record(context, "layer0", layer, is_before_quant=False)

    entries = summary_state["linear@rank_0"]["layer0"]
    assert list(entries) == ["proj"]


def test_linear_observer_aggregate_sums_logical_bytes_when_entries_are_recorded(summary_state, fake_linear_factory):
    context = SummaryContext(summary_state)
    observer = _make_observer()
    layer = _make_layer()

    observer.record(context, "layers_0", layer, is_before_quant=True)
    layer.self_attn.q_proj = fake_linear_factory("W8A8DynamicPerChannelFakeQuantLinear", 8, 16)
    observer.record(context, "layers_0", layer, is_before_quant=False)

    result = observer.aggregate(context)

    # q_proj 512B -> 128B；k_proj 256B、mlp 2048B 中性。
    assert isinstance(result, LinearStats)
    assert result.original_bytes == pytest.approx(512 + 256 + 2048)
    assert result.quantized_bytes == pytest.approx(128 + 256 + 2048)
    assert result.ratio == pytest.approx(2432 / 2816)
    # 只有 q_proj 实际替换为量化 IR，展示位宽集合不含中性模块。
    assert result.widths == (8,)


def test_linear_observer_aggregate_merges_rank_sections_and_dedups_shared_prefixes_when_ranks_share_a_prefix(
    summary_state, monkeypatch
):
    context = SummaryContext(summary_state)
    observer = _make_observer()
    rank_zero_layer = nn.Module()
    rank_zero_layer.add_module("shared", nn.Linear(4, 3, bias=False))
    rank_zero_layer.add_module("expert_0", nn.Linear(2, 2, bias=False))
    rank_one_layer = nn.Module()
    rank_one_layer.add_module("shared", nn.Linear(4, 3, bias=False))
    rank_one_layer.add_module("expert_1", nn.Linear(2, 2, bias=False))

    monkeypatch.setattr("msmodelslim.core.context.summary.summary_context.current_rank_key", lambda: "rank_0")
    _record(context, observer, "layers_0", rank_zero_layer)
    monkeypatch.setattr("msmodelslim.core.context.summary.summary_context.current_rank_key", lambda: "rank_1")
    _record(context, observer, "layers_0", rank_one_layer)

    result = observer.aggregate(context)

    # 每个 rank 独占顶层 key，整段回写不会覆盖对方分片。
    assert set(summary_state) == {"linear@rank_0", "linear@rank_1"}
    # shared 48B 只计一次，expert_0 与 expert_1 各 16B。
    assert result.original_bytes == pytest.approx(80.0)
    assert result.quantized_bytes == pytest.approx(80.0)
    assert result.widths == ()


def test_linear_observer_aggregate_returns_none_when_no_data(summary_state):
    assert _make_observer().aggregate(SummaryContext(summary_state)) is None
    assert LinearQuantStatisticsObserver().aggregate(SummaryContext(None)) is None


def test_linear_observer_format_reports_ratio_and_bytes_when_stats_exist():
    observer = LinearQuantStatisticsObserver()

    assert observer.format(LinearStats(original_bytes=1000, quantized_bytes=250, widths=(4, 8))) == (
        "[Linear] quantized to {4, 8}bit, compression ratio 25.00% (original 1000B -> quantized 250B)"
    )


def test_linear_observer_record_is_a_no_op_when_container_is_missing():
    context = SummaryContext(None)
    observer = LinearQuantStatisticsObserver()
    layer = _make_layer()

    _record(context, observer, "layer0", layer)


@pytest.mark.parametrize(
    "class_name,expected",
    [
        ("W4A8DynamicFakeQuantLinear", 4),
        ("W8A8DynamicPerChannelFakeQuantLinear", 8),
        ("W8A16StaticPerChannelFakeQuantLinear", 8),
        ("W16A16sLinear", 16),
        ("WFP8AFP8DynamicPerChannelFakeQuantLinear", 8),
        ("W4A4MXDynamicPerBlockFakeQuantLinear", 4),
        ("W8A8MXDynamicPerBlockFakeQuantLinear", 8),
        ("W4A4MXDualScaleFakeQuantLinear", 4),
        ("AutoFakeQuantLinear", None),
        ("WrapperIR", None),
        ("", None),
    ],
)
def test_linear_observer_ir_bits_reads_weight_width_from_class_name_when_name_matches(class_name, expected):
    assert ir_bits(class_name) == expected


def test_linear_observer_module_bits_falls_back_to_weight_element_size_when_class_name_has_no_width():
    assert module_bits(nn.Linear(4, 3, bias=False)) == torch.empty((), dtype=torch.float32).element_size() * 8


def test_linear_observer_module_bits_prefers_class_name_over_storage_dtype_when_both_are_available(fake_linear_factory):
    # 4bit 权重以 int8 存储，位宽必须来自类名而不是 element_size。
    module = fake_linear_factory("W4A8DynamicFakeQuantLinear", 4, 3)
    assert module_bits(module) == 4


def test_linear_observer_module_bits_returns_none_when_ir_has_no_weight():
    assert module_bits(nn.ReLU()) is None
