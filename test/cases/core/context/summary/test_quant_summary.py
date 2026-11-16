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

from types import SimpleNamespace

import json
import pytest
import torch
from torch import nn

from msmodelslim import ir as qir
from msmodelslim.core.context.summary import (
    FAQuantStatisticsObserver,
    KVQuantStatisticsObserver,
    LinearQuantStatisticsObserver,
    LinearStats,
    QuantSummary,
    SUMMARY_NAMESPACE,
    SummaryContext,
    TimeQuantStatisticsObserver,
    quant_summary,
)
from msmodelslim.core.context.summary.utils import load_source_weight_bits
from msmodelslim.ir.qal import QDType, QParam, QScheme, QScope


def _context(state):
    """构造 finish/record 所需的框架 context 形态：命名空间带 ``state``。"""
    return {SUMMARY_NAMESPACE: SimpleNamespace(state=state)}


class _RecordingLogger:
    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        self.messages.append(message % args if args else message)

    def warning(self, message, *args):
        self.messages.append(message % args if args else message)


def _make_layer():
    layer = nn.Module()
    attention = nn.Module()
    attention.add_module("q_proj", nn.Linear(16, 8, bias=False))
    attention.add_module("k_proj", nn.Linear(16, 4, bias=False))
    layer.add_module("self_attn", attention)
    layer.add_module("mlp", nn.Linear(16, 32, bias=False))
    return layer


def _attach_fa(layer, fa_q=QDType.INT8, fa_k=QDType.INT8, fa_v=QDType.INT8):
    """在层的 attention 上挂 FA3 量化后留下的分支 IR。"""
    for name, dtype in (("fa_q", fa_q), ("fa_k", fa_k), ("fa_v", fa_v)):
        scope = QScope.PER_BLOCK if dtype in (QDType.MXFP4, QDType.MXFP8) else QScope.PER_TOKEN
        layer.self_attn.add_module(
            name,
            qir.AutoFakeQuantActivation.create(QParam(scheme=QScheme(scope=scope, dtype=dtype, symmetric=True))),
        )
    return layer


def _attach_kv(layer, key=QDType.INT8, value=QDType.INT8, attention="self_attn"):
    """在层的 attention 上挂 dynamic_cache 部署的 K/V 量化器。"""
    target = getattr(layer, attention)
    for name, dtype in (("key_states_quantizer", key), ("value_states_quantizer", value)):
        target.add_module(
            name,
            qir.AutoFakeQuantDynamicCache.create(
                QParam(
                    scheme=QScheme(scope=QScope.PER_CHANNEL, dtype=dtype, symmetric=True),
                    ext={"scale": torch.ones(1), "offset": torch.zeros(1, dtype=torch.int32)},
                )
            ),
        )
    return layer


def _recorder_logger(monkeypatch):
    recorder = _RecordingLogger()
    monkeypatch.setattr("msmodelslim.core.context.summary.summary.get_logger", lambda: recorder)
    return recorder


def test_quant_summary_singleton_exposes_builtin_observers_when_created():
    assert isinstance(quant_summary, QuantSummary)
    assert isinstance(quant_summary.linear, LinearQuantStatisticsObserver)
    assert isinstance(quant_summary.fa, FAQuantStatisticsObserver)
    assert isinstance(quant_summary.kv, KVQuantStatisticsObserver)
    assert isinstance(quant_summary.time, TimeQuantStatisticsObserver)


def test_quant_summary_writes_state_by_dimension_and_rank_when_layers_are_recorded(summary_state, fake_linear_factory):
    summary = QuantSummary()
    context = SummaryContext(summary_state)
    layer = _make_layer()

    summary.linear.record(context, "model.layers.0", layer, is_before_quant=True)
    layer.self_attn.q_proj = fake_linear_factory("W8A8DynamicPerChannelFakeQuantLinear", 8, 16)
    summary.linear.record(context, "model.layers.0", layer, is_before_quant=False)
    _attach_fa(layer)
    summary.fa.record(context, "model.layers.0", layer)
    _attach_kv(layer)
    summary.kv.record(context, "model.layers.0", layer)
    summary.time.record(context, "model.layers.0", 1.5)

    assert set(summary_state) == {
        "time",
        "linear@rank_0",
        "fa@rank_0",
        "kv@rank_0",
    }
    assert list(summary_state["linear@rank_0"]["model.layers.0"]) == ["self_attn.q_proj", "self_attn.k_proj", "mlp"]
    assert summary_state["fa@rank_0"]["model.layers.0"]["self_attn"] == {"fa_q": 8, "fa_k": 8, "fa_v": 8}
    assert summary_state["kv@rank_0"]["model.layers.0"]["self_attn"] == {"key_states": 8, "value_states": 8}
    assert summary_state["time"]["model.layers.0"] == {"elapsed": 1.5, "avg": 1.5}


def test_quant_summary_aggregate_collects_every_builtin_dimension_when_all_dimensions_record(
    summary_state, fake_linear_factory
):
    summary = QuantSummary()
    context = SummaryContext(summary_state)
    layer = _make_layer()

    summary.linear.record(context, "layers_0", layer, is_before_quant=True)
    layer.self_attn.q_proj = fake_linear_factory("W8A8DynamicPerChannelFakeQuantLinear", 8, 16)
    summary.linear.record(context, "layers_0", layer, is_before_quant=False)
    _attach_fa(layer, fa_v=QDType.MXFP4)
    summary.fa.record(context, "layers_0", layer)

    summary.time.record(context, "layers_0", 1.0)
    summary.time.record(context, "layers_1", 3.0)
    _attach_kv(layer)
    summary.kv.record(context, "layers_0", layer)

    result = summary.finish(_context(summary_state))

    # q_proj 512B -> 128B；k_proj 256B、mlp 2048B 中性。
    assert isinstance(result["linear"], LinearStats)
    assert result["linear"].original_bytes == pytest.approx(2816.0)
    assert result["linear"].quantized_bytes == pytest.approx(2432.0)
    assert result["linear"].widths == (8,)
    assert result["time"].layer_count == 2
    assert result["time"].total_elapsed == pytest.approx(4.0)
    # FA 只统计有量化 IR 的层：(0.5 + 0.5 + 0.75) / 3 = 0.5833。
    assert result["fa"].saving == pytest.approx(0.5833333333333333)
    # KV 只统计部署了量化器的层：(0.5 + 0.5) / 2 = 0.5。
    assert result["kv"].saving == pytest.approx(0.5)


def test_quant_summary_omits_missing_dimensions_when_outputting(summary_state, monkeypatch):
    recorder = _recorder_logger(monkeypatch)

    summary = QuantSummary()
    result = summary.finish(_context(summary_state))

    assert not result
    assert not recorder.messages


def test_quant_summary_output_skips_missing_dimensions_and_prints_widths_when_some_dimensions_are_empty(
    summary_state, monkeypatch, fake_linear_factory
):
    recorder = _recorder_logger(monkeypatch)

    summary = QuantSummary()
    context = SummaryContext(summary_state)
    layer = nn.Module()
    layer.add_module("proj", nn.Linear(8, 8, bias=False))
    layer.add_module("self_attn", nn.Module())
    summary.linear.record(context, "layers_0", layer, is_before_quant=True)
    layer.proj = fake_linear_factory("W8A8DynamicPerChannelFakeQuantLinear", 8, 8)
    summary.linear.record(context, "layers_0", layer, is_before_quant=False)
    _attach_fa(layer)
    summary.fa.record(context, "layers_0", layer)
    summary.time.record(context, "layers_0", 1.0)

    summary.finish(_context(summary_state))

    assert recorder.messages == [
        "==================== QUANTIZATION SUMMARY ====================",
        "[Linear] quantized to 8bit, compression ratio 25.00% (original 256B -> quantized 64B)",
        "[FA] quantized to 8bit, compression ratio 50.00% (original activations assumed BF16)",
        "[Time] 1 layers total 1.00s, avg 1.00s/layer",
        "===========================================================",
    ]


def test_quant_summary_merges_rank_sections_without_double_counting_when_ranks_share_layers(summary_state, monkeypatch):
    summary = QuantSummary()
    context = SummaryContext(summary_state)

    rank_zero_layer = nn.Module()
    rank_zero_layer.add_module("shared", nn.Linear(4, 3, bias=False))
    rank_zero_layer.add_module("expert_0", nn.Linear(2, 2, bias=False))
    rank_one_layer = nn.Module()
    rank_one_layer.add_module("shared", nn.Linear(4, 3, bias=False))
    rank_one_layer.add_module("expert_1", nn.Linear(2, 2, bias=False))

    monkeypatch.setattr("msmodelslim.core.context.summary.summary_context.current_rank_key", lambda: "rank_0")
    summary.linear.record(context, "layers_0", rank_zero_layer, is_before_quant=True)
    summary.linear.record(context, "layers_0", rank_zero_layer, is_before_quant=False)
    monkeypatch.setattr("msmodelslim.core.context.summary.summary_context.current_rank_key", lambda: "rank_1")
    summary.linear.record(context, "layers_0", rank_one_layer, is_before_quant=True)
    summary.linear.record(context, "layers_0", rank_one_layer, is_before_quant=False)

    result = summary.finish(_context(summary_state))

    # shared (48B) 只计一次，expert_0 (16B) 与 expert_1 (16B) 分别计入。
    assert result["linear"].original_bytes == pytest.approx(80.0)
    assert result["linear"].quantized_bytes == pytest.approx(80.0)


def test_quant_summary_fa_kv_saving_is_independent_of_time_when_time_is_missing(summary_state):
    summary = QuantSummary()
    context = SummaryContext(summary_state)

    layer = _attach_fa(_make_layer())
    _attach_kv(layer)
    summary.fa.record(context, "layers_0", layer)
    summary.kv.record(context, "layers_0", layer)

    result = summary.finish(_context(summary_state))

    assert "time" not in result
    assert result["fa"].saving == pytest.approx(0.5)
    assert result["kv"].saving == pytest.approx(0.5)


def test_quant_summary_fa_counts_only_layers_with_fa_ir_when_some_layers_are_unquantized(summary_state):
    summary = QuantSummary()
    context = SummaryContext(summary_state)

    layer = _attach_fa(_make_layer(), fa_q=QDType.MXFP4, fa_k=QDType.MXFP4, fa_v=QDType.MXFP4)
    summary.fa.record(context, "layers_0", layer)

    result = summary.finish(_context(summary_state))

    # 没有 FA IR 的层不进入分母，压缩率不被稀释。
    assert result["fa"].saving == pytest.approx(0.75)


def test_quant_summary_finish_returns_empty_result_when_context_is_missing():
    assert not QuantSummary().finish({})


def test_load_source_weight_bits_keeps_only_fp8_weights_when_index_has_scale_inv(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layers.0.proj.weight": "model.safetensors",
                    "layers.0.proj.weight_scale_inv": "model.safetensors",
                    "layers.0.norm.weight": "model.safetensors",
                }
            }
        )
    )

    assert load_source_weight_bits(str(tmp_path)) == {"layers.0.proj": 8}


def test_quant_summary_uses_source_bits_when_checkpoint_weight_is_fp8(summary_state, monkeypatch, fake_linear_factory):
    recorder = _recorder_logger(monkeypatch)

    summary = QuantSummary()
    context = SummaryContext(summary_state)
    layer = nn.Module()
    layer.add_module("proj", nn.Linear(8, 8, bias=False))
    layer.add_module("head", nn.Linear(4, 4, bias=False))
    layer.to(torch.bfloat16)
    # proj 的源权重是 fp8；head 不在映射里，保持内存位宽。
    source_bits = {"layers_0.proj": 8}

    summary.linear.record(context, "layers_0", layer, is_before_quant=True, source_bits=source_bits)
    layer.proj = fake_linear_factory("W8A8DynamicPerChannelFakeQuantLinear", 8, 8)
    summary.linear.record(context, "layers_0", layer, is_before_quant=False)

    result = summary.finish(_context(summary_state))

    # proj 按源 fp8 计 64B；head 保持 bf16 中性计 32B。
    assert result["linear"].original_bytes == pytest.approx(96.0)
    assert result["linear"].quantized_bytes == pytest.approx(96.0)
    assert (
        recorder.messages[1] == "[Linear] quantized to 8bit, compression ratio 100.00% (original 96B -> quantized 96B)"
    )
