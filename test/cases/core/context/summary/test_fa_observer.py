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
from torch import nn

import msmodelslim.core.quantization_schema as qir
from msmodelslim.core.context.summary import SummaryContext, FAQuantStatisticsObserver, FAStats
from msmodelslim.core.quantization_schema.qal import QDType, QParam, QScheme, QScope


def _fa_ir(dtype):
    scope = QScope.PER_BLOCK if dtype in (QDType.MXFP4, QDType.MXFP8) else QScope.PER_TOKEN
    return qir.AutoFakeQuantActivation.create(QParam(scheme=QScheme(scope=scope, dtype=dtype, symmetric=True)))


def _attention(**branches):
    attention = nn.Module()
    for name, dtype in branches.items():
        attention.add_module(name, _fa_ir(dtype))
    return attention


def _layer(**attentions):
    layer = nn.Module()
    for name, attention in attentions.items():
        layer.add_module(name, attention)
    return layer


def test_fa_observer_reads_branch_bits_from_quantized_ir_when_recording(summary_state):
    context = SummaryContext(summary_state)
    observer = FAQuantStatisticsObserver()
    layer = _layer(self_attn=_attention(fa_q=QDType.INT8, fa_k=QDType.FP8_E4M3, fa_v=QDType.MXFP4))

    observer.record(context, "model.layers.0", layer)

    assert summary_state["fa@rank_0"]["model.layers.0"] == {"self_attn": {"fa_q": 8, "fa_k": 8, "fa_v": 4}}


def test_fa_observer_supports_fa3_prefixed_branches_when_recording(summary_state):
    context = SummaryContext(summary_state)
    observer = FAQuantStatisticsObserver()
    layer = _layer(self_attn=_attention(fa3_q=QDType.INT8, fa3_k=QDType.FP8_E4M3))

    observer.record(context, "model.layers.0", layer)

    assert summary_state["fa@rank_0"]["model.layers.0"] == {"self_attn": {"fa_q": 8, "fa_k": 8}}


def test_fa_observer_does_not_write_unconfigured_branches_when_recording(summary_state):
    context = SummaryContext(summary_state)
    observer = FAQuantStatisticsObserver()
    layer = _layer(self_attn=_attention(fa_q=QDType.INT8))

    observer.record(context, "model.layers.0", layer)

    assert summary_state["fa@rank_0"]["model.layers.0"] == {"self_attn": {"fa_q": 8}}


def test_fa_observer_ignores_non_fa_activation_ir_when_scanning(summary_state):
    context = SummaryContext(summary_state)
    observer = FAQuantStatisticsObserver()
    attention = _attention(fa_q=QDType.INT8)
    attention.add_module("q_proj_input", _fa_ir(QDType.INT8))
    layer = _layer(self_attn=attention)

    observer.record(context, "model.layers.0", layer)

    assert summary_state["fa@rank_0"]["model.layers.0"] == {"self_attn": {"fa_q": 8}}


def test_fa_observer_keeps_layers_separate_when_recording_consecutive_layers(summary_state):
    context = SummaryContext(summary_state)
    observer = FAQuantStatisticsObserver()
    first = _layer(self_attn=_attention(fa_q=QDType.INT8, fa_k=QDType.INT8, fa_v=QDType.INT8))
    second = _layer(self_attn=_attention(fa_q=QDType.MXFP4, fa_k=QDType.MXFP4, fa_v=QDType.MXFP4))

    observer.record(context, "layer0", first)
    observer.record(context, "layer1", second)

    fa_state = summary_state["fa@rank_0"]
    assert fa_state["layer0"]["self_attn"] == {"fa_q": 8, "fa_k": 8, "fa_v": 8}
    assert fa_state["layer1"]["self_attn"] == {"fa_q": 4, "fa_k": 4, "fa_v": 4}


def test_fa_observer_writes_nothing_when_no_fa_ir_exists(summary_state):
    context = SummaryContext(summary_state)
    observer = FAQuantStatisticsObserver()

    observer.record(context, "layer0", _layer(self_attn=nn.Module()))

    assert summary_state == {}


def test_fa_observer_skips_recording_when_layer_name_is_empty(summary_state):
    context = SummaryContext(summary_state)
    observer = FAQuantStatisticsObserver()
    layer = _layer(self_attn=_attention(fa_q=QDType.INT8))

    observer.record(context, "", layer)

    assert summary_state == {}


def test_fa_observer_counts_missing_branches_as_bf16_when_aggregating(summary_state):
    context = SummaryContext(summary_state)
    observer = FAQuantStatisticsObserver()

    observer.record(context, "layer0", _layer(self_attn=_attention(fa_q=QDType.INT8)))

    result = observer.aggregate(context)
    # 只有 q 量化：0.5 / 3；k/v 按 16bit 中性计入，但不进入展示位宽集合。
    assert result.saving == pytest.approx(0.5 / 3)
    assert result.widths == (8,)


def test_fa_observer_averages_attentions_then_layers_when_aggregating(summary_state):
    context = SummaryContext(summary_state)
    observer = FAQuantStatisticsObserver()
    observer.record(
        context,
        "layer0",
        _layer(
            self_attn=_attention(fa_q=QDType.INT8, fa_k=QDType.INT8, fa_v=QDType.INT8),
            cross_attn=_attention(fa_q=QDType.MXFP4, fa_k=QDType.MXFP4, fa_v=QDType.MXFP4),
        ),
    )
    observer.record(
        context, "layer1", _layer(self_attn=_attention(fa_q=QDType.INT8, fa_k=QDType.INT8, fa_v=QDType.INT8))
    )

    # 层内：(0.5 + 0.75) / 2 = 0.625；两层平均：(0.625 + 0.5) / 2。
    assert observer.aggregate(context).saving == pytest.approx(0.5625)


def test_fa_observer_unions_rank_sections_when_aggregating(summary_state, monkeypatch):
    context = SummaryContext(summary_state)
    observer = FAQuantStatisticsObserver()
    monkeypatch.setattr("msmodelslim.core.context.summary.summary_context.current_rank_key", lambda: "rank_0")
    observer.record(
        context, "layer0", _layer(self_attn=_attention(fa_q=QDType.INT8, fa_k=QDType.INT8, fa_v=QDType.INT8))
    )
    monkeypatch.setattr("msmodelslim.core.context.summary.summary_context.current_rank_key", lambda: "rank_1")
    observer.record(
        context, "layer0", _layer(cross_attn=_attention(fa_q=QDType.MXFP4, fa_k=QDType.MXFP4, fa_v=QDType.MXFP4))
    )

    # 每个 rank 独占顶层 key；同一层的两个分片按 attention 合并：(0.5 + 0.75) / 2。
    assert set(summary_state) == {"fa@rank_0", "fa@rank_1"}
    assert observer.aggregate(context).saving == pytest.approx(0.625)


def test_fa_observer_aggregate_returns_none_when_no_data(summary_state):
    assert FAQuantStatisticsObserver().aggregate(SummaryContext(summary_state)) is None
    assert FAQuantStatisticsObserver().aggregate(SummaryContext(None)) is None


def test_fa_observer_format_reports_widths_and_ratio_when_stats_exist():
    observer = FAQuantStatisticsObserver()

    assert observer.format(FAStats(saving=0.5, widths=(8,))) == (
        "[FA] quantized to 8bit, compression ratio 50.00% (original activations assumed BF16)"
    )
    assert observer.format(FAStats(saving=0.5, widths=(4, 8))) == (
        "[FA] quantized to {4, 8}bit, compression ratio 50.00% (original activations assumed BF16)"
    )


def test_fa_observer_record_is_a_no_op_when_container_is_missing():
    context = SummaryContext(None)
    observer = FAQuantStatisticsObserver()

    observer.record(context, "layer0", _layer(self_attn=_attention(fa_q=QDType.INT8)))
