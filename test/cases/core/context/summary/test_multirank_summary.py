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

import sys

import pytest
import torch
import torch.distributed as dist
from torch import nn

import msmodelslim.core.quantization_schema as qir
from msmodelslim.core.context.shared_dict_context.context import SharedDictContext
from msmodelslim.core.context.summary import SUMMARY_NAMESPACE, SummaryContext, quant_summary
from msmodelslim.core.quantization_schema.qal import QDType, QParam, QScheme, QScope
from test.cases.utils.distributed.dts_distributed_spawn import run_distributed_spawn, spawn_queue

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="SharedDictContext multi-process tests require Linux (AF_UNIX, SO_PEERCRED)",
)

_LAYER_NAME = "model.layers.0"


class _FakeQuantLinear(nn.Module):
    """类名即量化 IR 名，供 Linear 观察者从类名解析位宽；仅在子进程内构造。"""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, 4, dtype=torch.int8), requires_grad=False)


_FakeQuantLinear.__name__ = "W8A8DynamicPerChannelFakeQuantLinear"


def _fa_ir():
    return qir.AutoFakeQuantActivation.create(
        QParam(scheme=QScheme(scope=QScope.PER_TOKEN, dtype=QDType.INT8, symmetric=True))
    )


def _kv_ir():
    return qir.AutoFakeQuantDynamicCache.create(
        QParam(
            scheme=QScheme(scope=QScope.PER_CHANNEL, dtype=QDType.INT8, symmetric=True),
            ext={"scale": torch.ones(1), "offset": torch.zeros(1, dtype=torch.int32)},
        )
    )


def _build_layer(rank):
    """每个 rank 提供同一层中不同前缀的线性层，模拟 MoE/专家并行下的分片写入。"""
    attention = nn.Module()
    for branch in ("fa_q", "fa_k", "fa_v"):
        attention.add_module(branch, _fa_ir())
    attention.add_module("key_states_quantizer", _kv_ir())
    attention.add_module("value_states_quantizer", _kv_ir())

    mlp = nn.Module()
    mlp.add_module("rank%d_proj" % rank, nn.Linear(4, 4, bias=False).half())

    layer = nn.Module()
    layer.add_module("self_attn", attention)
    layer.add_module("mlp", mlp)
    return layer


def _run_rank_summary_worker(rank, world_size, shared_ctx, results_queue):
    """spawn 子进程：各 rank 并发写入共享 context，rank 0 在 barrier 后聚合并回传。"""
    _ = world_size
    context = SummaryContext.from_context(shared_ctx)
    layer = _build_layer(rank)

    quant_summary.linear.record(context, _LAYER_NAME, layer, is_before_quant=True)
    setattr(layer.mlp, "rank%d_proj" % rank, _FakeQuantLinear())
    quant_summary.linear.record(context, _LAYER_NAME, layer, is_before_quant=False)
    quant_summary.fa.record(context, _LAYER_NAME, layer)
    quant_summary.kv.record(context, _LAYER_NAME, layer)
    quant_summary.time.record(context, _LAYER_NAME, 1.0 + rank)

    dist.barrier()
    values = quant_summary.finish(shared_ctx) if rank == 0 else None
    results_queue.put((rank, sorted(context.state.keys()), values))


@pytest.fixture(scope="module")
def two_rank_summary():
    """真实 spawn 两个进程共享一个 ``SharedDictContext``，返回各 rank 的写入结果。"""
    ctx = SharedDictContext(enable_debug=True)
    try:
        SummaryContext.from_context(ctx)
        results_queue = spawn_queue()
        run_distributed_spawn(2, _run_rank_summary_worker, fn_args=(ctx, results_queue))
        results = sorted((results_queue.get(timeout=10) for _ in range(2)), key=lambda item: item[0])
        yield ctx, results
    finally:
        ctx._manager.shutdown()


def test_summary_writes_rank_owned_sections_when_two_ranks_record_the_same_layer(two_rank_summary):
    ctx, results = two_rank_summary
    for rank, keys, _ in results:
        for dimension in ("linear", "fa", "kv"):
            assert "%s@rank_%d" % (dimension, rank) in keys
    assert "time" in results[0][1]

    # 同层不同前缀的分片都留在各自 rank 段中，未互相覆盖。
    state = ctx[SUMMARY_NAMESPACE].state
    assert set(state["linear@rank_0"][_LAYER_NAME]) == {"mlp.rank0_proj"}
    assert set(state["linear@rank_1"][_LAYER_NAME]) == {"mlp.rank1_proj"}


def test_summary_aggregates_all_rank_sections_when_rank_zero_finishes(two_rank_summary):
    values = two_rank_summary[1][0][2]
    # 两个 rank 各贡献一个 4x4 FP16 线性层：64B -> 各 16B INT8，合计 50%。
    assert values["linear"].original_bytes == pytest.approx(64.0)
    assert values["linear"].quantized_bytes == pytest.approx(32.0)
    assert values["linear"].widths == (8,)
    assert values["fa"].saving == pytest.approx(0.5)
    assert values["kv"].saving == pytest.approx(0.5)
    # 耗时只由 rank 0 写入，不叠加其他 rank。
    assert values["time"].layer_count == 1
    assert values["time"].total_elapsed == pytest.approx(1.0)


def test_summary_merges_rank_sections_when_queried_outside_the_worker(two_rank_summary):
    ctx, _ = two_rank_summary
    context = SummaryContext.from_context(ctx)
    linear = quant_summary.linear.aggregate(context)
    fa = quant_summary.fa.aggregate(context)
    kv = quant_summary.kv.aggregate(context)
    assert linear.original_bytes == pytest.approx(64.0)
    assert linear.quantized_bytes == pytest.approx(32.0)
    assert fa.saving == pytest.approx(0.5)
    assert kv.saving == pytest.approx(0.5)
