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

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch.distributed as dist

from msmodelslim.core.context import get_current_context
from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.utils.logging import get_logger


def check_distributed_analysis_supported(supports_distributed: bool, method_name: str) -> None:
    """pre_run 多卡准入：指标未声明 supports_distributed 时 fail-fast。"""
    if dist.is_initialized() and dist.get_world_size() > 1 and not supports_distributed:
        raise UnsupportedError(
            f"Analysis method '{method_name}' does not support multi-device (DP) execution.",
            action=(
                "Set supports_distributed to True after implementing DP behavior, or run analysis on a single device."
            ),
        )


def maybe_barrier_before_linear_quant() -> None:
    """DP 多卡敏感层分析：进入 linear_quant 集体通信前对齐各 rank 进度。"""
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()


def dedupe_layer_scores_keep_max(
    layer_scores: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Collapse duplicate layer names, keeping the highest score.

    Omni adapters may visit the same module twice (e.g. visual for image then
    video); layer-wise analysis appends one score per visit.
    """
    best: Dict[str, float] = {}
    order: List[str] = []
    for item in layer_scores:
        name = item.get("name")
        score = item.get("score")
        if name is None or score is None:
            continue
        score_f = float(score)
        if name not in best:
            order.append(name)
            best[name] = score_f
        else:
            best[name] = max(best[name], score_f)
    if len(order) != len(layer_scores):
        get_logger().info(
            "Deduped analysis layer scores by name (keep max): %d -> %d",
            len(layer_scores),
            len(order),
        )
    return [{"name": name, "score": best[name]} for name in order]


def merge_layer_scores_across_ranks(
    local_scores: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Average layer sensitivity scores across DP ranks.

    Each rank may see a calib shard (DistributedSampler) and/or a subset of
    EP-local modules; missing layers on a rank are skipped when averaging.
    """
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return list(local_scores)

    gathered: List[Optional[List[Dict[str, Any]]]] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_scores)

    score_sums: Dict[str, float] = defaultdict(float)
    score_counts: Dict[str, int] = defaultdict(int)
    for rank_scores in gathered:
        if not rank_scores:
            continue
        for item in rank_scores:
            name = item.get("name")
            score = item.get("score")
            if name is None or score is None:
                continue
            score_sums[name] += float(score)
            score_counts[name] += 1

    merged = [{"name": name, "score": score_sums[name] / score_counts[name]} for name in sorted(score_sums.keys())]
    get_logger().info(
        "Merged analysis layer scores across %d ranks: local=%d, merged=%d",
        dist.get_world_size(),
        len(local_scores),
        len(merged),
    )
    return merged


def write_layer_analysis_result(
    layer_scores: List[Dict[str, Any]],
    method: str,
    *,
    patterns: Optional[List[str]] = None,
    quant_modules: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Write analysis payload to context (no dedupe/merge).

    Under DP only rank 0 writes. Used after method-specific enrich so extra
    fields (e.g. induction_heads) are stored as-is.
    """
    if dist.is_initialized() and dist.get_world_size() > 1 and dist.get_rank() != 0:
        return layer_scores

    ctx = get_current_context()
    if ctx is None:
        return layer_scores

    layer_analysis = ctx["layer_analysis"]  # pylint: disable=unsubscriptable-object
    payload = {
        "layer_scores": layer_scores,
        "method": method,
    }
    if patterns is not None:
        payload["patterns"] = patterns
    if quant_modules is not None:
        payload["quant_modules"] = list(quant_modules)

    def _write_store(store: Any) -> None:
        if store is None:
            return
        try:
            for key, value in payload.items():
                store[key] = value
        except (TypeError, AttributeError):
            return

    _write_store(getattr(layer_analysis, "state", None))
    _write_store(getattr(layer_analysis, "debug", None))
    return layer_scores


def publish_layer_analysis_result(
    layer_scores: List[Dict[str, Any]],
    method: str,
    *,
    patterns: Optional[List[str]] = None,
    quant_modules: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Dedupe, merge scores under DP if needed, then publish to context.

    Only handles ``name`` / ``score``. Method-specific fields should be added
    via ``enrich_layer_scores`` after publish, then ``write_layer_analysis_result``.

    Note: ``SharedNamespace.debug`` is process-local; parent after ``mp.spawn``
    must read ``state``, not ``debug``.
    """
    layer_scores = dedupe_layer_scores_keep_max(layer_scores)

    if dist.is_initialized() and dist.get_world_size() > 1:
        # all_gather_object already synchronizes ranks; no extra barrier.
        layer_scores = merge_layer_scores_across_ranks(layer_scores)

    return write_layer_analysis_result(
        layer_scores,
        method,
        patterns=patterns,
        quant_modules=quant_modules,
    )


def _store_get(store: Any, key: str, default: Any = None) -> Any:
    try:
        if hasattr(store, "get"):
            return store.get(key, default)
        return store[key] if key in store else default
    except (TypeError, KeyError):
        return default


def read_layer_analysis_result(ctx) -> Dict[str, Any]:
    """Read analysis results: prefer shared ``state``, fallback to ``debug``."""
    layer_analysis = ctx["layer_analysis"]
    for store_name in ("state", "debug"):
        store = getattr(layer_analysis, store_name, None)
        if store is None:
            continue
        try:
            if "layer_scores" not in store:
                continue
            layer_scores = store["layer_scores"]
        except (TypeError, KeyError):
            continue
        if not isinstance(layer_scores, list):
            continue
        return {
            "layer_scores": layer_scores,
            "method": _store_get(store, "method"),
            "patterns": _store_get(store, "patterns"),
            "quant_modules": _store_get(store, "quant_modules"),
        }
    raise KeyError(
        "layer_scores not found in layer_analysis.state/debug. Analysis processors may have failed to publish results."
    )
