#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

# QAT 训练指标：qparam(α) 全规约统计、token 吞吐窗口与带节流的统计收集器。
# α 统计按注入的 qparam_names 匹配（不绑定具体算法的参数命名）。

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch import nn

from msmodelslim.core.train.device import accelerator_device, all_reduce_sum_float
from msmodelslim.core.train.optimizer import matches_qparam
from msmodelslim.utils.logging import get_logger


def is_eval_log(logs: Dict[str, Any]) -> bool:
    """日志 payload 是否为评估日志（含 ``eval_`` 前缀指标）。"""
    return any(str(key).startswith("eval_") for key in logs)


@dataclass(frozen=True)
class QParamStats:
    """qparam(α) 全规约统计量。"""

    count: int
    mean: float
    vmin: float
    vmax: float


def _local_tensor(param: Any) -> Optional[torch.Tensor]:
    """Get a plain local tensor from a (possibly sharded / DTensor) param."""
    tensor = param
    if hasattr(tensor, "to_local"):
        try:
            tensor = tensor.to_local()
        except Exception:  # pragma: no cover - dist backend drift
            return None
    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return None
    return tensor.detach()


def _local_qparam_accum(
    model: nn.Module,
    param_names: Tuple[str, ...],
) -> Tuple[int, float, float, float]:
    """累加本地命中 qparam 命名约定的 (n, sum, min, max)。"""
    local_n = 0
    local_sum = 0.0
    local_min = float("inf")
    local_max = float("-inf")
    for name, param in model.named_parameters():
        if not matches_qparam(name, param_names):
            continue
        tensor = _local_tensor(param)
        if tensor is None:
            continue
        values = tensor.float()
        local_n += int(values.numel())
        local_sum += float(values.sum().item())
        local_min = min(local_min, float(values.min().item()))
        local_max = max(local_max, float(values.max().item()))
    return local_n, local_sum, local_min, local_max


@torch.no_grad()
def qparam_summary_distributed(
    model: nn.Module,
    param_names: Sequence[str],
) -> Optional[QParamStats]:
    """全规约 qparam 统计 (count, mean, min, max)；无命中时返回 None。

    All ranks must call this together. Falls back to the local shard if dist
    is unavailable or the reduce fails.
    """
    local_n, local_sum, local_min, local_max = _local_qparam_accum(model, tuple(param_names))
    # HCCL (NPU) does not support float64 all-reduce; fp32 is enough for log stats.
    reduce_dtype = torch.float32
    n_sum = torch.tensor(
        [float(local_n), local_sum],
        device=accelerator_device(),
        dtype=reduce_dtype,
    )
    min_t = torch.tensor(
        local_min if local_n > 0 else 1.0e38,
        device=n_sum.device,
        dtype=reduce_dtype,
    )
    max_t = torch.tensor(
        local_max if local_n > 0 else -1.0e38,
        device=n_sum.device,
        dtype=reduce_dtype,
    )
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        try:
            dist.all_reduce(n_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(min_t, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_t, op=dist.ReduceOp.MAX)
        except Exception:
            get_logger().warning(
                "QAT qparam all-reduce failed; using local-shard stats.",
                exc_info=True,
            )
            n_sum = torch.tensor([float(local_n), local_sum], dtype=reduce_dtype)
            min_t = torch.tensor(local_min if local_n > 0 else 1.0e38, dtype=reduce_dtype)
            max_t = torch.tensor(local_max if local_n > 0 else -1.0e38, dtype=reduce_dtype)

    count = int(n_sum[0].item())
    if count <= 0:
        return None
    return QParamStats(
        count=count,
        mean=float(n_sum[1].item()) / count,
        vmin=float(min_t.item()),
        vmax=float(max_t.item()),
    )


class TokenThroughputTracker:
    """训练 token 吞吐统计窗口（两次 log 之间的 token 数 / 墙钟时间）。"""

    def __init__(self) -> None:
        self._tokens: float = 0.0
        self._wall0: Optional[float] = None

    def add_tokens(self, count: int) -> None:
        """累计本窗口内的 CE token 数。"""
        self._tokens += float(count)

    def reset(self) -> None:
        """清零窗口并重置墙钟基准。"""
        self._tokens = 0.0
        self._wall0 = time.perf_counter()

    def consume_per_sec(self) -> Optional[float]:
        """消费当前窗口并返回全局 tokens/sec；首个窗口（无基准）返回 None。"""
        now = time.perf_counter()
        local_n = self._tokens
        self._tokens = 0.0
        prev = self._wall0
        self._wall0 = now
        if prev is None:
            return None
        elapsed = now - prev
        if elapsed <= 0:
            return None
        return all_reduce_sum_float(local_n) / elapsed


class QParamStatsCollector:
    """qparam(α) 统计收集器：eval 或每 N 步全规约刷新，其余步复用缓存。

    qparam 移动缓慢，``logging_steps=1`` 时每步全规约会浪费 HCCL；
    刷新间隔取 ``max(min_interval, MIN_REFRESH_INTERVAL)``。
    """

    MIN_REFRESH_INTERVAL = 10

    def __init__(self, param_names: Sequence[str]):
        self._param_names = tuple(param_names)
        self._cache: Optional[QParamStats] = None
        self._cache_step: int = -1

    def summary(
        self,
        model: nn.Module,
        *,
        step: int,
        is_eval: bool,
        min_interval: int = 1,
    ) -> Optional[QParamStats]:
        """取当前 α 统计（按节流策略决定是否全规约刷新）。"""
        interval = max(int(min_interval), self.MIN_REFRESH_INTERVAL)
        need_refresh = is_eval or self._cache is None or (step - self._cache_step) >= interval
        if need_refresh:
            self._cache = qparam_summary_distributed(model, self._param_names)
            self._cache_step = step
        return self._cache
