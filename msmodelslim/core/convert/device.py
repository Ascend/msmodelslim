#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Convert 阶段的设备解析与 worker 并发上限工具。

``worker_backend=process`` 时全程固定 CPU；thread 后端可解析到 NPU。
"""

from __future__ import annotations

import torch

from msmodelslim.utils.logging import get_logger

logger = get_logger()


def npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _npu_device_count() -> int:
    """当前可用 NPU 设备数量（仅在 ``npu_available()`` 为真时调用）。"""
    return int(torch.npu.device_count())


def resolve_worker_device(worker_device: str | None) -> str:
    """
    将 ``parallel.worker_device`` 解析为 safetensors / torch 可用的设备字符串。

    支持：
      - ``auto``：有 NPU 时用 ``npu:0``，否则 ``cpu``
      - ``cpu`` / ``npu``：简写；``npu`` 等价于 ``npu:0``
      - ``npu:0`` / ``npu:1`` 等：显式指定单卡
    """
    spec = (worker_device or "auto").strip()
    lowered = spec.lower()

    if lowered == "auto":
        if npu_available():
            return "npu:0"
        logger.warning("No NPU available; convert weights on CPU instead")
        return "cpu"

    if lowered == "cpu":
        return "cpu"

    if lowered == "npu":
        if not npu_available():
            logger.warning("worker_device=npu but NPU unavailable; falling back to CPU")
            return "cpu"
        return "npu:0"

    if lowered.startswith("npu"):
        if not npu_available():
            logger.warning("worker_device=%r but NPU unavailable; falling back to CPU", spec)
            return "cpu"
        return spec

    raise ValueError(
        f"Unsupported worker_device {worker_device!r}; "
        "expected cpu, npu, auto, or an explicit npu:<index> device string"
    )


def resolve_multi_worker_devices(device_indices: list[int] | None) -> list[str]:
    """
    将 CLI ``--device npu --device_id`` 的卡索引映射为每进程设备串。

    空 / None / NPU 不可用返回 ``[]``（走 CPU 路径）；否则 ``["npu:i", ...]``（走 NPU 路径）。
    索引须非负、唯一、且在 ``torch.npu.device_count()`` 范围内。
    """
    if not device_indices:
        return []
    if not npu_available():
        logger.warning(
            "device_indices=%s but NPU unavailable; convert multi-NPU disabled, falling back to default path",
            device_indices,
        )
        return []
    if len(device_indices) != len(set(device_indices)):
        raise ValueError(f"Duplicate device indices: {device_indices}")
    max_count = _npu_device_count()
    invalid = [idx for idx in device_indices if idx < 0 or idx >= max_count]
    if invalid:
        raise ValueError(f"Device indices {invalid} out of range [0, {max_count - 1}] for {max_count} NPU device(s)")
    return [f"npu:{idx}" for idx in device_indices]


def effective_convert_workers(
    max_workers: int,
    resolved_worker_device: str,
    npu_max_workers: int,
) -> int:
    """
    NPU 上多 worker 并发会把多张 2D 权重 + 量化中间张量同时驻留显存，易 OOM。
    在 accelerator 模式下将组内并发上限压到 ``npu_max_workers``（默认 1）。
    """
    workers = max(1, max_workers)
    if resolved_worker_device == "cpu":
        return workers
    cap = max(1, npu_max_workers)
    return min(workers, cap)
