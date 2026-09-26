#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Convert 阶段的设备解析工具。

设备由 CLI ``--device`` / ``--device_id`` 决定（对齐 quant 语义，YAML 不含设备字段）：
``resolve_multi_worker_devices`` 将卡索引映射为每进程设备串。
"""

from __future__ import annotations

import torch

from msmodelslim.utils.exception import EnvError, SchemaValidateError


def npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _npu_device_count() -> int:
    """当前可用 NPU 设备数量（仅在 ``npu_available()`` 为真时调用）。"""
    return int(torch.npu.device_count())


def resolve_multi_worker_devices(device_indices: list[int] | None) -> list[str]:
    """
    将 CLI ``--device npu --device_id`` 的卡索引映射为每进程设备串。

    空 / None 返回 ``[]``（不启用多卡）；否则 ``["npu:i", ...]``（走 NPU 路径）。
    索引须非负、唯一、且在 ``torch.npu.device_count()`` 范围内；
    NPU 不可用时直接抛错（对齐 quant 语义：指定 NPU 但环境不可用须失败，不静默回落）。
    """
    if not device_indices:
        return []
    if not npu_available():
        raise EnvError(
            f"device_indices={device_indices} but no NPU is available; "
            "check --device/--device_id and the NPU environment (msmodelslim quant semantics: fail fast)",
            action="Use --device cpu for CPU conversion, or ensure NPU is available when using --device npu.",
        )
    if len(device_indices) != len(set(device_indices)):
        raise SchemaValidateError(f"Duplicate device indices: {device_indices}")
    max_count = _npu_device_count()
    invalid = [idx for idx in device_indices if idx < 0 or idx >= max_count]
    if invalid:
        raise SchemaValidateError(
            f"Device indices {invalid} out of range [0, {max_count - 1}] for {max_count} NPU device(s)"
        )
    return [f"npu:{idx}" for idx in device_indices]
