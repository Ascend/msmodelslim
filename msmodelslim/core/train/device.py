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

# NPU 设备与分布式原语：QAT 训练栈的设备探测唯一入口（仅支持 NPU：
# torch_npu 包或 torch 内置 npu）。无 NPU 环境按 CPU 退化，仅供本地开发/单测。

from __future__ import annotations

import gc
import os
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist

from msmodelslim.utils.logging import get_logger


def _npu_module() -> Optional[Any]:
    """返回 NPU 命名空间（torch_npu.npu / torch.npu）；无 NPU 时为 None。"""
    try:
        import torch_npu

        if torch_npu.npu.is_available():
            return torch_npu.npu
    except Exception:
        # 注意：msmodelslim 自定义 logger 不支持 exc_info 透传，只保留消息本身。
        get_logger().debug("torch_npu is unavailable")
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        return npu
    return None


def accelerator_device() -> torch.device:
    """获取 all-reduce 等集合通信使用的当前 NPU 设备；无 NPU 时退化为 CPU。"""
    npu = _npu_module()
    if npu is None:
        return torch.device("cpu")
    current = getattr(npu, "current_device", lambda: 0)()
    return torch.device("npu", int(current))


def get_device_memory() -> Tuple[str, Optional[float], Optional[float]]:
    """返回 (device_name, allocated_GiB, reserved_GiB)；无 NPU 时 GiB 为 None。"""
    npu = _npu_module()
    if npu is None:
        return "cpu", None, None
    return (
        "npu",
        npu.memory_allocated() / 1024.0**3,
        npu.memory_reserved() / 1024.0**3,
    )


def os_local_rank() -> str:
    """读取进程本地 rank 环境变量（LOCAL_RANK 优先，回退 RANK）。"""
    return os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))


def log_device_memory(tag: str, rank: Optional[int] = None) -> None:
    """打印 NPU allocated/reserved 显存快照。"""
    device_name, allocated_gb, reserved_gb = get_device_memory()
    rank_s = rank if rank is not None else os_local_rank()
    if allocated_gb is None:
        get_logger().info("QAT memory[%s] rank=%s device=cpu (no accelerator stats)", tag, rank_s)
        return
    get_logger().info(
        "QAT memory[%s] rank=%s device=%s allocated=%.3fGiB reserved=%.3fGiB",
        tag,
        rank_s,
        device_name,
        allocated_gb,
        reserved_gb,
    )


def all_reduce_sum_float(value: float) -> float:
    """全规约 float32 SUM；未初始化分布式时直接返回 ``float(value)``。"""
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return float(value)
    tensor = torch.tensor([float(value)], device=accelerator_device(), dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def empty_device_cache() -> None:
    """gc 后清空 NPU 缓存。"""
    gc.collect()
    npu = _npu_module()
    if npu is not None:
        npu.empty_cache()
