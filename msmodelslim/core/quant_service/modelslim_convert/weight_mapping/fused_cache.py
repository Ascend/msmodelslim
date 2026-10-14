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

Dependency-group 内 fused MoE 张量缓存（convert P0 优化）。

同一 ``fused_from`` 源上的多个 expert 任务共享一块 3D fused 权重；
组内首次加载后缓存，后续任务仅切片，避免重复 ``safe_open`` / 读盘。
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import torch


class FusedTensorCache:
    """线程安全的 (fused_key, device) → Tensor 缓存，生命周期为一个 dependency group。"""

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], torch.Tensor] = {}
        self._guard = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get_or_load(
        self,
        fused_key: str,
        device: str,
        loader: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        cache_key = (fused_key, str(device))
        with self._guard:
            cached = self._data.get(cache_key)
            if cached is not None:
                self.hits += 1
                return cached

        # 读盘在锁外执行，避免组内多 worker 串行等待 I/O
        tensor = loader()

        with self._guard:
            existing = self._data.get(cache_key)
            if existing is not None:
                self.hits += 1
                return existing
            self._data[cache_key] = tensor
            self.misses += 1
            return tensor

    def clear(self) -> None:
        with self._guard:
            self._data.clear()
