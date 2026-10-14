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

Safetensors shard 级 handle 缓存（convert P0 IO 优化）。

同一 dependency / shard 组内多个 IR 任务共享已打开的 ``safe_open`` 句柄，
避免每个任务重复 ``open/mmap`` 同一 ``.safetensors`` 文件。
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

_CacheEntry = tuple[Any, threading.Lock]


class ShardHandleCache:
    """线程安全的 (shard_path, device) → open handle LRU 缓存，生命周期为一个 dependency group。"""

    def __init__(self, max_shards: int = 2) -> None:
        self._max = max(1, max_shards)
        self._entries: OrderedDict[tuple[str, str], _CacheEntry] = OrderedDict()
        self._guard = threading.Lock()
        self.opens = 0
        self.hits = 0

    def load_tensors(
        self,
        model_path: Path,
        inverse_weight_map: dict[str, list[str] | None],
        device: str = "cpu",
    ) -> dict[str, Any]:
        """与 ``CheckpointReader.load_tensors`` 相同语义，复用 shard handle。"""
        from safetensors import safe_open

        out: dict[str, Any] = {}
        for shard, names in inverse_weight_map.items():
            shard_path = Path(shard)
            if not shard_path.is_absolute():
                shard_path = model_path / shard
            abs_path = str(shard_path.resolve())
            cache_key = (abs_path, str(device))

            handle, handle_lock = self._get_or_open(
                cache_key,
                lambda path=abs_path: safe_open(path, framework="pt", device=device),
            )
            load_keys = names if names is not None else list(handle.keys())
            with handle_lock:
                for key in load_keys:
                    out[key] = handle.get_tensor(key)
        return out

    def _get_or_open(self, cache_key: tuple[str, str], opener: Callable[[], Any]) -> _CacheEntry:
        with self._guard:
            cached = self._entries.get(cache_key)
            if cached is not None:
                self._entries.move_to_end(cache_key)
                self.hits += 1
                return cached

        handle = opener()
        entry: _CacheEntry = (handle, threading.Lock())

        with self._guard:
            existing = self._entries.get(cache_key)
            if existing is not None:
                self._close_handle(handle)
                self._entries.move_to_end(cache_key)
                self.hits += 1
                return existing
            while len(self._entries) >= self._max:
                _, evicted = self._entries.popitem(last=False)
                self._close_handle(evicted[0])
            self._entries[cache_key] = entry
            self.opens += 1
            return entry

    @staticmethod
    def _close_handle(handle: Any) -> None:
        try:
            handle.__exit__(None, None, None)
        except Exception:  # nosec B110
            pass

    def clear(self) -> None:
        with self._guard:
            for handle, _ in self._entries.values():
                self._close_handle(handle)
            self._entries.clear()
