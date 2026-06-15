#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
CheckpointReader（convert_design.md §7.3 / infra 层）。

无模型代码读 safetensors：
  - ``read_catalog``：仅解析 index.json，O(keys) 不打开 shard（避免 6 万 key 重复 safe_open）
  - ``enrich_catalog``：按 binding key 集合按需打开 shard，每 shard 最多一次
  - ``load_tensors``：按 inverse_weight_map 批量加载张量数据
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from msmodelslim.core.convert.catalog import TensorCatalog, TensorEntry
from msmodelslim.core.convert.protocol import ICheckpointReader
from msmodelslim.infra.io.shard_handle_cache import ShardHandleCache
from msmodelslim.utils.logging import get_logger

logger = get_logger()


class CheckpointReader(ICheckpointReader):
    """``model_path`` 下含 ``model.safetensors.index.json`` 或单文件 ``model.safetensors``。"""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._weight_map: dict[str, str] | None = None
        self._config: dict[str, Any] | None = None
        self._shard_meta: dict[str, dict[str, tuple[str, tuple[int, ...]]]] | None = None

    def read_weight_map(self) -> dict[str, str]:
        """原始 index：tensor key -> shard 相对路径。"""
        if self._weight_map is not None:
            return self._weight_map
        index_path = self.model_path / "model.safetensors.index.json"
        if not index_path.is_file():
            single = self.model_path / "model.safetensors"
            if single.is_file():
                self._weight_map = {"__single__": str(single)}
                return self._weight_map
            raise FileNotFoundError(f"No safetensors index or single file under {self.model_path}")
        data = json.loads(index_path.read_text(encoding="utf-8"))
        self._weight_map = dict(data.get("weight_map", {}))
        return self._weight_map

    def read_catalog(self) -> TensorCatalog:
        """快速建 catalog：dtype/shape 为 UNKNOWN/()，由 ``enrich_catalog`` 按需填充。"""
        weight_map = self.read_weight_map()
        catalog = TensorCatalog()
        for key, shard in weight_map.items():
            if key == "__single__":
                continue
            catalog.add(TensorEntry(key=key, shard=shard, dtype="UNKNOWN", shape=()))
        logger.info("Built catalog from index (%d keys), headers deferred", len(catalog))
        return catalog

    def _load_all_shard_metadata(self) -> dict[str, dict[str, tuple[str, tuple[int, ...]]]]:
        """全量 shard header 缓存（``enrich_catalog(keys=None)`` 时使用）。"""
        if self._shard_meta is not None:
            return self._shard_meta
        from safetensors import safe_open

        weight_map = self.read_weight_map()
        shards = sorted({s for _, s in weight_map.items() if _ != "__single__"})
        meta: dict[str, dict[str, tuple[str, tuple[int, ...]]]] = {}
        for i, shard_name in enumerate(shards, 1):
            shard_path = self.model_path / shard_name
            shard_meta: dict[str, tuple[str, tuple[int, ...]]] = {}
            with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    sl = f.get_slice(key)
                    dtype = str(sl.get_dtype()) if hasattr(sl, "get_dtype") else "UNKNOWN"
                    shape = tuple(sl.get_shape()) if hasattr(sl, "get_shape") else ()
                    shard_meta[key] = (dtype, shape)
            meta[shard_name] = shard_meta
            logger.info("Shard metadata %d/%d: %s (%d tensors)", i, len(shards), shard_name, len(shard_meta))
        self._shard_meta = meta
        return meta

    def enrich_catalog(self, catalog: TensorCatalog, keys: set[str] | None = None) -> None:
        """
        写入 dtype/shape。

        ``keys`` 非空时只打开包含这些 key 的 shard（virtual_tree 绑定 + fused_from 源 key）。
        """
        weight_map = self.read_weight_map()
        if keys is None:
            meta = self._load_all_shard_metadata()
            for key, entry in catalog.items():
                if key in meta.get(entry.shard, {}):
                    d, s = meta[entry.shard][key]
                    catalog.add(TensorEntry(key=key, shard=entry.shard, dtype=d, shape=s))
            return

        shards_needed: dict[str, set[str]] = {}
        for key in keys:
            shard = weight_map.get(key)
            if shard:
                shards_needed.setdefault(shard, set()).add(key)

        from safetensors import safe_open

        for shard_name, wanted in shards_needed.items():
            if self._shard_meta and shard_name in self._shard_meta:
                for key in wanted:
                    if key in self._shard_meta[shard_name]:
                        d, s = self._shard_meta[shard_name][key]
                        entry = catalog.get(key)
                        if entry:
                            catalog.add(TensorEntry(key=key, shard=entry.shard, dtype=d, shape=s))
                continue
            shard_path = self.model_path / shard_name
            with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                for key in wanted:
                    if key not in f.keys():
                        continue
                    sl = f.get_slice(key)
                    d = str(sl.get_dtype()) if hasattr(sl, "get_dtype") else "UNKNOWN"
                    sh = tuple(sl.get_shape()) if hasattr(sl, "get_shape") else ()
                    entry = catalog.get(key)
                    if entry:
                        catalog.add(TensorEntry(key=key, shard=entry.shard, dtype=d, shape=sh))

    def read_header(self, key: str) -> tuple[str, tuple[int, ...]]:
        """单 key header（会触发全 shard meta 加载，大批量场景请用 enrich_catalog）。"""
        weight_map = self.read_weight_map()
        shard = weight_map.get(key)
        if shard is None:
            raise KeyError(key)
        meta = self._load_all_shard_metadata()
        return meta[shard][key]

    def load_tensors(
        self,
        inverse_weight_map: dict[str, list[str] | None],
        device: str = "cpu",
    ) -> dict[str, Any]:
        """按 shard 批量读 tensor；返回 dict[key, Tensor]。"""
        cache = getattr(self, "shard_handle_cache", None)
        if isinstance(cache, ShardHandleCache):
            return cache.load_tensors(self.model_path, inverse_weight_map, device=device)
        local_cache = ShardHandleCache(max_shards=1)
        try:
            return local_cache.load_tensors(
                self.model_path,
                inverse_weight_map,
                device=device,
            )
        finally:
            local_cache.clear()

    def read_model_config(self) -> dict[str, Any]:
        """读取 ``config.json``（不存在则返回空 dict）。"""
        if self._config is not None:
            return self._config
        config_path = self.model_path / "config.json"
        if config_path.is_file():
            self._config = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            self._config = {}
        return self._config
