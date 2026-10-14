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

CheckpointReader（convert_design.md §7.3 / infra 层）。

无模型代码读 safetensors：
  - ``read_catalog``：优先解析 index.json；无 index 且仅有 ``model.safetensors`` 时
    打开该文件枚举 key 建 map（与 HuggingFace 小模型布局兼容）
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

_SINGLE_MODEL_SAFETENSORS = "model.safetensors"
_INDEX_JSON = "model.safetensors.index.json"


class CheckpointReader(ICheckpointReader):
    """``model_path`` 下含 ``model.safetensors.index.json``，或单文件 ``model.safetensors``。"""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._weight_map: dict[str, str] | None = None
        self._config: dict[str, Any] | None = None
        self._shard_meta: dict[str, dict[str, tuple[str, tuple[int, ...]]]] | None = None

    def read_weight_map(self) -> dict[str, str]:
        """tensor key -> shard 相对路径（与 index.json 的 weight_map 同构）。"""
        if self._weight_map is not None:
            return self._weight_map
        index_path = self.model_path / _INDEX_JSON
        if index_path.is_file():
            data = json.loads(index_path.read_text(encoding="utf-8"))
            self._weight_map = dict(data.get("weight_map", {}))
            return self._weight_map

        single = self.model_path / _SINGLE_MODEL_SAFETENSORS
        if single.is_file():
            self._weight_map = self._weight_map_from_single_file(single)
            return self._weight_map

        raise FileNotFoundError(f"No {_INDEX_JSON} or {_SINGLE_MODEL_SAFETENSORS} under {self.model_path}")

    @staticmethod
    def _weight_map_from_single_file(single: Path) -> dict[str, str]:
        """无 index 时从单文件枚举 key，映射到相对文件名 ``model.safetensors``。"""
        from safetensors import safe_open

        shard_name = single.name
        with safe_open(str(single), framework="pt", device="cpu") as f:
            weight_map = {key: shard_name for key in f.keys()}
        logger.info(
            "No %s; built weight_map from %s (%d keys)",
            _INDEX_JSON,
            shard_name,
            len(weight_map),
        )
        return weight_map

    def read_catalog(self) -> TensorCatalog:
        """快速建 catalog：dtype/shape 为 UNKNOWN/()，由 ``enrich_catalog`` 按需填充。"""
        weight_map = self.read_weight_map()
        catalog = TensorCatalog()
        for key, shard in weight_map.items():
            catalog.add(TensorEntry(key=key, shard=shard, dtype="UNKNOWN", shape=()))
        logger.info("Built catalog from weight_map (%d keys), headers deferred", len(catalog))
        return catalog

    def _load_all_shard_metadata(self) -> dict[str, dict[str, tuple[str, tuple[int, ...]]]]:
        """全量 shard header 缓存（``enrich_catalog(keys=None)`` 时使用）。"""
        if self._shard_meta is not None:
            return self._shard_meta
        from safetensors import safe_open

        weight_map = self.read_weight_map()
        shards = sorted(set(weight_map.values()))
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
