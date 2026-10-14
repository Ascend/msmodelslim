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

msmodelslim.infra.io.shard_handle_cache 模块的单元测试
"""

import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors.torch import save_file

from msmodelslim.infra.io.checkpoint_reader import CheckpointReader
from msmodelslim.infra.io.shard_handle_cache import ShardHandleCache


class TestShardHandleCache:
    """测试 ShardHandleCache 类"""

    def test_load_tensors_reuse_handle_when_same_shard_loaded_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "w.safetensors"
            save_file({"a": torch.ones(2, 2), "b": torch.zeros(3, 3)}, shard)

            cache = ShardHandleCache(max_shards=1)
            reader = CheckpointReader(root)
            reader.shard_handle_cache = cache

            out1 = reader.load_tensors({str(shard.name): ["a"]})
            out2 = reader.load_tensors({str(shard.name): ["b"]})

            assert out1["a"].shape == (2, 2)  # 校验第一次加载
            assert out2["b"].shape == (3, 3)  # 校验第二次复用 handle
            assert cache.opens == 1  # 校验仅打开一次
            assert cache.hits == 1  # 校验命中缓存
            cache.clear()

    def test_load_tensors_share_handle_when_parallel_loads_same_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "w.safetensors"
            save_file({f"t{i}": torch.ones(2, 2) for i in range(8)}, shard)

            cache = ShardHandleCache(max_shards=1)
            reader = CheckpointReader(root)
            reader.shard_handle_cache = cache

            def _load(i: int):
                return reader.load_tensors({str(shard.name): [f"t{i}"]})[f"t{i}"]

            with ThreadPoolExecutor(max_workers=4) as pool:
                tensors = list(pool.map(_load, range(8)))

            assert len(tensors) == 8  # 校验并发加载数量
            assert cache.opens == 1  # 校验单 shard 只 open 一次
            assert cache.hits == 7  # 校验其余为 cache hit
            cache.clear()

    def test_clear_close_handles_when_called(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "w.safetensors"
            save_file({"a": torch.ones(2, 2)}, shard)

            cache = ShardHandleCache(max_shards=2)
            cache.load_tensors(root, {str(shard.name): ["a"]})
            assert len(cache._entries) == 1  # 校验缓存有条目
            cache.clear()
            assert len(cache._entries) == 0  # 校验 clear 后为空
