#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# pylint: disable=use-implicit-booleaness-not-comparison

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

msmodelslim.infra.io.buffered_safetensors_writer 模块的单元测试
"""

import json
from unittest.mock import MagicMock

import torch
from safetensors.torch import load_file

from msmodelslim.infra.io import buffered_safetensors_writer
from msmodelslim.infra.io.buffered_safetensors_writer import BufferedSafetensorsWriter, get_index_json


class TestGetIndexJson:
    """测试 get_index_json 索引结构"""

    def test_build_metadata_and_weight_map(self):
        """正常路径：索引包含 total_size 元数据与 weight_map 映射"""
        result = get_index_json({"a": "model-00001-of-00001.safetensors"}, 1234)

        assert result == {
            "metadata": {"total_size": 1234},
            "weight_map": {"a": "model-00001-of-00001.safetensors"},
        }


class TestBufferedSafetensorsWriter:
    """测试 BufferedSafetensorsWriter 的分片与索引落盘行为"""

    def test_write_skip_meta_tensor(self, tmp_path):
        """边界：meta 设备张量被跳过，不进入待写队列也不计入体积"""
        writer = BufferedSafetensorsWriter(
            logger=MagicMock(), max_gb_size=4, save_directory=str(tmp_path), save_prefix="model"
        )

        writer.write("meta.weight", torch.empty(2, 2, device="meta"))

        assert writer.wait_save_keys == {}
        assert writer.total_size == 0

    def test_close_write_shards_and_index(self, tmp_path):
        """正常路径：多个张量经 close 落盘为分片文件并生成索引 JSON"""
        writer = BufferedSafetensorsWriter(
            logger=MagicMock(), max_gb_size=0, save_directory=str(tmp_path), save_prefix="model"
        )
        writer.write("a", torch.ones(2, 2))
        writer.write("b", torch.zeros(3))

        writer.close()

        index = json.loads((tmp_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
        assert index["metadata"]["total_size"] == writer.total_size
        assert sorted(index["weight_map"]) == ["a", "b"]
        shards = sorted(p.name for p in tmp_path.glob("model-*-of-*.safetensors"))
        assert len(shards) == 2
        for shard_name in index["weight_map"].values():
            assert shard_name in shards

    def test_write_flushes_shard_when_accumulated_size_reaches_limit(self, tmp_path, monkeypatch):
        """边界：累计体积恰好达到分片上限时，先落盘当前分片。"""
        monkeypatch.setattr(buffered_safetensors_writer, "ONE_GB_FILE_BYTES", 8)
        writer = BufferedSafetensorsWriter(
            logger=MagicMock(), max_gb_size=1, save_directory=str(tmp_path), save_prefix="model"
        )

        writer.write("first", torch.ones(1, dtype=torch.float32))
        writer.write("second", torch.ones(1, dtype=torch.float32))
        writer.close()

        shards = sorted(tmp_path.glob("model-*-of-*.safetensors"))
        assert len(shards) == 2

    def test_close_preserves_both_keys_when_tensors_share_storage(self, tmp_path):
        """不变量：共享存储的 tied weights 以独立张量落盘，两个键均可读取。"""
        writer = BufferedSafetensorsWriter(
            logger=MagicMock(), max_gb_size=4, save_directory=str(tmp_path), save_prefix="model"
        )
        shared = torch.tensor([1.0, 2.0, 3.0])
        writer.write("model.embed_tokens.weight", shared)
        writer.write("lm_head.weight", shared)

        writer.close()

        index = json.loads((tmp_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
        shard_path = tmp_path / index["weight_map"]["lm_head.weight"]
        saved = load_file(str(shard_path))
        assert set(saved) == {"model.embed_tokens.weight", "lm_head.weight"}
        assert torch.equal(saved["model.embed_tokens.weight"], shared)
        assert torch.equal(saved["lm_head.weight"], shared)
