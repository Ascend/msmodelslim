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
Unit tests for BufferedSafetensorsWriter.
"""

# pylint: disable=redefined-outer-name

import tempfile
from unittest.mock import MagicMock

import pytest
import torch

from msmodelslim.core.quant_service.modelslim_v1.save.utils.safetensors import BufferedSafetensorsWriter


@pytest.fixture
def writer(temp_dir):
    """BufferedSafetensorsWriter with mock logger and temp directory."""
    logger = MagicMock()
    # BufferedSafetensorsWriter sets save_directory via setter which calls get_write_directory
    w = BufferedSafetensorsWriter(
        logger=logger,
        max_gb_size=4,
        save_directory=temp_dir,
        save_prefix="quant_model_weights",
    )
    return w


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    return d


class TestDedupeSharedStorage:
    """Tests for _dedupe_shared_storage."""

    def test_dedupe_returns_same_tensors_when_no_shared_storage(self, writer):
        """场景：各 key 独立 tensor。预期：输出与输入相同。"""
        a = torch.randn(2, 3)
        b = torch.randn(4, 5)
        keys_dict = {"model.layer0.weight": a, "model.layer1.weight": b}
        out = writer._dedupe_shared_storage(keys_dict)
        assert set(out.keys()) == set(keys_dict.keys())
        assert out["model.layer0.weight"] is a
        assert out["model.layer1.weight"] is b

    def test_dedupe_clones_lm_head_when_shared_with_embed_tokens(self, writer):
        """场景：embed_tokens 与 lm_head 共享 storage。预期：保留 embed_tokens，克隆 lm_head。"""
        t = torch.randn(4, 8)
        keys_dict = {
            "model.embed_tokens.weight": t,
            "model.lm_head.weight": t,
        }
        out = writer._dedupe_shared_storage(keys_dict)
        assert "model.embed_tokens.weight" in out
        assert "model.lm_head.weight" in out
        # embed_tokens is first in sort order (0), so it gets the original
        assert out["model.embed_tokens.weight"] is t
        # lm_head shares storage, so it gets a clone
        assert out["model.lm_head.weight"] is not t
        assert out["model.lm_head.weight"].data_ptr() != t.data_ptr()
        torch.testing.assert_close(out["model.lm_head.weight"], t)


class TestBufferedSafetensorsWriterIO:
    """Tests for write / save_one_file / close / save_index."""

    def test_write_skips_meta_tensor(self, writer):
        """场景：meta device tensor。预期：跳过写入。"""
        meta_t = torch.randn(2, 2, device="meta")
        writer.write("meta.weight", meta_t)
        assert "meta.weight" not in writer.wait_save_keys

    def test_write_and_close_persists_safetensors(self, writer, temp_dir):
        """场景：写入小张量后 close。预期：生成 safetensors 分片文件。"""
        writer.write("w", torch.ones(2, 2))
        writer.close()
        import os

        files = [f for f in os.listdir(temp_dir) if f.endswith(".safetensors")]
        assert len(files) >= 1

    def test_save_one_file_noop_when_empty(self, writer):
        """场景：无待写 tensor 调用 save_one_file。预期：不增加 save 计数。"""
        writer.save_one_file()
        assert writer._save_count == 0

    def test_get_index_json_includes_metadata_when_called(self):
        """场景：调用 get_index_json。预期：含 total_size 与 weight_map。"""
        from msmodelslim.core.quant_service.modelslim_v1.save.utils.safetensors import get_index_json

        idx = get_index_json({"a": "f1.safetensors"}, 100)
        assert idx["metadata"]["total_size"] == 100
        assert idx["weight_map"]["a"] == "f1.safetensors"
