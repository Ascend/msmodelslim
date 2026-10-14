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

msmodelslim.infra.io.checkpoint_reader 模块的单元测试
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from msmodelslim.infra.io.checkpoint_reader import CheckpointReader


class TestCheckpointReader:
    """测试 CheckpointReader 类"""

    def test_read_weight_map_return_map_when_index_json_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "a.weight": "model-00001-of-00002.safetensors",
                            "b.weight": "model-00002-of-00002.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            reader = CheckpointReader(root)
            weight_map = reader.read_weight_map()
            assert weight_map == {
                "a.weight": "model-00001-of-00002.safetensors",
                "b.weight": "model-00002-of-00002.safetensors",
            }

    def test_read_weight_map_build_from_keys_when_only_model_safetensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_file(
                {
                    "model.layers.0.mlp.gate_proj.weight": torch.ones(2, 2),
                    "lm_head.weight": torch.zeros(3, 3),
                },
                root / "model.safetensors",
            )
            reader = CheckpointReader(root)
            weight_map = reader.read_weight_map()
            assert weight_map == {
                "model.layers.0.mlp.gate_proj.weight": "model.safetensors",
                "lm_head.weight": "model.safetensors",
            }

    def test_read_catalog_include_all_keys_when_only_model_safetensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_file(
                {
                    "embed.weight": torch.ones(2, 2),
                    "norm.weight": torch.ones(2),
                },
                root / "model.safetensors",
            )
            reader = CheckpointReader(root)
            catalog = reader.read_catalog()
            assert len(catalog) == 2
            assert catalog.get("embed.weight").shard == "model.safetensors"
            assert catalog.get("norm.weight").shard == "model.safetensors"

    def test_load_tensors_return_tensor_when_only_model_safetensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_file({"w": torch.arange(4).reshape(2, 2).float()}, root / "model.safetensors")
            reader = CheckpointReader(root)
            weight_map = reader.read_weight_map()
            out = reader.load_tensors({weight_map["w"]: ["w"]})
            assert out["w"].shape == (2, 2)
            assert torch.equal(out["w"], torch.arange(4).reshape(2, 2).float())

    def test_read_weight_map_raise_error_when_no_index_and_no_model_safetensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_file({"a": torch.ones(1)}, root / "model-00001-of-00001.safetensors")
            reader = CheckpointReader(root)
            with pytest.raises(FileNotFoundError, match="No model.safetensors.index.json"):
                reader.read_weight_map()

    def test_read_weight_map_prefer_index_when_both_index_and_single_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_file({"ignored": torch.ones(1)}, root / "model.safetensors")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"from_index": "model.safetensors"}}),
                encoding="utf-8",
            )
            reader = CheckpointReader(root)
            assert reader.read_weight_map() == {"from_index": "model.safetensors"}
