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
"""
# pylint: disable=consider-using-with

# Unit tests for the expert-parallel helpers added to the VL model adapter.
# The tests build a miniature safetensors shard on disk and drive the adapter
# methods through ``object.__new__`` so no real 235B checkpoint is required.

import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from msmodelslim.model.qwen3_vl_moe import model_adapter


def _write_safetensors(path: Path, tensors: dict) -> None:
    """Serialise ``tensors`` (name -> float32 tensor) into a safetensors file."""
    header = {}
    blobs = []
    offset = 0
    for name, tensor in tensors.items():
        data = tensor.contiguous().view(torch.float32).numpy().tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    header_bytes = json.dumps(header).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        for blob in blobs:
            handle.write(blob)


def _new_adapter(model_path: Path):
    """Create an adapter instance without running the heavy __init__."""
    adapter = object.__new__(model_adapter.Qwen3VLMoeModelAdapter)
    adapter.model_path = str(model_path)
    return adapter


class TestCollectTensorRanges(unittest.TestCase):
    """``_collect_tensor_ranges`` maps tensor names to absolute byte ranges."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.tensors = {
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(4, 6),
            "model.language_model.layers.0.self_attn.k_proj.weight": torch.randn(2, 6),
        }
        _write_safetensors(self.root / "model-00001-of-00002.safetensors", self.tensors)

    def tearDown(self):
        self.tmp.cleanup()

    def _expected_ranges(self, file_name: str):
        path = self.root / file_name
        with path.open("rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
            base = 8 + header_len
        return {
            name: (base + meta["data_offsets"][0], meta["data_offsets"][1] - meta["data_offsets"][0])
            for name, meta in header.items()
        }

    def test_returns_one_request_per_tensor(self):
        adapter = _new_adapter(self.root)
        groups = {"model-00001-of-00002.safetensors": list(self.tensors)}
        requests = adapter._collect_tensor_ranges(groups)
        self.assertEqual(len(requests), len(self.tensors))
        for file_path, offset, length in requests:
            self.assertEqual(file_path, str(self.root / "model-00001-of-00002.safetensors"))
            self.assertGreater(length, 0)
            self.assertGreaterEqual(offset, 8)

    def test_offsets_match_header_metadata(self):
        adapter = _new_adapter(self.root)
        expected = self._expected_ranges("model-00001-of-00002.safetensors")
        groups = {"model-00001-of-00002.safetensors": sorted(self.tensors)}
        requests = adapter._collect_tensor_ranges(groups)
        for (_, offset, length), name in zip(requests, sorted(self.tensors)):
            self.assertEqual((offset, length), expected[name])

    def test_missing_tensor_is_skipped(self):
        adapter = _new_adapter(self.root)
        groups = {"model-00001-of-00002.safetensors": ["not.there", list(self.tensors)[0]]}
        requests = adapter._collect_tensor_ranges(groups)
        self.assertEqual(len(requests), 1)

    def test_missing_file_is_skipped(self):
        adapter = _new_adapter(self.root)
        groups = {"model-99999-of-00002.safetensors": ["anything"]}
        self.assertEqual(adapter._collect_tensor_ranges(groups), [])

    def test_empty_groups_returns_empty_list(self):
        adapter = _new_adapter(self.root)
        self.assertEqual(adapter._collect_tensor_ranges({}), [])


class TestPrefetchLayerRanges(unittest.TestCase):
    """``_prefetch_layer_ranges`` reads the requested ranges into the page cache."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "shard.safetensors"
        self.path.write_bytes(bytes(range(256)) * 400)  # 102400 bytes

    def tearDown(self):
        self.tmp.cleanup()

    def test_reads_without_error(self):
        adapter = _new_adapter(self.root)
        adapter._prefetch_layer_ranges([(str(self.path), 0, 1024), (str(self.path), 2048, 512)])

    def test_tolerates_missing_file(self):
        adapter = _new_adapter(self.root)
        adapter._prefetch_layer_ranges([(str(self.root / "nope.bin"), 0, 64)])

    def test_tolerates_offset_beyond_eof(self):
        adapter = _new_adapter(self.root)
        adapter._prefetch_layer_ranges([(str(self.path), 10**7, 64)])

    def test_reads_in_multiple_blocks(self):
        """A request larger than the 4 MiB block size must loop until satisfied."""
        adapter = _new_adapter(self.root)
        adapter._prefetch_layer_ranges([(str(self.path), 0, self.path.stat().st_size)], workers=2)

    def test_empty_request_list(self):
        adapter = _new_adapter(self.root)
        adapter._prefetch_layer_ranges([])


class TestLoadExpertWeightSlices(unittest.TestCase):
    """``_load_expert_weight_slices`` pulls only the requested expert range."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.num_experts = 8
        self.hidden = 4
        self.inter = 3
        self.gate_up = torch.randn(self.num_experts, self.hidden, 2 * self.inter)
        self.down = torch.randn(self.num_experts, self.inter, self.hidden)
        shard_name = "model-00001-of-00001.safetensors"
        _write_safetensors(
            self.root / shard_name,
            {
                "model.language_model.layers.0.mlp.experts.gate_up_proj": self.gate_up,
                "model.language_model.layers.0.mlp.experts.down_proj": self.down,
            },
        )
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {},
                    "weight_map": {
                        "model.language_model.layers.0.mlp.experts.gate_up_proj": shard_name,
                        "model.language_model.layers.0.mlp.experts.down_proj": shard_name,
                    },
                }
            )
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_slice_matches_full_tensor_slice(self):
        adapter = _new_adapter(self.root)
        gate_slice, down_slice = adapter._load_expert_weight_slices(0, 2, 5)
        self.assertEqual(gate_slice.shape, (3, self.hidden, 2 * self.inter))
        self.assertEqual(down_slice.shape, (3, self.inter, self.hidden))
        self.assertTrue(torch.equal(gate_slice, self.gate_up[2:5]))
        self.assertTrue(torch.equal(down_slice, self.down[2:5]))

    def test_full_range_covers_every_expert(self):
        adapter = _new_adapter(self.root)
        gate_slice, down_slice = adapter._load_expert_weight_slices(0, 0, self.num_experts)
        self.assertTrue(torch.equal(gate_slice, self.gate_up))
        self.assertTrue(torch.equal(down_slice, self.down))

    def test_empty_range_returns_zero_length_slice(self):
        adapter = _new_adapter(self.root)
        gate_slice, down_slice = adapter._load_expert_weight_slices(0, 3, 3)
        self.assertEqual(gate_slice.shape[0], 0)
        self.assertEqual(down_slice.shape[0], 0)

    def test_missing_index_raises(self):
        adapter = _new_adapter(self.root)
        (self.root / "model.safetensors.index.json").unlink()
        with self.assertRaises(Exception):
            adapter._load_expert_weight_slices(0, 0, 1)

    def test_missing_expert_key_raises(self):
        adapter = _new_adapter(self.root)
        index = json.loads((self.root / "model.safetensors.index.json").read_text())
        index["weight_map"].pop("model.language_model.layers.0.mlp.experts.down_proj")
        (self.root / "model.safetensors.index.json").write_text(json.dumps(index))
        with self.assertRaises(model_adapter.InvalidModelError):
            adapter._load_expert_weight_slices(0, 0, 1)


class TestGetStateDictEpFilter(unittest.TestCase):
    """Under EP the fused expert tensors are skipped in ``_get_state_dict``."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.tensors = {
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),
            "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(8, 4, 6),
            "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(8, 3, 4),
        }
        shard = "model-00001-of-00001.safetensors"
        _write_safetensors(self.root / shard, self.tensors)
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {name: shard for name in self.tensors}})
        )

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def _module():
        return SimpleNamespace(
            named_parameters=lambda: [
                ("self_attn.q_proj.weight", nn.Parameter(torch.zeros(4, 4))),
                ("mlp.experts.gate_up_proj", nn.Parameter(torch.zeros(8, 4, 6))),
                ("mlp.experts.down_proj", nn.Parameter(torch.zeros(8, 3, 4))),
            ]
        )

    def test_ep_mode_skips_fused_experts(self):
        adapter = _new_adapter(self.root)
        with patch.object(model_adapter.dist, "is_initialized", return_value=True):
            with patch.object(model_adapter.dist, "get_world_size", return_value=4):
                state = adapter._get_state_dict(self._module(), prefix="model.language_model.layers.0")
        self.assertIn("self_attn.q_proj.weight", state)
        self.assertNotIn("mlp.experts.gate_up_proj", state)
        self.assertNotIn("mlp.experts.down_proj", state)

    def test_single_card_loads_experts(self):
        adapter = _new_adapter(self.root)
        with patch.object(model_adapter.dist, "is_initialized", return_value=False):
            state = adapter._get_state_dict(self._module(), prefix="model.language_model.layers.0")
        self.assertEqual(len(state), 3)
        self.assertTrue(
            torch.equal(
                state["mlp.experts.gate_up_proj"],
                self.tensors["model.language_model.layers.0.mlp.experts.gate_up_proj"],
            )
        )

    def test_world_size_one_keeps_experts(self):
        adapter = _new_adapter(self.root)
        with patch.object(model_adapter.dist, "is_initialized", return_value=True):
            with patch.object(model_adapter.dist, "get_world_size", return_value=1):
                state = adapter._get_state_dict(self._module(), prefix="model.language_model.layers.0")
        self.assertIn("mlp.experts.gate_up_proj", state)


class TestLnFuseMap(unittest.TestCase):
    """``_qwen3_vl_moe_get_ln_fuse_map`` only names locally owned experts."""

    @staticmethod
    def _config(layers=2, experts=8):
        return SimpleNamespace(num_hidden_layers=layers, num_experts=experts)

    def test_default_names_all_experts(self):
        fuse = model_adapter._qwen3_vl_moe_get_ln_fuse_map(self._config(experts=8))
        experts = [
            name for name in fuse["model.language_model.layers.0.post_attention_layernorm"] if ".mlp.experts." in name
        ]
        # 8 experts x (gate_proj, up_proj)
        self.assertEqual(len(experts), 16)

    def test_local_range_filters_experts(self):
        fuse = model_adapter._qwen3_vl_moe_get_ln_fuse_map(self._config(experts=8), 2, 4)
        entries = fuse["model.language_model.layers.0.post_attention_layernorm"]
        for expert_id in range(8):
            present = any(f".mlp.experts.{expert_id}." in name for name in entries)
            self.assertEqual(present, 2 <= expert_id < 4, f"expert {expert_id}")

    def test_router_gate_always_present(self):
        fuse = model_adapter._qwen3_vl_moe_get_ln_fuse_map(self._config(experts=8), 2, 4)
        entries = fuse["model.language_model.layers.0.post_attention_layernorm"]
        self.assertIn("model.language_model.layers.0.mlp.gate", entries)

    def test_qkv_mapping_is_independent_of_expert_range(self):
        full = model_adapter._qwen3_vl_moe_get_ln_fuse_map(self._config(experts=8))
        local = model_adapter._qwen3_vl_moe_get_ln_fuse_map(self._config(experts=8), 4, 8)
        key = "model.language_model.layers.0.input_layernorm"
        self.assertEqual(full[key], local[key])
        self.assertEqual(len(local[key]), 3)

    def test_every_decoder_layer_present(self):
        fuse = model_adapter._qwen3_vl_moe_get_ln_fuse_map(self._config(layers=3, experts=4))
        for layer in range(3):
            self.assertIn(f"model.language_model.layers.{layer}.input_layernorm", fuse)
            self.assertIn(f"model.language_model.layers.{layer}.post_attention_layernorm", fuse)
        self.assertIn("model.language_model.norm", fuse)
        self.assertEqual(fuse["model.language_model.norm"], ["lm_head"])

    def test_empty_expert_range_leaves_only_gate(self):
        fuse = model_adapter._qwen3_vl_moe_get_ln_fuse_map(self._config(experts=8), 4, 4)
        entries = fuse["model.language_model.layers.0.post_attention_layernorm"]
        self.assertFalse(any(".mlp.experts." in name for name in entries))
        self.assertIn("model.language_model.layers.0.mlp.gate", entries)


if __name__ == "__main__":
    unittest.main()
