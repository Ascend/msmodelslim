#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""direct_save：merge 合同与写后释放引用。"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from safetensors.torch import save_file
from torch import nn
import torch

from msmodelslim.core.quant_service.modelslim_convert.impl.direct_save import (
    collect_saver_meta,
    main_staging_dir,
    merge_hf_staged_output,
    merge_staged_output,
    worker_staging_dir,
    write_result,
)
from msmodelslim.format.compressed_tensors_format.compressed_tensors import CompressedTensorsQuantFormat
from msmodelslim.processor.save.processor import QuantSaveProcessor
from msmodelslim.utils.exception import UnexpectedError


def _write_shard(dirpath, tensors):
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    fname = "quant_model_weights-00001-of-00001.safetensors"
    save_file(tensors, str(dirpath / fname))
    return fname, list(tensors.keys())


class TestMergeStagedOutput:
    def test_merge_renames_shards_and_cleans_staging(self, tmp_path):
        save_path = str(tmp_path / "out")
        worker0 = Path(save_path) / worker_staging_dir(save_path, 0)
        main_dir = Path(save_path) / main_staging_dir(save_path)
        f0, k0 = _write_shard(worker0, {"a.weight": torch.randn(2, 2)})
        fm, km = _write_shard(main_dir, {"embed.weight": torch.randn(2, 2)})

        merge_staged_output(
            save_path,
            str(tmp_path),
            [(worker_staging_dir(save_path, 0), {k0[0]: f0}, {k0[0]: "W8A8_MXFP8"})],
            ({km[0]: fm}, {km[0]: "FLOAT"}),
        )

        out = Path(save_path)
        shards = sorted(p.name for p in out.glob("quant_model_weights-*.safetensors"))
        assert len(shards) == 2
        with open(out / "quant_model_weights.safetensors.index.json", encoding="utf-8") as f:
            index = json.load(f)
        assert set(index["weight_map"].keys()) == {k0[0], km[0]}
        with open(out / "quant_model_description.json", encoding="utf-8") as f:
            desc = json.load(f)
        assert desc[k0[0]] == "W8A8_MXFP8"
        assert desc["model_quant_type"] == "W8A8_MXFP8"
        assert desc["group_size"] == 32
        assert not worker0.exists() and not main_dir.exists()

    def test_merge_infers_header_from_tensor_types(self, tmp_path):
        """文件级类型由张量描述推断，不写死 W8A8_MXFP8。"""
        save_path = str(tmp_path / "out")
        main_dir = Path(save_path) / main_staging_dir(save_path)
        fm, km = _write_shard(main_dir, {"embed.weight": torch.randn(2, 2)})

        merge_staged_output(
            save_path,
            str(tmp_path),
            [],
            ({km[0]: fm}, {km[0]: "FLOAT"}),
        )

        with open(Path(save_path) / "quant_model_description.json", encoding="utf-8") as f:
            desc = json.load(f)
        assert desc["model_quant_type"] == "Unknown"
        assert desc["group_size"] == 0


def _hf_saver(writer):
    saver = MagicMock(spec=QuantSaveProcessor)
    saver._format = MagicMock(spec=CompressedTensorsQuantFormat)
    saver._format.written_safetensors_writer.return_value = writer
    return saver


class TestCollectSaverMeta:
    def test_collect_reads_closed_writer_when_live_writer_none(self):
        """QuantSaveProcessor.finalize_export 会把 live writer 置空。"""
        closed = MagicMock()
        closed.saved_keys_map = {"model.norm.weight": "model-00001-of-00001.safetensors"}

        weight_map, desc_map = collect_saver_meta(_hf_saver(closed))

        assert weight_map == {"model.norm.weight": "model-00001-of-00001.safetensors"}
        assert not desc_map

    def test_collect_raise_when_writer_missing(self):
        with pytest.raises(UnexpectedError, match="no safetensors writer"):
            collect_saver_meta(_hf_saver(None))


class TestMergeHfStagedOutput:
    def test_merge_includes_main_passthrough_tensors(self, tmp_path):
        save_path = str(tmp_path / "out")
        worker0 = Path(save_path) / worker_staging_dir(save_path, 0)
        main_dir = Path(save_path) / main_staging_dir(save_path)
        f0, k0 = _write_shard(worker0, {"experts.0.weight": torch.randn(2, 2)})
        fm, km = _write_shard(main_dir, {"model.norm.weight": torch.ones(2)})
        (main_dir / "config.json").write_text('{"model_type": "kimi"}', encoding="utf-8")

        merge_hf_staged_output(
            save_path,
            str(tmp_path),
            [(worker_staging_dir(save_path, 0), {k0[0]: f0}, {})],
            ({km[0]: fm}, {}),
        )

        out = Path(save_path)
        with open(out / "model.safetensors.index.json", encoding="utf-8") as f:
            index = json.load(f)
        assert set(index["weight_map"].keys()) == {k0[0], km[0]}
        assert (out / "config.json").is_file()
        assert not worker0.exists() and not main_dir.exists()

    def test_merge_reads_staging_index_when_main_meta_empty(self, tmp_path):
        """post_run 置空 writer 后 collect 得到空 map 时，仍从 staging index 找回。"""
        save_path = str(tmp_path / "out")
        worker0 = Path(save_path) / worker_staging_dir(save_path, 0)
        main_dir = Path(save_path) / main_staging_dir(save_path)
        f0, k0 = _write_shard(worker0, {"experts.0.weight": torch.randn(2, 2)})
        fm, km = _write_shard(main_dir, {"model.norm.weight": torch.ones(2)})
        (main_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 8}, "weight_map": {km[0]: fm}}),
            encoding="utf-8",
        )

        merge_hf_staged_output(
            save_path,
            str(tmp_path),
            [(worker_staging_dir(save_path, 0), {k0[0]: f0}, {})],
            ({}, {}),
        )

        with open(Path(save_path) / "model.safetensors.index.json", encoding="utf-8") as f:
            index = json.load(f)
        assert set(index["weight_map"].keys()) == {k0[0], km[0]}

    def test_merge_hf_total_size_sums_staging_index_metadata(self, tmp_path):
        """直写 HF 的 index.total_size 应与非直写路径一样累加张量字节，而不是文件体积。"""
        save_path = str(tmp_path / "out")
        worker0 = Path(save_path) / worker_staging_dir(save_path, 0)
        main_dir = Path(save_path) / main_staging_dir(save_path)
        f0, k0 = _write_shard(worker0, {"experts.0.weight": torch.randn(2, 2)})
        fm, km = _write_shard(main_dir, {"model.norm.weight": torch.ones(2)})
        (worker0 / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 16}, "weight_map": {k0[0]: f0}}),
            encoding="utf-8",
        )
        (main_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 8}, "weight_map": {km[0]: fm}}),
            encoding="utf-8",
        )

        merge_hf_staged_output(
            save_path,
            str(tmp_path),
            [(worker_staging_dir(save_path, 0), {k0[0]: f0}, {})],
            ({km[0]: fm}, {}),
        )

        with open(Path(save_path) / "model.safetensors.index.json", encoding="utf-8") as f:
            index = json.load(f)
        assert index["metadata"]["total_size"] == 24
        assert set(index["weight_map"].keys()) == {k0[0], km[0]}


class TestWriteResult:
    def test_write_result_clears_processed_modules(self):
        saver = MagicMock()
        saver.processed_modules = {"held"}
        write_result(saver, "layers.0.q_proj", nn.Module())
        assert saver.processed_modules == set()
        saver.postprocess.assert_called_once()

    def test_write_result_release_format_refs_when_hf_saver(self):
        saver = _hf_saver(MagicMock())
        saver.processed_modules = set()
        write_result(saver, "layers.0.q_proj", nn.Module())
        saver._format.release_module_refs.assert_called_once()
