#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""direct_save：merge 合同与写后释放引用。"""

import json
from pathlib import Path
from unittest.mock import MagicMock

from safetensors.torch import save_file
from torch import nn
import torch

from msmodelslim.core.quant_service.modelslim_convert.impl.direct_save import (
    main_staging_dir,
    merge_staged_output,
    worker_staging_dir,
    write_result,
)


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


class TestWriteResult:
    def test_write_result_clears_processed_modules(self):
        saver = MagicMock()
        saver.processed_modules = {"held"}
        write_result(saver, "layers.0.q_proj", nn.Module())
        assert saver.processed_modules == set()
        saver.postprocess.assert_called_once()
