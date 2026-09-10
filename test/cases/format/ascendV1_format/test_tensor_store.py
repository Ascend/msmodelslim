# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You may use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------

msmodelslim/format/ascendV1_format/tensor_store.py 的单元测试。
"""

import json

import pytest
import torch
from safetensors.torch import save_file

from msmodelslim.format.ascendV1_format.tensor_store import (
    ASCENDV1_INDEX_NAME,
    ASCENDV1_SAFETENSORS_NAME,
    AscendV1TensorStore,
)
from msmodelslim.utils.exception import InvalidModelError, SchemaValidateError


def _write_single(tmp_path, tensors=None):
    tensors = tensors or {"a.b.weight": torch.ones(2, 3)}
    save_file(tensors, str(tmp_path / ASCENDV1_SAFETENSORS_NAME))
    return tmp_path


class TestAscendV1TensorStore:
    """对应 AscendV1TensorStore。"""

    def test_init_and_get_when_single_file(self, tmp_path):
        tensors = {"fc.weight": torch.arange(6).reshape(2, 3)}
        _write_single(tmp_path, tensors)
        store = AscendV1TensorStore(str(tmp_path))

        assert store.has("fc.weight") is True
        assert store.has("missing") is False
        assert torch.equal(store.get("fc.weight"), tensors["fc.weight"])
        assert store.get_shape("fc.weight") == (2, 3)

    def test_init_resolves_weight_map_when_index_file(self, tmp_path):
        shard = tmp_path / "weights-00001-of-00001.safetensors"
        tensors = {"w": torch.zeros(2)}
        save_file(tensors, str(shard))
        index = {"weight_map": {"w": shard.name}}
        (tmp_path / ASCENDV1_INDEX_NAME).write_text(json.dumps(index), encoding="utf-8")

        store = AscendV1TensorStore(str(tmp_path))

        assert store.has("w") is True
        assert store.has("other") is False
        assert torch.equal(store.get("w"), tensors["w"])
        assert store.get_shape("w") == (2,)

    def test_init_raises_when_weights_missing(self, tmp_path):
        with pytest.raises(InvalidModelError):
            AscendV1TensorStore(str(tmp_path))

    def test_init_raises_when_index_malformed(self, tmp_path):
        (tmp_path / ASCENDV1_INDEX_NAME).write_text(json.dumps({"weight_map": []}), encoding="utf-8")

        with pytest.raises(InvalidModelError):
            AscendV1TensorStore(str(tmp_path))

    def test_get_raises_when_key_not_in_file(self, tmp_path):
        _write_single(tmp_path)
        store = AscendV1TensorStore(str(tmp_path))

        with pytest.raises(SchemaValidateError):
            store.get("nope")

    def test_get_raises_when_key_not_in_weight_map(self, tmp_path):
        (tmp_path / ASCENDV1_INDEX_NAME).write_text(
            json.dumps({"weight_map": {"w": "w.safetensors"}}), encoding="utf-8"
        )
        store = AscendV1TensorStore(str(tmp_path))

        with pytest.raises(SchemaValidateError):
            store.get("missing")
