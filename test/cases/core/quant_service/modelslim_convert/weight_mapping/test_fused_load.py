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

msmodelslim.core.quant_service.modelslim_convert.weight_mapping.fused_load 模块的单元测试
"""

from unittest.mock import MagicMock

import torch

from msmodelslim.core.quant_service.modelslim_convert.weight_mapping.fused_cache import FusedTensorCache
from msmodelslim.core.quant_service.modelslim_convert.weight_mapping.fused_load import load_logical_tensor


class TestLoadLogicalTensor:
    """测试 load_logical_tensor 函数"""

    def test_load_logical_tensor_slice_expert_when_fused_meta_given(self):
        reader = MagicMock()
        reader.read_weight_map.return_value = {"fused": "s0"}
        fused = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
        reader.load_tensors.return_value = {"fused": fused}

        meta = {
            "fused_from": "fused",
            "expert_id": 1,
            "projection": "up_proj",
            "split_dim": 1,
            "chunk_parts": ["gate_proj", "up_proj"],
        }
        out = load_logical_tensor(reader, "logical.key", meta, device="cpu")
        assert out.shape == (2, 2)
        assert torch.equal(out, fused[1].chunk(2, dim=1)[1])

    def test_load_logical_tensor_reuse_fused_cache_when_reader_has_cache(self):
        reader = MagicMock()
        reader.read_weight_map.return_value = {"fused": "s0"}
        fused = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
        reader.load_tensors.return_value = {"fused": fused}
        reader.fused_tensor_cache = FusedTensorCache()

        meta = {
            "fused_from": "fused",
            "expert_id": 0,
            "projection": "gate_proj",
            "split_dim": 1,
            "chunk_parts": ["gate_proj", "up_proj"],
        }
        load_logical_tensor(reader, "logical.a", meta, device="cpu")
        load_logical_tensor(
            reader,
            "logical.b",
            {**meta, "expert_id": 1, "projection": "up_proj"},
            device="cpu",
        )
        assert reader.load_tensors.call_count == 1

    def test_load_logical_tensor_load_direct_when_no_fused_from(self):
        reader = MagicMock()
        reader.read_weight_map.return_value = {"w": "s0"}
        reader.load_tensors.return_value = {"w": torch.ones(2, 2)}
        out = load_logical_tensor(reader, "w", {}, device="cpu")
        assert out.shape == (2, 2)
