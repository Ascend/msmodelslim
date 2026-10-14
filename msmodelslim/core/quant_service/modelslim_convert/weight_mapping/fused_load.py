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

Fused MoE 逻辑张量加载（convert_design.md §6.3）。

预处理只注册逻辑 key；本模块在 lazy_init 时读取 ``meta.fused_from`` 指向的 3D fused 张量，
按 ``expert_id`` 与 ``projection`` 切片为 2D，避免在 catalog 中物化全部 expert 副本。
"""

from __future__ import annotations

import torch

from msmodelslim.core.convert.protocol import ICheckpointReader
from msmodelslim.core.quant_service.modelslim_convert.weight_mapping.fused_cache import FusedTensorCache


def _load_fused_tensor(reader: ICheckpointReader, fused_key: str, device: str) -> torch.Tensor:
    """加载 fused 源张量；reader 携带 ``fused_tensor_cache`` 时组内复用，避免重复读盘。"""

    def _loader() -> torch.Tensor:
        weight_map = reader.read_weight_map()
        shard = weight_map[fused_key]
        return reader.load_tensors({shard: [fused_key]}, device=device)[fused_key]

    cache = getattr(reader, "fused_tensor_cache", None)
    if not isinstance(cache, FusedTensorCache):
        return _loader()
    return cache.get_or_load(fused_key, device, _loader)


def load_logical_tensor(
    reader: ICheckpointReader,
    key: str,
    meta: dict,
    device: str = "cpu",
) -> torch.Tensor:
    """
    加载单个逻辑 tensor。

    Args:
        reader: checkpoint 读取器
        key: 逻辑 key（catalog 中的名字，可能不在 index 独立列出）
        meta: 须含 ``fused_from``、``expert_id``、``projection`` 等（由 split_fused_gate_up 写入）
        device: 加载设备
    """
    fused_key = meta.get("fused_from")
    if not fused_key:
        weight_map = reader.read_weight_map()
        shard = weight_map[key]
        return reader.load_tensors({shard: [key]}, device=device)[key]

    fused = _load_fused_tensor(reader, fused_key, device)

    expert_id = int(meta["expert_id"])
    projection = meta["projection"]
    split_dim = int(meta.get("split_dim", 1))
    parts = meta.get("chunk_parts", ["gate_proj", "up_proj"])

    expert_slice = fused[expert_id]
    chunks = torch.chunk(expert_slice, len(parts), dim=split_dim)
    return chunks[parts.index(projection)].contiguous()
