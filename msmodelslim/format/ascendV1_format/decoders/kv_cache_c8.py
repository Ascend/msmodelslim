#!/usr/bin/env python
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

AscendV1 KV-cache (C8) decoder. Inverse of AscendV1Saver.on_dynamic_cache.

  disk : {k_proj}.kv_cache_scale / .kv_cache_offset (C8 label)
  IR   : FakeQuantDynamicCache (int8_per_channel_sym / int8_per_channel_asym)
"""

import torch

from msmodelslim.format.common.ir_builder.kv_cache import KvCacheParamSet, KvCacheStructureSpec
from msmodelslim.ir.attention import FakeQuantDynamicCache
from msmodelslim.ir.const import int8_per_channel_sym

from .base import KvCacheIrDecoder


class KvCacheC8Decoder(KvCacheIrDecoder):
    """Decode AscendV1 C8 KV-cache tensors into IR-semantic DTOs."""

    label = "C8"
    ir_type = FakeQuantDynamicCache

    def structure(self, prefix: str, group_size: int = 0) -> KvCacheStructureSpec:
        del group_size
        store = self.store
        scale_key = f"{prefix}.kv_cache_scale"
        offset_key = f"{prefix}.kv_cache_offset"
        if not store.has(scale_key) or not store.has(offset_key):
            from msmodelslim.utils.exception import SchemaValidateError

            raise SchemaValidateError(
                f"Missing C8 KV-cache tensors at '{prefix}'",
                action="Please ensure the export was produced by AscendV1Saver.on_dynamic_cache.",
            )
        return KvCacheStructureSpec(
            scale_shape=store.get_shape(scale_key),
            offset_shape=store.get_shape(offset_key),
            scheme=int8_per_channel_sym,
        )

    def params(self, prefix: str, device: torch.device, group_size: int = 0) -> KvCacheParamSet:
        del group_size
        store = self.store
        scale = store.get(f"{prefix}.kv_cache_scale").to(device=device, dtype=torch.float32)
        offset = store.get(f"{prefix}.kv_cache_offset").to(device=device, dtype=torch.float32)
        return KvCacheParamSet(
            kv_cache_scale=scale,
            kv_cache_offset=offset,
            scheme=int8_per_channel_sym,
        )
