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

import torch
from torch import nn

from msmodelslim.core.quantization_schema.auto import AutoFakeQuantDynamicCache
from msmodelslim.core.quantization_schema.qal.qregistry import QABCRegistry
from msmodelslim.core.quantization_schema.qal.qbase import QParam
from msmodelslim.utils.logging import logger_setter

from msmodelslim.core.quantization_schema.const import (
    int8_per_token_sym,
    int8_per_tile_sym,
    fp8_e4m3_per_token_sym,
    fp8_e4m3_per_tile_sym,
    mxfp8_per_block_sym,
    mxfp8_per_token_sym,
    mxfp8_per_tile_sym,
    mxfp4_per_block_sym,
    hif4_per_block_sym,
    fp4_e2m1_per_token_sym,
    fp4_e2m1_per_block_sym,
)


@QABCRegistry.multi_register(
    dispatch_key=[
        # int8 dynamic
        int8_per_token_sym,
        int8_per_tile_sym,
        # fp8 dynamic
        fp8_e4m3_per_token_sym,
        fp8_e4m3_per_tile_sym,
        # fp4 dynamic
        fp4_e2m1_per_token_sym,
        fp4_e2m1_per_block_sym,
        # mxfp8 dynamic
        mxfp8_per_block_sym,
        mxfp8_per_token_sym,
        mxfp8_per_tile_sym,
        # mxfp4 dynamic
        mxfp4_per_block_sym,
        # hif4 dynamic
        hif4_per_block_sym,
    ],
    abc_type=AutoFakeQuantDynamicCache,
)
@logger_setter()
class FakeQuantDynamicCache(AutoFakeQuantDynamicCache):
    """
    Unified dynamic KV cache IR.

    All dynamic schemes (per-token/per-tile/per-block) share this single IR class.
    The forward is a no-op (returns x unchanged); only x_q_scheme is stored for
    the saver to read the quantization configuration at export time.
    """

    def __init__(self, x_q_param: QParam):
        super().__init__()
        self.x_q_scheme = x_q_param.scheme

        ext = x_q_param.ext or {}
        scale_data = ext.get("scale")
        if scale_data is None:
            scale_data = torch.ones(1, dtype=torch.float32)
        elif not isinstance(scale_data, torch.Tensor):
            scale_data = torch.tensor(scale_data, dtype=torch.float32)
        else:
            scale_data = scale_data.to(dtype=torch.float32)

        self.kv_cache_scale = nn.Parameter(scale_data, requires_grad=False)
        offset_data = ext.get("offset")
        if offset_data is None:
            self.kv_cache_offset = None
        elif not isinstance(offset_data, torch.Tensor):
            self.kv_cache_offset = nn.Parameter(torch.tensor(offset_data, dtype=torch.float32), requires_grad=False)
        else:
            self.kv_cache_offset = nn.Parameter(offset_data.to(dtype=torch.float32), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
