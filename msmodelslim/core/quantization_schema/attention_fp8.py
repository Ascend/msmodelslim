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

from msmodelslim.core.quantization_schema.api import fake_quantize
from msmodelslim.core.quantization_schema.auto import AutoFakeQuantDynamicCache
from msmodelslim.core.quantization_schema.qal.qregistry import QABCRegistry
from msmodelslim.core.quantization_schema.qal.qbase import QDType, QParam, QScheme, QScope, QStorage

from msmodelslim.core.quantization_schema.const import (
    fp8_e4m3_per_channel_sym,
    fp8_e4m3_per_head_sym,
)


@QABCRegistry.multi_register(
    dispatch_key=[
        fp8_e4m3_per_channel_sym,
    ],
    abc_type=AutoFakeQuantDynamicCache,
)
class FakeQuantCacheFP8PerChannel(AutoFakeQuantDynamicCache):
    """
    FP8 per-channel static KV cache.

    Per-channel means along the last dimension (head_dim).
    Each channel uses a pre-computed FP8 scale, shared across seq_len.
    """

    def __init__(self, x_q_param: QParam):
        super().__init__()
        self.x_q_scheme = x_q_param.scheme

        ext = x_q_param.ext or {}
        scale_data = ext.get("scale")
        if scale_data is None:
            class_name = self.__class__.__name__
            raise ValueError(f"`scale` is needed in ext but is missing for {class_name}")
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
        x = x.transpose(-2, -3)
        x_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        x_q_param = QParam(scheme=self.x_q_scheme, ext={"scale": self.kv_cache_scale})
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x), x_q_param).value
        x_q_dq = x_q_dq.reshape(x_shape)
        x_q_dq = x_q_dq.transpose(-2, -3).to(x.dtype)
        return x_q_dq


@QABCRegistry.multi_register(dispatch_key=[fp8_e4m3_per_head_sym], abc_type=AutoFakeQuantDynamicCache)
class FakeQuantCacheFP8PerHead(AutoFakeQuantDynamicCache):
    def __init__(self, x_q_param: QParam):
        super().__init__()
        self.x_q_scheme = x_q_param.scheme

        scale_data = x_q_param.ext.get("scale")
        if scale_data is None:
            class_name = self.__class__.__name__
            raise ValueError(f"`scale` is needed in ext but is missing for {class_name}")

        self.kv_cache_scale = nn.Parameter(scale_data, requires_grad=False)
        offset_data = x_q_param.ext.get("offset")
        if offset_data is None:
            self.kv_cache_offset = None
        elif not isinstance(offset_data, torch.Tensor):
            self.kv_cache_offset = nn.Parameter(torch.tensor(offset_data, dtype=torch.float32), requires_grad=False)
        else:
            self.kv_cache_offset = nn.Parameter(offset_data.to(dtype=torch.float32), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_perm = x.movedim(1, -1)
        last_dim = x_perm.shape[-1]
        x_2d = x_perm.reshape(-1, last_dim)

        per_channel_scheme = QScheme(scope=QScope.PER_CHANNEL, dtype=self.x_q_scheme.dtype, symmetric=True)
        ext = {"scale": self.kv_cache_scale.to(x.device, non_blocking=True).data}

        x_q_param = QParam(scheme=per_channel_scheme, ext=ext)
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x_2d), x_q_param)

        x_out = x_q_dq.value.reshape(x_perm.shape)
        x_out = x_out.movedim(-1, 1)
        return x_out
