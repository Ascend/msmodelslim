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
# pylint: disable=consider-using-from-import

import torch
import torch.nn as nn

from msmodelslim.ir.api import fake_quantize
from msmodelslim.ir.qal import QABCRegistry, QDType, QParam, QStorage
from msmodelslim.utils.logging import logger_setter
from .auto import AutoFakeQuantDynamicCache
from .const import int8_per_channel_sym, int8_per_channel_asym


@QABCRegistry.multi_register(
    dispatch_key=[
        int8_per_channel_sym,
        int8_per_channel_asym,
    ],
    abc_type=AutoFakeQuantDynamicCache,
)
@logger_setter()
class FakeQuantDynamicCache(AutoFakeQuantDynamicCache):
    """
    动态缓存伪量化IR。

    动态缓存伪量化方式可以用以下参数描述：
        kv_cache_scale: KV缓存的量化参数，类型为torch.Tensor, dtype为torch.float32
        kv_cache_offset: KV缓存的量化参数，类型为torch.Tensor, dtype为torch.int32
    """

    def __init__(
        self,
        x_q_param: QParam,
    ):
        super().__init__()

        self.x_q_scheme = x_q_param.scheme
        self.kv_cache_scale = nn.Parameter(x_q_param.ext.get("scale"), requires_grad=False)
        self.kv_cache_offset = nn.Parameter(x_q_param.ext.get("offset"), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(-2, -3)
        x_shape = x.shape
        x = x.reshape(-1, x.shape[-1] * x.shape[-2])
        x_q_param = QParam(scheme=self.x_q_scheme, ext={"scale": self.kv_cache_scale, "offset": self.kv_cache_offset})
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x), x_q_param).value
        x_q_dq = x_q_dq.reshape(x_shape)
        x_q_dq = x_q_dq.transpose(-2, -3).to(x.dtype)
        return x_q_dq
