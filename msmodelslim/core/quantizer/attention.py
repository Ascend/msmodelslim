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
from pydantic import validate_call
from torch import nn

from msmodelslim.ir.qal import QStorage
from msmodelslim.ir import AutoFakeQuantDynamicCache
from .base import AutoActQuantizer, QConfig


class DynamicCacheQuantizer(nn.Module):
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(self, config: QConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.input_quantizer = AutoActQuantizer.from_config(config)

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def setup(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with QStorage.set_value_float_type(x.dtype):
            x = x.transpose(-2, -3)
            x_shape = x.shape
            x = x.reshape(-1, x.shape[-1] * x.shape[-2])
            dequant_x = self.input_quantizer(x)
            dequant_x = dequant_x.reshape(x_shape)
            dequant_x = dequant_x.transpose(-2, -3)
        return dequant_x

    def deploy(self):
        return AutoFakeQuantDynamicCache.create(
            self.input_quantizer.get_q_param(),
        )
