#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs import linear_quantization_params


class FakeLinearQuantizerOfW8A8(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = None
        self.dtype = None
        self.device = None
        self.deq_scale = None
        self.weight = None
        self.input_offset = None
        self.input_scale = None
        self.weight_scale = None
        self.is_dynamic = False
        self.x_min = None
        self.x_max = None
        self.bit = 8

    def int64tofp32(self):
        tensor = np.frombuffer(self.deq_scale.numpy().astype(np.int32).tobytes(), dtype=np.float32)
        return torch.from_numpy(tensor)

    def deqscale_process(self):
        if self.input_scale == 0:
            raise ValueError("Input scale cannot be zero.")
        scale = self.deq_scale.to(torch.float32) / self.input_scale.to(torch.float32)
        return scale

    # 传入deq_scale、input_offset、input_scale、quant_weight参数的字典
    def set_param(self, linear: nn.Module = None, **quant_params):
        self.device = linear.weight.device
        self.dtype = linear.weight.dtype
        self.is_dynamic = quant_params.get('is_dynamic', None)
        if not self.is_dynamic:
            self.deq_scale = quant_params.get('deq_scale', None)
            if self.deq_scale is not None:
                self.deq_scale = self.int64tofp32().to(self.device)
            self.input_offset = quant_params.get('input_offset', None)
            if self.input_offset is not None:
                self.input_offset = self.input_offset.to(self.device)
            self.input_scale = quant_params.get('input_scale', None)
            if self.input_scale is not None:
                self.input_scale = self.input_scale.to(self.device)
            self.weight_scale = self.deqscale_process().to(self.device).view(-1, 1)
        else:
            self.weight_scale = quant_params.get('weight_scale', None)
            if self.weight_scale is not None:
                self.weight_scale = self.weight_scale.to(self.device)
        self.weight = quant_params.get('quant_weight', None)
        if self.weight is not None:
            self.weight = (self.weight.to(self.device) * self.weight_scale).to(self.dtype)
        try:
            self.bias = linear.bias.data.to(self.device)
        except AttributeError:
            self.bias = None

    def forward(self, x):
        if self.is_dynamic:
            self.x_min = x.min(2)[0]
            self.x_max = x.max(2)[0]
            self.x_min = self.x_min.view(-1, self.x_min.shape[1], 1)
            self.x_max = self.x_max.view(-1, self.x_max.shape[1], 1)
            self.input_scale, self.input_offset = linear_quantization_params(
                self.bit, self.x_min, self.x_max, q_signed=True, sym=False,
            )
        x = x / self.input_scale + self.input_offset
        n = 2 ** (8 - 1)
        x = x.round()
        x = torch.clamp(x, -n, n - 1)
        deq_tensor = (x - self.input_offset) * self.input_scale
        return F.linear(deq_tensor, self.weight, self.bias).to(x.device)