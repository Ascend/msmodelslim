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


import torch.nn as nn
import torch.nn.functional as F


class FakeLinearQuantizerOfW8A16OrW4A16(nn.Module):
    def __init__(self):
        super().__init__()
        self.deq_weight = None
        self.quant_bias = None
        self.device = None
        self.dtype = None

    # 传入linear、quant_weight、weight_offset、weight_scale参数的字典
    def set_param(self, linear: nn.Module = None, **quant_params):
        self.device = linear.weight.device
        self.dtype = linear.weight.dtype
        quant_weight = quant_params.get('quant_weight', None)
        if quant_weight is not None:
            quant_weight = quant_weight.to(self.device)
        weight_offset = quant_params.get('weight_offset', None)
        if weight_offset is not None:
            weight_offset = weight_offset.to(self.device)
        weight_scale = quant_params.get('weight_scale', None)
        if weight_scale is not None:
            weight_scale = weight_scale.to(self.device)
        ori_weight_shape = quant_weight.shape
        if len(ori_weight_shape) != 2:
            raise ValueError("original weight scale is not valid.")
        if len(weight_scale.shape) != 2:
            raise ValueError("Weight scale shape is not valid.")
        if weight_scale.shape[1] != 1:
            channel_num = ori_weight_shape[1]
            if weight_scale.shape[1] != 0:
                group_size = int(channel_num / weight_scale.shape[1])
            else:
                raise ZeroDivisionError("Weight scale shape[1] is 0, please check quant_params.")
            quant_weight = quant_weight.reshape(-1, group_size)
            weight_offset = weight_offset.reshape(-1, 1)
            weight_scale = weight_scale.reshape(-1, 1)
            deq_weight = (quant_weight - weight_offset) * weight_scale
            deq_weight = deq_weight.reshape(ori_weight_shape)
        else:
            deq_weight = (quant_weight - weight_offset) * weight_scale
        self.deq_weight = deq_weight.to(self.dtype)
        try:
            self.quant_bias = linear.bias.data.to(self.device)
        except AttributeError:
            self.quant_bias = None

    def forward(self, x):
        return F.linear(x, self.deq_weight, self.quant_bias)
