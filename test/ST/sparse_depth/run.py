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
# pylint: disable=invalid-envvar-default

import os
import apex
import torch
import torch_npu  # noqa: F401
from apex import amp
from torch import nn

from ascend_utils.common.utils import count_parameters
from msmodelslim.pytorch import sparse

device = torch.device("npu:{}".format(os.getenv('DEVICE_ID', 0)))
torch.npu.set_device(device)
model = nn.Sequential(
    nn.Conv2d(3, 32, 1, 1, bias=False),
    nn.Sequential(nn.Conv2d(32, 64, 1, 1, bias=False), nn.BatchNorm2d(64), nn.Conv2d(64, 32, 1, 1, bias=False)),
    nn.Sequential(nn.Conv2d(32, 64, 1, 1, bias=False), nn.BatchNorm2d(64), nn.Conv2d(64, 32, 1, 1, bias=False)),
    nn.Sequential(nn.Conv2d(32, 64, 1, 1, bias=False), nn.BatchNorm2d(64), nn.Conv2d(64, 32, 1, 1, bias=False)),
    nn.Sequential(nn.Conv2d(32, 64, 1, 1, bias=False), nn.BatchNorm2d(64), nn.Conv2d(64, 32, 1, 1, bias=False)),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(32, 10, bias=False),
).to(device)

optimizer = apex.optimizers.NpuFusedSGD(model.parameters(), lr=0.1)
steps_per_epoch, epochs_each_stage = 10, [2, 3, 1]
oring_model_params = count_parameters(model)  # 10826
model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O2", combine_grad=False)

# 添加宽度稀疏化训练方式
model = sparse.sparse_model_depth(
    model, optimizer, steps_per_epoch=steps_per_epoch, epochs_each_stage=epochs_each_stage
)

# 模型训练
for _ in range(steps_per_epoch * sum(epochs_each_stage)):
    optimizer.zero_grad()
    output = model(torch.ones([1, 3, 32, 32]).npu())
    loss = torch.mean(output)
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    optimizer.step()
