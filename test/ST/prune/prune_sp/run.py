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
import torchvision
from msmodelslim.pytorch.prune.prune_torch import PruneTorch

model = torchvision.models.vgg16(pretrained=False)
model.eval()

desc = PruneTorch(model, torch.ones([1, 3, 224, 224]).type(torch.float32)).prune(0.8)


def eval_func_l2(chn_weight):
    return torch.norm(chn_weight).item() / chn_weight.nelement()


desc = (
    PruneTorch(model, torch.ones([1, 3, 224, 224]).type(torch.float32))
    .set_importance_evaluation_function(eval_func_l2)
    .prune(0.8)
)

desc = PruneTorch(model, torch.ones([1, 3, 224, 224]).type(torch.float32)).set_node_reserved_ratio(0.5).prune(0.8)

left_params, desc = PruneTorch(model, torch.ones([1, 3, 224, 224]).type(torch.float32)).analysis()

left_params, desc = PruneTorch(model, torch.ones([1, 3, 224, 224])).analysis()
PruneTorch(model, torch.ones([1, 3, 224, 224])).prune_by_desc(desc)
