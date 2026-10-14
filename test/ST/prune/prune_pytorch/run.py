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

import os
import torch
from msmodelslim.common.prune.transformer_prune.prune_model import PruneConfig
from msmodelslim.common.prune.transformer_prune.prune_model import prune_model_weight
from msmodelslim import set_logger_level


class TorchOriModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 2)
        self.fc2 = torch.nn.Linear(2, 2)

    def forward(self, inputs):
        output = self.fc1(inputs)
        output = self.fc2(output)
        return output


weight_file_path = f"{os.environ['PROJECT_PATH']}/resource/prune/torch_model_weights.pth"
torch_ori_model = TorchOriModel()
torch.save(torch_ori_model.state_dict(), weight_file_path)
set_logger_level("info")  # 根据实际情况配置
config = PruneConfig()
config.set_steps(['prune_blocks', 'prune_bert_intra_block'])
config.add_blocks_params(r'fc(\d+)', {1: 2})
prune_model_weight(
    TorchOriModel(), config, weight_file_path
)  # model根据实际情况配置待剪枝模型实例，weight_file_path根据实际情况配置原模型的权重文件
