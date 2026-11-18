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

from ascend_utils.common.prune.transformer_prune.prune_utils_torch import PRUNE_STATE_DICT_FUNCS_TORCH
from ascend_utils.common.prune.transformer_prune.prune_utils_torch import PruneUtilsTorch
from ascend_utils.common.security import get_valid_read_path
from ascend_utils.common.security.pytorch import check_torch_module
from msmodelslim.common.prune.transformer_prune.prune_model import PruneConfig


def prune_model_weight_torch(model: nn.Module, config: PruneConfig, weight_file_path: str):
    check_torch_module(model)
    PruneConfig.check_prune_config(config, target_steps=list(PRUNE_STATE_DICT_FUNCS_TORCH.keys()))
    weight_file_path = get_valid_read_path(path=weight_file_path, extensions=["pt", "pth", "pkl", "bin"])

    state_dict = PruneUtilsTorch.get_state_dict(weight_file_path)
    for step_name in config.prune_state_dict_steps:
        state_dict = PRUNE_STATE_DICT_FUNCS_TORCH.get(step_name)(model, state_dict, config)
    model.load_state_dict(state_dict, strict=False)
