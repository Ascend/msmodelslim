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

import os

from copy import deepcopy
from torchvision import models

from ascend_utils.common.utils import count_parameters
from msmodelslim import logger as msmodelslim_logger

from msmodelslim.mindspore import low_rank_decompose as lrd_ms
from msmodelslim.pytorch import low_rank_decompose as lrd_pt
from sample_net_mindspore import LrdSampleNetwork


msmodelslim_logger.info("==== 1. Test PyTorch ====")

# load model
resnet50 = models.resnet50()
model_param = count_parameters(resnet50)
msmodelslim_logger.info("[PyTorch] Original model parameters: %s", model_param)

# instantialize decomposer
decomposer = lrd_pt.Decompose(resnet50).from_ratio(0.5)
new_resnet50 = decomposer.decompose_network()
new_model_param = count_parameters(new_resnet50)
msmodelslim_logger.info("[PyTorch] After decomposition, model parameters: %s", new_model_param)

decomposer = lrd_pt.Decompose(resnet50).from_fixed(64, divisor=16)
new_resnet50 = decomposer.decompose_network()
new_model_param = count_parameters(new_resnet50)
msmodelslim_logger.info("[PyTorch] After decomposition, model parameters: %s", new_model_param)

decomposer = lrd_pt.Decompose(resnet50).from_dict({"fc": 0.5, ".*.conv1": 64, ".*.conv2": 128, ".*.conv3": "vbmf"})
new_resnet50 = decomposer.decompose_network()
new_model_param = count_parameters(new_resnet50)
msmodelslim_logger.info("[PyTorch] After decomposition, model parameters: %s", new_model_param)

decomposer = lrd_pt.Decompose(resnet50).from_vbmf(divisor=16)
new_resnet50 = decomposer.decompose_network()
new_model_param = count_parameters(new_resnet50)
msmodelslim_logger.info("[PyTorch] After decomposition, model parameters: %s", new_model_param)

config_file = f"{os.environ['PROJECT_PATH']}/resource/lowrank/torch_resnet50_low_rank_decompose_from_ratio_0.5.json"
decomposer = lrd_pt.Decompose(resnet50, config_file=config_file).from_ratio(0.5, excludes=["fc"])
new_resnet50 = decomposer.decompose_network(do_decompose_weight=False)
new_model_param = count_parameters(new_resnet50)
msmodelslim_logger.info("[PyTorch] After decomposition, model parameters: %s", new_model_param)

decomposer = lrd_pt.Decompose(resnet50, config_file=config_file).from_file()
new_resnet50 = decomposer.decompose_network(do_decompose_weight=True)
new_model_param = count_parameters(new_resnet50)
msmodelslim_logger.info("[PyTorch] After decomposition, model parameters: %s", new_model_param)

msmodelslim_logger.info("==== 2. Test MindSpore ====")

lrd_model = LrdSampleNetwork()
lrd_model_param = count_parameters(lrd_model)
msmodelslim_logger.info("[MindSpore] Origin model parameters: %s", lrd_model_param)

decomposer_ms = lrd_ms.Decompose(deepcopy(lrd_model)).from_ratio(0.5)
new_lrd_model = decomposer_ms.decompose_network(do_decompose_weight=False)
new_lrd_model_param = count_parameters(new_lrd_model)
msmodelslim_logger.info("[MindSpore] After decomposition, model parameters: %s", new_lrd_model_param)

decomposer_ms = lrd_ms.Decompose(deepcopy(lrd_model)).from_ratio(0.5)
new_lrd_model = decomposer_ms.decompose_network()
new_lrd_model_param = count_parameters(new_lrd_model)
msmodelslim_logger.info("[MindSpore] After decomposition, model parameters: %s", new_lrd_model_param)

decomposer_ms = lrd_ms.Decompose(deepcopy(lrd_model)).from_fixed(64, divisor=16)
new_lrd_model = decomposer_ms.decompose_network()
new_lrd_model_param = count_parameters(new_lrd_model)
msmodelslim_logger.info("[MindSpore] After decomposition, model parameters: %s", new_lrd_model_param)

decomposer = lrd_ms.Decompose(deepcopy(lrd_model)).from_dict(
    {"classifier.0": 0.5, "feature.*": 64, "embedding.*": "vbmf"},
    excludes=["classifier.1"],
)
new_lrd_model = decomposer.decompose_network()
new_lrd_model_param = count_parameters(new_lrd_model)
msmodelslim_logger.info("[MindSpore] After decomposition, model parameters: %s", new_lrd_model_param)

decomposer_ms = lrd_ms.Decompose(deepcopy(lrd_model)).from_vbmf(divisor=16)
new_lrd_model = decomposer_ms.decompose_network()
new_lrd_model_param = count_parameters(new_lrd_model)
msmodelslim_logger.info("[MindSpore] After decomposition, model parameters: %s", new_lrd_model_param)

config_file = f"{os.environ['PROJECT_PATH']}/resource/lowrank/ms_resnet50_low_rank_decompose_from_ratio_0.5.json"
decomposer_ms = lrd_ms.Decompose(deepcopy(lrd_model), config_file=config_file).from_ratio(
    0.5, excludes=["classifier.0", "classifier.1"]
)
new_lrd_model = decomposer_ms.decompose_network(do_decompose_weight=False)
new_lrd_model_param = count_parameters(new_lrd_model)
msmodelslim_logger.info("[MindSpore] After decomposition, model parameters: %s", new_lrd_model_param)

decomposer = lrd_ms.Decompose(deepcopy(lrd_model), config_file=config_file).from_file()
new_resnet50 = decomposer.decompose_network(do_decompose_weight=True)
new_model_param = count_parameters(new_resnet50)
msmodelslim_logger.info("[MindSpore] After decomposition, model parameters: %s", new_model_param)
