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


from torch.nn.modules import Module

from ascend_utils.pytorch.knowledge_distill.distill_losses_func_torch import DISTILL_LOSS_FUNC_TORCH
from ascend_utils.common.security.pytorch import check_torch_module
from ascend_utils.pytorch.knowledge_distill.distill_dual_models_torch import DistillDualModelsTorch
from msmodelslim.common.knowledge_distill.knowledge_distill import KnowledgeDistillConfig
from msmodelslim import logger


def get_distill_model_torch(teacher: Module, student: Module, config: KnowledgeDistillConfig):
    """
    Build a model for knowledge distillation that contains teacher, student, and loss functions.
    And you can get fine-tuned student model from this model after training.

    Args:
        teacher(nn.Module): teacher model.
        student(nn.Module): student model.
        config(KnowledgeDistillConfig): Configuration for knowledge distillation.

    Returns:
        a model contains teacher and student.
    """
    logger.info("================ Start build distill model ===============")

    check_torch_module(teacher)
    check_torch_module(student)
    KnowledgeDistillConfig.check_config(config, DISTILL_LOSS_FUNC_TORCH, Module, is_mindspore=False)
    KnowledgeDistillConfig.generate_loss_instance(config, DISTILL_LOSS_FUNC_TORCH, Module)

    distill_dual_models = DistillDualModelsTorch(config, student, teacher)
    distill_dual_models.set_train_state(config.train_teacher)

    logger.info("================ Finish build distill model ===============")
    return distill_dual_models
