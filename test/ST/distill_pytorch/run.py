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
# pylint: disable=super-with-arguments,useless-parent-delegation

# 导入相关依赖
import torch

from msmodelslim import set_logger_level
from msmodelslim import logger as msmodelslim_logger  # noqa: F401
from msmodelslim.common.knowledge_distill.knowledge_distill import KnowledgeDistillConfig
from msmodelslim.common.knowledge_distill.knowledge_distill import get_distill_model


class TorchTeacherModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.teacher_fc = torch.nn.Linear(1, 1)

    def forward(self, inputs):
        output = self.teacher_fc(inputs)
        return output


class TorchStudentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.student_fc = torch.nn.Linear(1, 1)

    def forward(self, inputs):
        output = self.student_fc(inputs)
        return output


class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target):
        # calculate custom loss
        loss = torch.mean(torch.abs(output - target))
        return loss


# 根据实际情况配置
set_logger_level("info")

# verify custom
distill_config = KnowledgeDistillConfig()
distill_config.set_hard_label(0.5, 0).add_custom_loss_func("custom_loss", CustomLoss())

distill_model = get_distill_model(TorchTeacherModel(), TorchStudentModel(), distill_config)
distill_model(torch.randn((1, 1)))
student_model = distill_model.get_student_model()

# verify t_module s_module
distill_config = KnowledgeDistillConfig()
distill_config.set_hard_label(0.5, 0).add_inter_soft_label(
    {
        "t_module": "teacher_fc",
        "s_module": "student_fc",
        "t_output_idx": 0,
        "s_output_idx": 0,
        "loss_func": [{"func_name": "KDCrossEntropy", "func_weight": 1, "temperature": 1}],
    }
)

distill_model = get_distill_model(TorchTeacherModel(), TorchStudentModel(), distill_config)
distill_model(torch.randn((1, 1)))
student_model = distill_model.get_student_model()

# verify not module
distill_config = KnowledgeDistillConfig()
distill_config.set_hard_label(0.5, 0).add_inter_soft_label(
    {"t_output_idx": 0, "s_output_idx": 0, "loss_func": [{"func_name": "KDCrossEntropy", "func_weight": 1}]}
)

distill_model = get_distill_model(TorchTeacherModel(), TorchStudentModel(), distill_config)
distill_model(torch.randn((1, 1)))
student_model = distill_model.get_student_model()
