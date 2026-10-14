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
# pylint: disable=super-with-arguments

import logging

from mindspore.nn import Cell

from ascend_utils.mindspore.knowledge_distill.distill_losses_manager_ms import DistillLossesManager


class DistillDualModelsMs(Cell):
    def __init__(self, config, student_model, teacher_model, output_shapes=None):
        super(DistillDualModelsMs, self).__init__()

        self.student_model = student_model
        self.teacher_model = teacher_model
        self.hard_label_loss_weight = config.hard_label_loss_weight
        self.distill_loss = 0

        self.distill_losses_manager = DistillLossesManager(
            config, self.student_model, self.teacher_model, output_shapes
        )

        logging.info("DistillDualModels inited.")

    def construct(self, *args, **kwargs):
        """
        Calculate distillation loss
        data: tuple type
        """
        t_outputs = self._construct_teacher(*args, **kwargs)
        s_outputs = self._construct_student(*args, **kwargs)

        distill_loss, _ = self.distill_losses_manager.construct(s_outputs, t_outputs)
        return distill_loss

    def set_train_state(self, is_teacher_train=False):
        self.teacher_model.set_train(is_teacher_train)
        self.student_model.set_train(True)
        if not is_teacher_train:
            for param in self.teacher_model.get_parameters():
                param.requires_grad = False

    def get_student_model(self):
        """
        Please use this method after training.
        Cannot train the model of knowledge distillation again after using this method.
        """
        self.distill_losses_manager.restore_modules(self.student_model, self.teacher_model)
        return self.student_model

    def _construct_teacher(self, *args, **kwargs):
        output = self.teacher_model(*args, **kwargs)
        return output

    def _construct_student(self, *args, **kwargs):
        output = self.student_model(*args, **kwargs)
        return output
