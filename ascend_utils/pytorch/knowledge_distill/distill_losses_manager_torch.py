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
from torch.nn.modules import Module


class DistillLossesManager(Module):
    def __init__(self, config, s_model, t_model):
        super().__init__()
        self.hard_label_loss_weight = config.hard_label_loss_weight
        self.model_parallel = config.model_parallel

        self.s_name2module = dict(s_model.named_modules())
        self.t_name2module = dict(t_model.named_modules())
        self.s_outputs = dict()
        self.t_outputs = dict()
        self.s_module2name = dict()
        self.t_module2name = dict()

        self.inter_matches = config.inter_matches
        self.output_matches = config.output_matches

        self.register_output_hook(s_model, t_model)

    @staticmethod
    def index_able(data):
        if isinstance(data, (list, tuple)):
            return data
        else:
            return [data]

    def register_output_hook(self, s_model, t_model):
        self.s_name2module = dict(s_model.named_modules())
        self.t_name2module = dict(t_model.named_modules())

        for name, module in s_model.named_modules():
            self.s_module2name[module] = name
        for name, module in t_model.named_modules():
            self.t_module2name[module] = name

        for match in self.inter_matches:
            t_module = self.t_name2module.get(match["t_module"])
            s_module = self.s_name2module.get(match["s_module"])

            if t_module is None:
                raise ValueError(
                    "The module \"{}\" not in teacher model. Please check config.".format(match["t_module"])
                )
            if s_module is None:
                raise ValueError(
                    "The module \"{}\" not in student model. Please check config.".format(match["s_module"])
                )

            t_module.register_forward_hook(self.teacher_forward_output_hook)
            s_module.register_forward_hook(self.student_forward_output_hook)

    def teacher_forward_output_hook(self, module, inputs, outputs):
        self.t_outputs[self.t_module2name.get(module)] = outputs

    def student_forward_output_hook(self, module, inputs, outputs):
        self.s_outputs[self.s_module2name.get(module)] = outputs

    def forward_teacher(self, t_model, batch):
        """
        data: dict, model input data
        """
        self.t_outputs = {kk: None for kk in self.t_outputs.keys()}
        with torch.no_grad():
            if isinstance(batch, dict):
                output = t_model(**batch)
            elif isinstance(batch, tuple):
                output = t_model(*batch)
            else:
                raise NotImplementedError('teacher model forward input only support tuple, dict type')

        return output

    def forward_student(self, s_model, batch):
        """
        data: dict, model input data
        """
        self.s_outputs = {kk: None for kk in self.s_outputs.keys()}
        if isinstance(batch, dict):
            output = s_model(**batch)
        elif isinstance(batch, tuple):
            output = s_model(*batch)
        else:
            raise NotImplementedError('student model forward input only support tuple, dict type')
        return output

    def compute_loss_pt(self, s_outputs, t_outputs):
        unweighted_losses = {}
        # compute inter loss
        inter_loss, inter_unweighted_losses = self.compute_inter_loss()
        unweighted_losses.update(inter_unweighted_losses)

        # compute output loss
        output_loss, output_unweighted_losses = self.compute_output_loss(s_outputs, t_outputs)
        unweighted_losses.update(output_unweighted_losses)

        loss = inter_loss + output_loss
        return loss, unweighted_losses

    def compute_output_loss(self, s_outputs, t_outputs):
        distill_loss = 0
        unweighted_losses = {}

        for match_ix, match in enumerate(self.output_matches):
            s_output_idx = match["s_output_idx"]
            t_output_idx = match["t_output_idx"]

            if not 0 <= s_output_idx < len(s_outputs):
                raise ValueError(
                    "s_output_idx in {} out of range. Please check add_output_soft_label config.".format(match)
                )
            if not 0 <= t_output_idx < len(t_outputs):
                raise ValueError(
                    "t_output_idx in {} out of range. Please check add_output_soft_label config.".format(match)
                )

            s_output = s_outputs[s_output_idx]
            t_output = t_outputs[t_output_idx]

            for loss_module in match["loss_func"]:
                loss_weight = loss_module["func_weight"]
                loss_func = loss_module["func_instance"]
                loss_temperature = loss_module["temperature"]
                if loss_temperature:
                    loss = loss_func(s_output, t_output, loss_temperature)
                else:
                    loss = loss_func(s_output, t_output)

                loss_name = f'output_match{match_ix}_{loss_module["func_name"]}_loss'
                unweighted_losses[loss_name] = loss.item()
                distill_loss += loss * loss_weight

        return distill_loss, unweighted_losses

    def compute_inter_loss(self):
        distill_loss = 0
        unweighted_losses = {}

        for match_ix, match in enumerate(self.inter_matches):
            s_output_idx = match["s_output_idx"]
            t_output_idx = match["t_output_idx"]

            s_outputs = self.index_able(self.s_outputs.get(match["s_module"]))
            t_outputs = self.index_able(self.t_outputs.get(match["t_module"]))
            if not 0 <= s_output_idx < len(s_outputs):
                raise ValueError(
                    "s_output_idx in {} out of range. Please check add_inter_soft_label config.".format(match)
                )
            if not 0 <= t_output_idx < len(t_outputs):
                raise ValueError(
                    "t_output_idx in {} out of range. Please check add_inter_soft_label config.".format(match)
                )

            s_output = s_outputs[s_output_idx]
            t_output = t_outputs[t_output_idx]

            for loss_module in match["loss_func"]:
                loss_weight = loss_module["func_weight"]
                loss_func = loss_module["func_instance"]
                loss_temperature = loss_module["temperature"]
                if loss_temperature:
                    loss = loss_func(s_output, t_output, loss_temperature)
                else:
                    loss = loss_func(s_output, t_output)
                loss_name = f'inter_match{match_ix}_{loss_module["func_name"]}_loss'
                unweighted_losses[loss_name] = loss.item()
                distill_loss += loss * loss_weight

        return distill_loss, unweighted_losses
