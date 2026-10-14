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

import mindspore
import numpy as np
from mindspore import dtype as mstype, Parameter
from mindspore import Tensor
from mindspore.nn import Cell

from ascend_utils.common.knowledge_distill.utils import replace_module

TENSOR_SHAPE_MAX_LEN = 64


class SaveOutputShapeModule(Cell):
    def __init__(self, module, name, output_idx):
        super().__init__()
        self.name = name
        self.module = module
        self.output_idx = output_idx
        self.output_shape = Parameter(
            mindspore.Tensor(np.zeros((TENSOR_SHAPE_MAX_LEN,)), mstype.float32),
            requires_grad=False,
            name="{}.{}".format(self.name, "module_output_shape"),
        )
        self.output_shape_len = Parameter(
            mindspore.Tensor(0, mstype.int64),
            requires_grad=False,
            name="{}.{}".format(self.name, "module_output_shape_len"),
        )
        self.scatter_nd_update = mindspore.ops.ScatterNdUpdate()

    def construct(self, *args, **kwargs):
        ori_output = self.module(*args, **kwargs)
        if self.output_idx is not None:
            output = ori_output[self.output_idx]
        else:
            output = ori_output

        output_shape = output.shape
        output_shape_len = len(output_shape)
        shape_indices = Tensor([[x] for x in range(0, output_shape_len)], mindspore.int32)
        output_shape = Tensor(output_shape, mstype.int32)
        self.output_shape = self.scatter_nd_update(self.output_shape, shape_indices, output_shape)
        self.output_shape_len = Tensor(output_shape_len, mstype.int64)

        return ori_output


class GetOutputShapeModule:
    def __init__(self, module):
        self.node = module

    def get_name(self):
        return self.node.name

    def get_output_shape(self):
        output_shape_len = self.node.output_shape_len
        output_shape_len = output_shape_len.asnumpy().tolist()

        output_shape = self.node.output_shape
        output_shape = mindspore.ops.cast(output_shape, mindspore.int64)
        output_shape = output_shape[:output_shape_len]
        return output_shape


class SaveOutputModule(Cell):
    def __init__(self, module, name, output_idx, output_shape):
        super().__init__()
        self.name = name
        self.module = module
        self.output_idx = output_idx
        self.output = Parameter(
            mindspore.Tensor(np.zeros(output_shape), mstype.float32),
            requires_grad=False,
            name="{}.{}".format(self.name, "module_output"),
        )

    def construct(self, *args, **kwargs):
        ori_output = self.module(*args, **kwargs)
        # MindSpore only support comparison with 1 operator
        if self.output_idx >= len(ori_output) or self.output_idx < 0:
            raise ValueError(
                "s_output_idx/t_output_idx:{} out of range. Please check add_inter_soft_label config.".format(
                    self.output_idx
                )
            )
        if self.output_idx is not None:
            output = ori_output[self.output_idx]
        else:
            output = ori_output

        self.output = output
        return ori_output


class GetOutputModule(Cell):
    def __init__(self, module):
        super().__init__()
        self.node = module

    def construct(self):
        output = self._get_output()
        return output

    def _get_output(self):
        return self.node.output


class LossModuleBase(Cell):
    def __init__(self, match):
        super().__init__()
        self.modules = []
        self.loss_scale_weights = []
        self.temperatures = []
        for loss in match["loss_func"]:
            self.modules.append(loss["func_instance"])
            self.loss_scale_weights.append(loss["func_weight"])
            self.temperatures.append(loss["temperature"])
        self.s_output_idx = match["s_output_idx"]
        self.t_output_idx = match["t_output_idx"]

    def calc_loss(self, s_output, t_output):
        total_loss = 0.0
        for idx, module in enumerate(self.modules):
            if self.temperatures and self.temperatures[idx]:
                temperature = self.temperatures[idx]
                loss = module(s_output, t_output, temperature=temperature)
            else:
                loss = module(s_output, t_output)
            weight = self.loss_scale_weights[idx]
            loss = loss * weight
            total_loss += loss
        return total_loss


class LossModule(LossModuleBase):
    def __init__(self, match, t_output_module=None, s_output_module=None):
        super().__init__(match)
        self.t_output_module = GetOutputModule(t_output_module) if t_output_module else None
        self.s_output_module = GetOutputModule(s_output_module) if s_output_module else None

    def construct(self, s_output, t_output):
        # when self.t_output_module and self.s_output_module is not None,
        # it means current module is an intermediate module. Otherwise, it's
        # output module of whole network.
        if self.t_output_module and self.s_output_module:
            local_t_output = self.t_output_module.construct()
            local_s_output = self.s_output_module.construct()
        else:
            if self.s_output_idx >= len(s_output) or self.s_output_idx < 0:
                raise ValueError(
                    "s_output_idx:{} out of range. Please check add_output_soft_label config.".format(self.s_output_idx)
                )
            if self.t_output_idx >= len(t_output) or self.t_output_idx < 0:
                raise ValueError(
                    "t_output_idx:{} out of range. Please check add_output_soft_label config.".format(self.t_output_idx)
                )
            local_t_output = t_output[self.t_output_idx]
            local_s_output = s_output[self.s_output_idx]

        return self.calc_loss(local_s_output, local_t_output)


class DistillLossesManagerBase(Cell):
    def __init__(self, config):
        super().__init__()
        self.inter_matches = config.inter_matches
        self.output_matches = config.output_matches

        self.hard_label_loss_weight = config.hard_label_loss_weight
        self.model_parallel = config.model_parallel
        self.output_replace_idx = config.output_replace_idx

    @staticmethod
    def index_able(data):
        if isinstance(data, (list, tuple)):
            return data
        else:
            return [data]

    # the output of some network is a tuple whose element is hard label loss
    # we should read it and add to distill loss as total loss. Finally, we should
    # replace original hard label loss with newly calculated total loss.
    def try_merge_to_ori_output(self, distill_loss, s_outputs):
        if self.output_replace_idx == -1:
            return distill_loss

        # AUTOML: should check if other model has the same output
        idx = self.output_replace_idx
        if idx >= len(s_outputs) or idx < -len(s_outputs):
            raise ValueError("Please check the index of your hard label, it must be an index of the student's outputs.")

        hard_label_loss = s_outputs[idx]
        total_loss = hard_label_loss * self.hard_label_loss_weight + distill_loss * (1 - self.hard_label_loss_weight)
        return self._reconstruct_loss(total_loss, s_outputs)

    def add_t_loss(self, t_outputs, s_outputs):
        if self.output_replace_idx == -1:
            return s_outputs

        idx = self.output_replace_idx
        if not isinstance(s_outputs, (tuple, list)):
            raise ValueError("Please check your student model's type of output, it must be tuple or list.")
        if len(s_outputs) <= idx:
            raise ValueError(
                f"Please check the value of output_replace_idx in yml, it must be smaller than {len(s_outputs)}"
            )
        s_loss = s_outputs[idx]
        total_loss = s_loss + t_outputs
        return self._reconstruct_loss(total_loss, s_outputs)

    def _reconstruct_loss(self, total_loss, s_outputs):
        if self.model_parallel:
            return total_loss
        ret_value = []
        i = 0
        while i < len(s_outputs):
            if i == self.output_replace_idx:
                ret_value.append(total_loss)
            else:
                ret_value.append(s_outputs[i])
            i += 1
        return ret_value


class DistillLossesManager(DistillLossesManagerBase):
    def __init__(self, config, s_model, t_model, output_shapes=None):
        super().__init__(config)
        self._init_ms_modules(s_model, t_model, output_shapes)

    def construct(self, s_output, t_output):
        return self.compute_loss_ms(s_output, t_output)

    def compute_loss_ms(self, s_outputs, t_outputs):
        t_outputs = self.index_able(t_outputs)
        s_outputs = self.index_able(s_outputs)

        # compute distill loss
        distill_loss, _ = self._compute_distill_loss(s_outputs, t_outputs)

        total_loss = self.try_merge_to_ori_output(distill_loss, s_outputs)

        if isinstance(total_loss, Tensor):
            return total_loss, {}
        idx = self.output_replace_idx
        return total_loss[idx], {}

    def get_module_output_shapes(self):
        output_shapes = {}
        for module in self.get_output_shape_modules:
            name = module.get_name()
            output_shape = module.get_output_shape()
            output_shapes[name] = output_shape.asnumpy().tolist()
        return output_shapes

    def restore_modules(self, s_model, t_model):
        # restore modules of s_model and t_model from SaveOutputModule to their original module
        for loss_module in self.loss_modules:
            s_output_module = loss_module.s_output_module
            t_output_module = loss_module.t_output_module
            if not s_output_module or not t_output_module:
                continue

            s_module_name = s_output_module.node.name
            t_module_name = t_output_module.node.name
            ori_s_module = s_output_module.node.module
            ori_t_module = t_output_module.node.module
            replace_module(t_model, t_module_name, ori_t_module)
            replace_module(s_model, s_module_name, ori_s_module)

        # reset parameter names of s_model and t_model
        for name, cell in s_model.name_cells().items():
            cell.update_parameters_name(name + ".")
        for name, cell in t_model.name_cells().items():
            cell.update_parameters_name(name + ".")

    def _compute_distill_loss(self, s_outputs, t_outputs):
        distill_loss = 0
        unweighted_losses = {}
        for loss_module in self.loss_modules:
            loss = loss_module(s_outputs, t_outputs)
            distill_loss += loss
        return distill_loss, unweighted_losses

    def _init_ms_modules(self, s_model, t_model, output_shapes=None):
        self.s_name2module = dict(s_model.cells_and_names())
        self.t_name2module = dict(t_model.cells_and_names())

        self.loss_modules = []
        self.get_output_shape_modules = []
        for match in self.inter_matches:
            t_module_name = match["t_module"]
            s_module_name = match["s_module"]
            t_module = self.t_name2module.get(t_module_name)
            if not t_module:
                raise ValueError(
                    "The module name: \"{}\" is not in the teacher model! Please check config".format(t_module_name)
                )
            s_module = self.s_name2module.get(s_module_name)
            if not s_module:
                raise ValueError(
                    "The module name: \"{}\" is not in the student model! Please check config".format(s_module_name)
                )
            s_output_idx = match["s_output_idx"]
            t_output_idx = match["t_output_idx"]

            if output_shapes is None:  # for generating shape automatically
                tmp_t_module = SaveOutputShapeModule(t_module, t_module_name, t_output_idx)
                tmp_s_module = SaveOutputShapeModule(s_module, s_module_name, s_output_idx)
                self.get_output_shape_modules.append(GetOutputShapeModule(tmp_t_module))
                self.get_output_shape_modules.append(GetOutputShapeModule(tmp_s_module))
            else:
                tmp_t_module = SaveOutputModule(t_module, t_module_name, t_output_idx, output_shapes[t_module_name])
                tmp_s_module = SaveOutputModule(s_module, s_module_name, s_output_idx, output_shapes[s_module_name])
                loss_module = LossModule(match, tmp_t_module, tmp_s_module)
                self.loss_modules.append(loss_module)

            replace_module(t_model, t_module_name, tmp_t_module)
            replace_module(s_model, s_module_name, tmp_s_module)
        if output_shapes:
            # new a LossModule to calculate loss of output_match modules
            for match in self.output_matches:
                loss_module = LossModule(match)
                self.loss_modules.append(loss_module)

        s_model.update_parameters_name("student_model.")
        t_model.update_parameters_name("teacher_model.")
