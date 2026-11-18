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


from collections import namedtuple, defaultdict
from copy import deepcopy

import torch
import numpy as np

from ascend_utils import count_parameters
from ascend_utils.common import security
from msmodelslim.pytorch.sparse.width_scale_network import WidthScaleNetwork
from msmodelslim.pytorch.sparse.depth_scale_network import DepthScaleNetwork
from msmodelslim import logger


SparseMethods = namedtuple('SparseMethods', ['DEPTH_GROWTH', 'WIDTH_GROWTH'])('depth_growth', 'width_growth')


class OptimizerWithReset:
    def __init__(self, model, optimizer):
        security.check_type(model, torch.nn.Module, param_name="model")
        security.check_type(optimizer, torch.optim.Optimizer, param_name="optimizer_with_reset")

        self._init_params_group_info(model, optimizer)
        self._optimizer = optimizer

    def count_parameters(self):
        return sum([sum([np.prod(list(jj.shape)) for jj in ii['params']]) for ii in self._optimizer.param_groups])

    def reset(self, model):
        self._optimizer.param_groups = []
        param_groups = deepcopy(self._param_groups_args)
        for name, param in model.named_parameters():
            if not param.requires_grad:  # Like running_var and running_mean
                continue
            param_groups[self._params_group_dict[name]]["params"].append(param)
        for param_group in param_groups:
            self._optimizer.add_param_group(param_group)
        setattr(self._optimizer, "state", defaultdict(dict))

    def _init_params_group_info(self, model, optimizer):
        id_dict = {}
        for group_id, group in enumerate(optimizer.param_groups):
            for param in group['params']:
                id_dict[id(param)] = group_id
        self._params_group_dict = {name: id_dict[id(param)] for name, param in model.named_parameters()}

        self._param_groups_args = []
        for group in optimizer.param_groups:
            group_args = {arg_name: [] if arg_name == "params" else arg_val for arg_name, arg_val in group.items()}
            self._param_groups_args.append(deepcopy(group_args))


class SparseForward:
    def __init__(self, model, optimizer, steps_per_epoch, epochs_each_stage, sparse_mode):
        security.check_type(model, torch.nn.Module, param_name="model")
        security.check_type(optimizer, torch.optim.Optimizer, param_name="optimizer")
        security.check_int(steps_per_epoch, min_value=1, param_name="steps_per_epoch")
        security.check_type(epochs_each_stage, (list, tuple), param_name="epochs_each_stage")
        security.check_int(len(epochs_each_stage), min_value=2, param_name="len(epochs_each_stage)")
        security.check_type(
            epochs_each_stage[:-1],
            value_type=(list, tuple),
            additional_check_func=lambda xx: isinstance(xx, int) and xx > 0,
            param_name="epochs_each_stage",
        )
        security.check_int(epochs_each_stage[-1], min_value=-1, param_name="epochs_each_stage[-1]")  # -1 elem can be -1
        security.check_type(
            sparse_mode,
            value_type=str,
            param_name="sparse_mode",
            additional_check_func=lambda xx: xx in SparseMethods,
            additional_msg=f"Should be one of {list(SparseMethods)}",
        )
        logger.info(f"SparseForward steps_per_epoch: {steps_per_epoch}, epochs_each_stage: {epochs_each_stage}")
        logger.info(f"Sparse mode: {sparse_mode}")

        self._sparse_mode = sparse_mode
        self._dag_module = WidthScaleNetwork if sparse_mode == SparseMethods.WIDTH_GROWTH else DepthScaleNetwork
        self._steps_per_epoch, self._epochs_each_stage = steps_per_epoch, epochs_each_stage
        self._optimizer_with_reset = OptimizerWithReset(model, optimizer)

        self._num_stages = len(epochs_each_stage)
        cum_steps_each_stage = [0]  # epochs_each_stage=[1, 2, 3], steps_per_epoch=10 -> [10, 30, 60]
        for stage_epochs in epochs_each_stage:
            cur_stage_steps = stage_epochs * steps_per_epoch
            cum_steps_each_stage.append(cur_stage_steps + cum_steps_each_stage[-1])
        if epochs_each_stage[-1] == -1:
            cum_steps_each_stage[-1] = np.inf  # Keep in the last stage till total training epoch ends
        self._cum_steps_each_stage = np.array(cum_steps_each_stage[1:])  # 1: means excluding the first 0

        # For 3 stages -> [0.25, 2, 2], where 0.25 means prune model width to 1/4, 2 means double model width
        self._scale_list = [1 / 2 ** (self._num_stages - 1)] + [2] * (self._num_stages - 1)
        self._model = model
        self._model.original_forward = model.forward  # Rename forward -> original_forward, will overwrite forward later

        self._dag_model, self._global_batch_id, self._built_stage = None, 0, -1  # Init values
        self._is_on_npu = next(model.parameters()).device.type == "npu"

    def __call__(self, *args, **kwargs):
        if not self._model.training:
            return self._model.original_forward(*args, **kwargs)

        cur_stage = self.get_cur_stage()
        if cur_stage != self._built_stage and cur_stage < self._num_stages:
            if cur_stage != self._built_stage + 1:
                raise ValueError(
                    f"cur_stage={cur_stage} not equal with built_stage={self._built_stage}+1, this should not happen."
                )
            logger.info(
                f"SparseForward Stage: {self._built_stage} -> {cur_stage}, scale: {self._scale_list[cur_stage]}"
            )
            logger.info(f"Previous model parameters: {count_parameters(self._model)}")
            logger.info(f"Previous optimizer parameters: {self._optimizer_with_reset.count_parameters()}")

            input_shapes = [arg_input.shape for arg_input in args if hasattr(arg_input, "shape")]
            input_shapes += [kwarg_input.shape for kwarg_input in kwargs.values() if hasattr(kwarg_input, "shape")]
            if len(input_shapes) == 0:
                raise ValueError("At least 1 input should be provided.")
            logger.debug(f"SparseForward input_shapes: {input_shapes}")

            with torch.no_grad():
                self._model = self.scale_model(self._model, input_shapes, self._scale_list[cur_stage])
                self._optimizer_with_reset.reset(self._model)
            logger.info(f"New model parameters: {count_parameters(self._model)}")
            logger.info(f"New optimizer parameters: {self._optimizer_with_reset.count_parameters()}")

            self._built_stage += 1
        return self._model.original_forward(*args, **kwargs)

    def scale_model(self, model, input_shapes, scale):
        is_training = model.training
        model.eval()

        if self._dag_model is None:
            dummy_inputs = [torch.ones([1] + list(input_shape)[1:]) for input_shape in input_shapes]  # batch_size as 1
            if self._is_on_npu:
                dummy_inputs = [dummy_input.npu() for dummy_input in dummy_inputs]
            self._dag_model = self._dag_module(model, dummy_inputs[0] if len(dummy_inputs) == 1 else dummy_inputs)
        self._dag_model.scale(scale)

        if is_training:
            model.train()
        return model

    def get_cur_stage(self):
        self._global_batch_id += 1
        logger.debug(f"SparseForward global_batch_id: {self._global_batch_id}")
        return np.sum(self._cum_steps_each_stage < self._global_batch_id)

    def set_initial_epoch(self, initial_epoch):
        self._global_batch_id = initial_epoch * self._steps_per_epoch
        self._built_stage = self.get_cur_stage()


def sparse_model_width(model, optimizer, steps_per_epoch, epochs_each_stage):
    """ Wrapper a model for width sparse training
    Args:
      model: initialized torch model. `torch.nn.Module` instance.
      optimizer: initialized PyTorch optimizer.
      steps_per_epoch: int value for steps in each training repoch.
        The last element can be `-1` means training till total epochs end.
      epochs_each_stage: list or tuple value for epochs in each stage.
        For value like `[10, 20, -1]` means 3 training stages:
        - 1st stage, prune model width to `1/4` as the start model and train 10 epochs.
        - 2nd stage, double start model in the 1st stage to `1/2` as original model and train 20 epochs.
        - 3rd stage, double model in the 2nd stage to same as the original model.
          Epoch number `-1` means training till total epochs end.

    Return
      - model with `forward` being replaced by `SparseForward`.

    Examples:
    >>> import torch
    >>> from torchvision import models
    >>> from msmodelslim.pytorch.sparse import sparse_model_width
    >>> model = models.vgg16()
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    >>> model = sparse_model_width(model, optimizer, steps_per_epoch=100, epochs_each_stage=[1, 2, 1])
    >>> _ = model.train()
    >>> print("output shape:", model(torch.ones([1, 3, 224, 224])).shape)
    >>> # msmodelslim-logger - INFO - SparseForward Stage: -1 -> 0, scale: 4
    >>> # msmodelslim-logger - INFO - Previous model parameters: 138357544
    >>> # msmodelslim-logger - INFO - Previous optimizer parameters: 138357544
    >>> # msmodelslim-logger - INFO - New model parameters: 9418936
    >>> # msmodelslim-logger - INFO - New optimizer parameters: 9418936
    >>> # output shape: torch.Size([1, 1000])
    """
    sparse_mode = SparseMethods.WIDTH_GROWTH
    model.forward = SparseForward(model, optimizer, steps_per_epoch, epochs_each_stage, sparse_mode=sparse_mode)
    return model


def sparse_model_depth(model, optimizer, steps_per_epoch, epochs_each_stage):
    """ Wrapper a model for depth sparse training
    Args:
      model: initialized torch model. `torch.nn.Module` instance.
      optimizer: initialized PyTorch optimizer.
      steps_per_epoch: int value for steps in each training repoch.
        The last element can be `-1` means training till total epochs end.
      epochs_each_stage: list or tuple value for epochs in each stage.
        For value like `[10, 20, -1]` means 3 training stages:
        - 1st stage, prune model depth to `1/4` as the start model and train 10 epochs.
        - 2nd stage, double start model in the 1st stage to `1/2` as original model and train 20 epochs.
        - 3rd stage, double model in the 2nd stage to same as the original model.
          Epoch number `-1` means training till total epochs end.

    Return model with `forward` being replaced by `SparseForward`

    Examples:
    >>> import torch
    >>> from torchvision import models
    >>> from msmodelslim.pytorch.sparse import sparse_model_depth
    >>> model = models.vit_b_16()
    >>> opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    >>> model = sparse_model_depth(model, opt, steps_per_epoch=100, epochs_each_stage=[1, 2, 1])
    >>> _ = model.train()
    >>> print("output shape:", model(torch.ones([1, 3, 224, 224])).shape)
    >>> # msmodelslim-logger - INFO - SparseForward Stage: -1 -> 0, scale: 4
    >>> # msmodelslim-logger - INFO - Previous model parameters: 86567656
    >>> # msmodelslim-logger - INFO - Previous optimizer parameters: 86567656
    >>> # msmodelslim-logger - INFO - New model parameters: 22776808
    >>> # msmodelslim-logger - INFO - New optimizer parameters: 22776808
    >>> # output shape: torch.Size([1, 1000])
    """
    sparse_mode = SparseMethods.DEPTH_GROWTH
    model.forward = SparseForward(model, optimizer, steps_per_epoch, epochs_each_stage, sparse_mode=sparse_mode)
    return model
