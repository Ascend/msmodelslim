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

import itertools
import math
from typing import List, Union, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from ascend_utils.common.utils import CallParams
from ascend_utils.pytorch.dag.dag_torch_hook import DagTorchHook
from msmodelslim import logger


class DepthScaleNetwork:
    class RepeatOperatorInfo:
        def __init__(self, name, parent_module, attr_name, module, enable):
            self.name, self.parent_module, self.attr_name, self.module = name, parent_module, attr_name, module
            self.enable = enable

    def __init__(self, network: Union[torch.nn.Module, DagTorchHook],
                 inputs: Union[Tensor, List[Tensor], Tuple[Tensor], CallParams] = None):
        if isinstance(network, torch.nn.Module):
            self._network = network
            self._dag_network = DagTorchHook(network, inputs)
        elif isinstance(network, DagTorchHook):
            self._network = network.network
            self._dag_network = network
        else:
            raise ValueError("network must be torch.nn.Module or Dag")
        self._repeat_operators: List[List[DepthScaleNetwork.RepeatOperatorInfo]] = []
        self._analysis(self.network)
        self._cur_scale = 1.0

    @property
    def dag(self):
        return self._dag_network

    @property
    def network(self):
        return self._network

    @property
    def repeat_operators(self):
        return self._repeat_operators

    @staticmethod
    def disable_operator(operator: RepeatOperatorInfo):
        setattr(operator.parent_module, operator.attr_name, JumpingOffOperator())
        operator.enable = False

    @staticmethod
    def enable_operator(operator: RepeatOperatorInfo, param_copy_operator: RepeatOperatorInfo):
        if operator is param_copy_operator:
            return

        setattr(operator.parent_module, operator.attr_name, operator.module)
        operator.enable = True

        src_param_info = {name: parameter for name, parameter in param_copy_operator.module.named_parameters()}
        dest_param_info = {name: parameter for name, parameter in operator.module.named_parameters()}

        for name, parameter in dest_param_info.items():
            if name not in src_param_info:
                raise ValueError(f"name [{name}] not in src param info ")
            parameter.data = src_param_info[name].data.clone()
            
    @staticmethod
    def _is_repeat_opts_with_weight(last_module: Module, this_module: Module):
        if this_module is None:
            return False
        if this_module.__class__ != last_module.__class__:
            return False

        last_param_info = [(name, parameter.shape) for name, parameter in last_module.named_parameters()]
        this_param_info = [(name, parameter.shape) for name, parameter in this_module.named_parameters()]
        if len(last_param_info) == 0 or len(this_param_info) == 0:
            return False

        return last_param_info == this_param_info

    @staticmethod
    def _calc_enable_list(ori_enable_count, new_enable_count, all_count):
        new_enable_count = min(new_enable_count, all_count)
        if ori_enable_count >= new_enable_count:
            enable_list = list(range(new_enable_count))
        else:
            if ori_enable_count == 0:
                raise ZeroDivisionError("The ori_enable_count can not be zero!")
            floor_value = math.floor(new_enable_count / ori_enable_count)

            enable_list = list(itertools.chain(
                *itertools.repeat(list(range(ori_enable_count)), floor_value),
                range(new_enable_count - floor_value * ori_enable_count)
            ))

            enable_list.sort()
        unable_list = [None] * (all_count - len(enable_list))
        return enable_list + unable_list
    
    def scale(self, scale: float):
        if not isinstance(scale, (int, float)):
            raise TypeError(f"scale={scale} is required to be an int or float")
        if scale <= 0:
            raise ValueError(f"scale={scale} is required to be an int or float and larger than 0")

        self._cur_scale = self._cur_scale * scale
        prune_multiple = int(1 / self._cur_scale)
        self.prune(prune_multiple)

    def prune(self, prune_multiple: int):
        if prune_multiple == 0:
            raise ZeroDivisionError("The prune_multiple can not be zero.")
        for repeat_operator_list in self.repeat_operators:
            all_count = len(repeat_operator_list)
            ori_enable_count = len(list(filter(lambda x: x.enable, repeat_operator_list)))
            new_enable_count = math.ceil(all_count / prune_multiple)
            enable_list = self._calc_enable_list(ori_enable_count, new_enable_count, all_count)
            logger.info(enable_list)
            for r_index, is_enable in enumerate(reversed(enable_list)):
                index = all_count - r_index - 1
                if is_enable is not None:
                    self.enable_operator(repeat_operator_list[index], repeat_operator_list[is_enable])
                else:
                    self.disable_operator(repeat_operator_list[index])

    def _analysis(self, module: Module):
        sub_module_list = list(module.named_children())
        sub_module_list.sort(key=lambda x: self.dag.calc_order.index(x[1]) if x[1] in self.dag.calc_order else -1)

        last_module = None
        no_analysis_opts = []
        repeat_opts: List[DepthScaleNetwork.RepeatOperatorInfo] = list()
        if len(sub_module_list) >= 2:
            for attr_name, sub_module in sub_module_list:
                is_repeat = self._is_repeat_opts_with_weight(last_module, sub_module)
                if not is_repeat and repeat_opts:
                    self._repeat_operators.append(repeat_opts)
                    repeat_opts = []
                if not is_repeat:
                    last_module = sub_module
                    continue

                if not repeat_opts:
                    repeat_opts.append(
                        self.RepeatOperatorInfo(self._get_module_name(last_module), module, attr_name, last_module,
                                                enable=True))
                    no_analysis_opts.append(last_module)

                repeat_opts.append(
                    self.RepeatOperatorInfo(self._get_module_name(sub_module), module, attr_name, sub_module,
                                            enable=True))
                no_analysis_opts.append(sub_module)

            if repeat_opts:
                self._repeat_operators.append(repeat_opts)

        for _, sub_module in module.named_children():
            if sub_module in no_analysis_opts:
                continue
            self._analysis(sub_module)

    def _get_module_name(self, module):
        if module not in self.dag.structure_tree:
            return ""
        return self.dag.structure_tree[module].get("name_in_network", "")


class JumpingOffOperator(Module):
    def forward(self, *args):
        return args[0] if len(args) == 1 else args
