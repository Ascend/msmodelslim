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

from collections import defaultdict
import torch
from msmodelslim import logger
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs import simulate_tp


def get_scale(amax, num_bits):
    max_bound = 2 ** (num_bits - 1) - 1
    try:
        scale = amax / max_bound
    except ZeroDivisionError as ex:
        logging.error('max_bound can not be zero. %s', str(ex))
        raise ex
    scale = scale.squeeze()
    return scale


class ParallelLinearCol(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w_split = None
        self.bias = None
        self.tp_list = None
        self.is_calib = True
        self.is_dynamic = False
        self.reduce_max = None
        self.gather_max = None
        self.name = ''
        self.cfg = None
        self.quant_type = QuantType.FLOAT

    def set_param(self, linear, name, cfg):
        self.cfg = cfg
        self.name = name
        weight = linear.weight.data
        if (weight.shape[-1] % self.cfg.tp_size) != 0:
            raise ValueError(f'Linear.weight shape is {weight.shape}, '
                             f'and cannot be divided by tp_size {self.cfg.tp_size}')
        try:
            self.bias = linear.bias.data
        except AttributeError:
            self.bias = None
        w_split = torch.tensor_split(weight, self.cfg.tp_size, dim=-1)

        # 如果linear有bias则只在0卡上加bias
        tp_list = torch.nn.ModuleList(
            [torch.nn.Linear(in_features=1, out_features=1, bias=False) for _ in range(self.cfg.tp_size)])
        for i in range(self.cfg.tp_size):
            tp_list[i].weight = torch.nn.Parameter(w_split[i])

        if self.bias is not None:
            tp_list[0].bias = torch.nn.Parameter(self.bias)

        if isinstance(linear, torch.nn.Linear):
            self.tp_list = tp_list
            self.quant_type = QuantType.FLOAT
        else:
            self.quant_type = self.cfg.model_quant_type
            quant_tp_list = torch.nn.ModuleList(
                [linear.__class__(cfg=cfg, logger=logger) for _ in range(self.cfg.tp_size)]
            )
            for i in range(self.cfg.tp_size):
                quant_tp_list[i].set_param(tp_list[i])
            self.tp_list = quant_tp_list
            del tp_list

    def get_quant_param(self):
        quant_param = defaultdict(torch.Tensor)
        attach_map = defaultdict(list)
        if not self.is_dynamic and self.cfg.enable_communication_quant:
            quant_param[self.name + '.gather_scale'] = get_scale(self.gather_max, num_bits=self.cfg.simulate_bit)
            attach_map[self.name + '.weight'].append(self.name + '.gather_scale')
            if self.cfg.enable_per_device_quant:
                for i in range(self.cfg.tp_size):
                    quant_param[self.name + f'.reduce_scale.{i}'] = \
                        get_scale(self.reduce_max[i], num_bits=self.cfg.simulate_bit)
                    attach_map[self.name + '.weight'].append(self.name + f'.reduce_scale.{i}')
            else:
                quant_param[self.name + '.reduce_scale'] = get_scale(self.reduce_max, num_bits=self.cfg.simulate_bit)
                attach_map[self.name + '.weight'].append(self.name + '.reduce_scale')
        if self.quant_type != QuantType.FLOAT:
            quant_param[self.name + '.weight'] = torch.concat([linear.weight.cpu() for linear in self.tp_list])

        for tp_index, _ in enumerate(self.tp_list):
            linear_name = '.'.join([self.name, f'tp_list', str(tp_index)])

            # W8A16、W4A16和W8A8动态量化
            if self.quant_type in [QuantType.W8A16, QuantType.W4A16, QuantType.W8A8_DYNAMIC]:
                attach_map[self.name + '.weight'].append(linear_name + '.weight_scale')
                attach_map[self.name + '.weight'].append(linear_name + '.weight_offset')

            # W8A8/W8A8S 需要提供 deq_scale、quant_bias、input_scale、input_offset
            if self.quant_type in [QuantType.W8A8, QuantType.W8A8S]:
                attach_map[self.name + '.weight'].append(linear_name + '.input_scale')
                attach_map[self.name + '.weight'].append(linear_name + '.input_offset')
                attach_map[self.name + '.weight'].append(linear_name + '.quant_bias')
                attach_map[self.name + '.weight'].append(linear_name + '.deq_scale')

        return quant_param, attach_map

    def forward(self, tensor):
        with torch.no_grad():
            if (tensor.shape[-1] % self.cfg.tp_size) > 0:
                raise ValueError(f"Linear.input shape is {tensor.shape}, "
                                 f"and cannot be divided by tp_size {self.cfg.tp_size}")
            x_split = torch.tensor_split(tensor, self.cfg.tp_size, dim=-1)
            output_split = [self.tp_list[i](x_split[i]) for i in range(self.cfg.tp_size)]

            if not self.cfg.enable_communication_quant:
                output = sum(output_split)
                return output

            if self.is_calib or self.is_dynamic:
                output, coming_reduce_max, coming_gather_max = \
                    simulate_tp(output_split, enable_per_device_quant=self.cfg.enable_per_device_quant)
                self.reduce_max = self.update_reduce_max(coming_reduce_max)
                self.gather_max = self.update_gather_max(coming_gather_max)
            else:
                output, _, _ = simulate_tp(output_split,
                                           output_reduce_max=self.reduce_max,
                                           output_gather_max=self.gather_max,
                                           enable_per_device_quant=self.cfg.enable_per_device_quant)
            return output

    def update_reduce_max(self, coming_reduce_max):
        if self.reduce_max is None:
            return coming_reduce_max
        if self.cfg.enable_per_device_quant:
            reduce_max = []
            for i in range(self.cfg.tp_size):
                reduce_max.append(torch.max(self.reduce_max[i], coming_reduce_max[i]))
        else:
            reduce_max = torch.max(self.reduce_max, coming_reduce_max)
        return reduce_max

    def update_gather_max(self, coming_gather_max):
        if self.gather_max is None:
            return coming_gather_max
        return torch.max(self.gather_max, coming_gather_max)

    def disable_calib(self):
        self.is_calib = False
