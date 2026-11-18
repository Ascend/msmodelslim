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
# pylint: disable=unnecessary-dunder-call

import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import initializer


class SaveInput(nn.Cell):
    def __init__(self, num_channels, source_name, is_channel_first=True, **kwargs):
        super().__init__(**kwargs)

        self.num_samples = ms.Parameter(initializer(0, [1]), requires_grad=False, name='num_samples')
        self.input_data = ms.Parameter(
            initializer(0, [num_channels, num_channels]), requires_grad=False, name='input_data'
        )
        self.self_matmul = ms.ops.MatMul(transpose_a=True)
        self.source_name = source_name
        self.is_channel_first = is_channel_first

    def construct(self, inputs: ms.Tensor, *args):
        if inputs.ndim != 2:
            # For input shape [1, 2, 3, 4], same as transpose([0, 2, 3, 1])
            input_data = inputs.expand_dims(-1).swapaxes(1, -1).squeeze(1) if self.is_channel_first else inputs
            input_data = input_data.reshape([-1, input_data.shape[-1]])
        else:
            input_data = inputs

        cur_num_samples = input_data.shape[0]
        input_data = self.self_matmul(input_data, input_data)
        input_data = self.input_data * self.num_samples + input_data
        num_samples = self.num_samples + cur_num_samples
        self.input_data = ms.ops.div(input_data, num_samples)
        self.num_samples = num_samples
        return inputs


def update_cell(network, old_cell, name, new_cell):
    sub_module_names = name.strip().split('.')
    for sub_module_name in sub_module_names[0:-1]:
        network = network.__getattr__(sub_module_name)
    old_cell.__del__()
    if isinstance(network, nn.SequentialCell):
        network.cell_list[int(sub_module_names[-1])] = new_cell
    network.__setattr__(sub_module_names[-1], new_cell)
    new_cell.update_parameters_name(name + ".")
