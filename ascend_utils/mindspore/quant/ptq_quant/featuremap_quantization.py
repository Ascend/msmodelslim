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
# pylint: disable=consider-using-from-import

import numpy as np
import mindspore.nn as nn
from mindspore import Tensor

from ascend_utils.mindspore.quant.ptq_quant.simulated_quant import SimulatedQuant


def rename_parameters(cell, prefix):
    """rename the parameters of cell with input prefix"""
    for key in cell.parameters_dict().keys():
        sufix = key.split('.')[-1]
        new_name = '.'.join([prefix, sufix])
        cell.parameters_dict()[key].name = new_name


def insert_child_to_cell(network, target_cell_name, new_subcell):
    """insert the new_subcell to original's father cell"""
    change = False
    cell_name_list = target_cell_name.split('.')
    sub_cell = None
    for cell_name, sub_cell in network.cells_and_names():
        if cell_name == '.'.join(cell_name_list[:-1]):
            sub_cell.insert_child_to_cell(cell_name_list[-1], new_subcell)
            change = True
            break
    if isinstance(sub_cell, nn.SequentialCell) and change:
        sub_cell.cell_list = list(sub_cell.cells())


def quant_activation(network, cell_name, cell):
    num_bins = 128
    max_percentile = 0.999999
    min_percentile = 0.999999
    search_range_start = 0.7
    search_range_end = 1.3
    search_step = 0.01
    shape = (1,)
    scale_init = Tensor(np.ones(*shape).astype(np.float32))
    offset_init = Tensor(np.zeros(*shape).astype(np.float32))
    new_subcell = SimulatedQuant(
        cell,
        cell_name,
        scale_init,
        offset_init,
        num_bins,
        min_percentile,
        max_percentile,
        search_range_start,
        search_range_end,
        search_step,
        True,
    )
    insert_child_to_cell(network, cell_name, new_subcell)
