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

import mindspore.nn as nn
from ascend_utils.mindspore.quant.ptq_quant.simulated_quant import SimulatedQuant
from ascend_utils.mindspore.quant.ptq_quant.deploy.deploy_quant_factory import DeployQuantFactory


def get_name_prefix(cell):
    param = list(cell.parameters_dict().keys())[0]
    name_list = param.split('.')[:-1]
    name_prefix = '.'.join(name_list)
    return name_prefix


def rename_parameters(cell, prefix):
    for key in cell.parameters_dict().keys():
        sufix = key.split('.')[-1]
        new_name = '.'.join([prefix, sufix])
        cell.parameters_dict()[key].name = new_name


def convert_to_inference_network(network):
    for name, cell in network.name_cells().items():
        if cell == network:
            continue
        if isinstance(cell, SimulatedQuant):
            new_subcell = DeployQuantFactory.creat_deploy_quant_op(cell)
            name_prefix = get_name_prefix(cell)
            rename_parameters(new_subcell, name_prefix)
            network.insert_child_to_cell(name, new_subcell)
        else:
            convert_to_inference_network(cell)
    if isinstance(network, nn.SequentialCell):
        network.cell_list = list(network.cells())
