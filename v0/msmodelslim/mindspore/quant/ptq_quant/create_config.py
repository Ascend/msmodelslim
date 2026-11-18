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

from ascend_utils.mindspore.quant.ptq_quant.utils import is_quant_cell
from ascend_utils.common.security.mindspore import check_mindspore_cell
from ascend_utils.common.security import json_safe_dump
from msmodelslim import logger


def create_quant_config(config_file, model):
    """
    Find all quantifiable layers according to the network structure, and automatically generate a quantization
    configuration file. Writes the quantization configuration information of the quantifiable layer into the
    configuration file.
    :param config_file: Storage path and name of the quantization configuration file to be generated. If the file
    already exists in the storage path, the existing file will be overwritten.
    :param model: Model instance to be quantified.
    :return: Configuration file.
    :raises ValueError: Error occurred when accessing the wrong config_file/model.
    """
    check_mindspore_cell(model)
    quant_layer_name_list = get_quant_layer(model)
    generate_config(config_file, quant_layer_name_list)


def get_quant_layer(model):
    quant_layer_name_list = []
    for _, (name, cell) in enumerate(model.cells_and_names()):
        if is_quant_cell(cell):
            quant_layer_name_list.append(name)
    return quant_layer_name_list


def generate_config(config_file, name_list):
    raw_data = {"version": "1.0", "fuse_bn": True, "num_rollback_nodes": 0}
    for name in name_list:
        raw_data[name] = {"quant": True}
    json_safe_dump(raw_data, config_file, indent=4)
    logger.info("Config file generated successfully: %r", config_file)
