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

from ascend_utils.mindspore.quant.ptq_quant.weight_quantization import quant_weight
from ascend_utils.mindspore.quant.ptq_quant.featuremap_quantization import quant_activation
from ascend_utils.mindspore.quant.ptq_quant.process_utils import fuse_bn
from ascend_utils.mindspore.quant.ptq_quant.utils import custom_deepcopy
from ascend_utils.common.security import json_safe_load
from ascend_utils.common.security.mindspore import check_mindspore_cell
from ascend_utils.common.security.mindspore import check_mindspore_input
from msmodelslim.mindspore.quant.ptq_quant.rollback_quant_nodes import get_rollback_nodes
from msmodelslim import logger


def quantize_model(config_file, model, *input_data):
    """
    Modify the network structure according to the quantization configuration file. Insert relevant operators such as
    weight quantization and activations quantization, and then return the modified network.
    :param config_file: Storage path and name of the quantization configuration file.
    :param model: Pretrained model instance to be quantified.
    :return: A modified network that needed calibration.
    :raises ValueError: Error occurred when accessing the wrong config_file/model/input_data.
    """
    raw_dict = json_safe_load(config_file)
    check_mindspore_cell(model)
    check_mindspore_input(input_data)
    check_config_file(raw_dict)
    is_fuse_bn, num_rollback_nodes, layer_need_quantization_list = get_config_file_param(raw_dict)
    model_quant = quantize(model, input_data, is_fuse_bn, layer_need_quantization_list)

    if num_rollback_nodes > 0:
        model_original = custom_deepcopy(model)
        rollback_nodes = get_rollback_nodes(model_original, model_quant,
                                            input_data, num_rollback_nodes, layer_need_quantization_list)
        newline = "\n"
        logger.info("rollback node(s): %s", newline + newline.join(rollback_nodes))
        for node in rollback_nodes:
            if node in layer_need_quantization_list:
                layer_need_quantization_list.remove(node)
        model_quant = quantize(model, input_data, is_fuse_bn, layer_need_quantization_list)

    return model_quant


def check_config_file(dict_value):
    for key, value in dict_value.items():
        if key == "fuse_bn" and not isinstance(value, bool):
            raise TypeError("The value of the fuse_bn must be True or False.")
        if key == "num_rollback_nodes" and not isinstance(value, int):
            raise TypeError("The value of the num_rollback_nodes must be int.")
        if key != "version" and key != "fuse_bn" and key != "num_rollback_nodes":
            if not value or not isinstance(value, dict) or "quant" not in value:
                raise ValueError("The key of the quant must be set.")
            if not isinstance(value["quant"], bool):
                raise TypeError("The value of the quant must be True or False.")

    num_quantized_nodes = sum([key not in ("version", "fuse_bn", "num_rollback_nodes") for key in dict_value.keys()])
    num_rollback_nodes = dict_value.get("num_rollback_nodes")
    if num_rollback_nodes:
        if not (0 <= num_rollback_nodes < num_quantized_nodes):
            raise ValueError("The value of the num_rollback_nodes must be in [0, num of quantized nodes)")


def get_config_file_param(dict_value):
    layer_need_quantization_list = []
    is_fuse_bn = True
    num_rollback_nodes = dict_value.get("num_rollback_nodes", 0)
    for key, value in dict_value.items():
        if key == "version":
            continue
        if key == "fuse_bn" and not value:
            is_fuse_bn = False
        if isinstance(value, dict) and value.get("quant"):
            layer_need_quantization_list.append(key)
    logger.info("Config file read successfully")
    return is_fuse_bn, num_rollback_nodes, layer_need_quantization_list


def quantize(model, input_data, is_fuse_bn, layer_need_quantization_list):
    model_quant = custom_deepcopy(model)
    if is_fuse_bn:
        model_quant, _ = fuse_bn(model_quant, input_data)
    for _, (name, cell) in enumerate(model_quant.cells_and_names()):
        if name in layer_need_quantization_list:
            quant_weight(cell)
            quant_activation(model_quant, name, cell)
    return model_quant
