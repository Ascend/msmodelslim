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
import torch
import mindspore
from mindspore import nn
try:
    from mindspore.nn.transformer.layers import _Linear
except ModuleNotFoundError:
    try:
        from mindspore.nn.layer.transformer import _Linear
    except ImportError:
        from mindspore.nn import Dense as _Linear

from ascend_utils.common.mindspore_utils import SaveInput, update_cell
from ascend_utils.mindspore.quant.ptq_quant.simulated_quant import SimulatedQuant


def get_rollback_nodes(model, model_quant, input_data, num_rollback_nodes, layer_need_quantization_list):
    model = replace_module_with_saving_input(model, layer_need_quantization_list)
    model_quant = replace_module_with_saving_input(model_quant, layer_need_quantization_list)

    model_outputs = extract_intermediate_outputs(model, input_data, layer_need_quantization_list)
    model_quant_outputs = extract_intermediate_outputs(model_quant, input_data, layer_need_quantization_list)
    if model_outputs.keys() != model_quant_outputs.keys():
        raise RuntimeError("output modules of original model and quantized model are not same, check model")
    output_mse = {}
    for node, model_output in model_outputs.items():
        model_quant_output = model_quant_outputs.get(node, mindspore.Tensor(0, mstype.int64))
        output_mse[node] = mse(model_output, model_quant_output)
    output_mse = dict(sorted(output_mse.items(), key=lambda kv: kv[1], reverse=True))

    rollback_nodes = list(output_mse.keys())[:num_rollback_nodes]
    return rollback_nodes


def replace_module_with_saving_input(network, layer_need_quantization_list):
    for name, cell in network.cells_and_names():
        if name in layer_need_quantization_list:
            if isinstance(cell, SimulatedQuant):
                compute_cell = cell.compute_cell
            else:
                compute_cell = cell
            if isinstance(compute_cell, (nn.Dense, _Linear)):
                layer_with_input = \
                    nn.SequentialCell([cell, SaveInput(compute_cell.out_channels, name, is_channel_first=False)])
                update_cell(network, cell, name, layer_with_input)
            elif isinstance(compute_cell, (nn.Conv2d, nn.Conv1d)):
                layer_with_input = \
                    nn.SequentialCell([cell, SaveInput(compute_cell.out_channels, name, is_channel_first=True)])
                update_cell(network, cell, name, layer_with_input)
    return network


def extract_intermediate_outputs(model, input_data, layer_need_quantization_list):
    model(*input_data)
    outputs = {}
    for name, cell in model.cells_and_names():
        if name in layer_need_quantization_list:
            outputs[name] = cell[1].input_data.asnumpy()
    return outputs


def mse(input1, input2):
    input1 = input1.flatten()
    input2 = input2.flatten()
    return ((input1 - input2) ** 2).mean()
