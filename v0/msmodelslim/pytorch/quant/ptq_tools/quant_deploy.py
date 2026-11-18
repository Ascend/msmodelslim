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

import copy
import logging

import numpy as np
import onnx
from onnx import helper
import torch

from msmodelslim.pytorch.quant.ptq_tools.ptq_kia.weight_transform import (
    calculate_conv_int_bias,
    create_new_weight,
    transform_conv_input,
    transform_matmul_input,
    transform_add_input,
    tansform_quant_output,
    delete_same_bias_name,
    get_quant_index,
    get_onnx_datatype,
    get_np_datatype
)  # squant algorithm api


class InitQuantParams:
    def __init__(self,
                 weight_name,
                 input_scale_dict,
                 input_offset_dict,
                 weight_scale_dict,
                 weight_offset_dict,
                 quant_weight_dict):
        self.weight_name = weight_name
        self.input_scale_dict = input_scale_dict
        self.input_offset_dict = input_offset_dict
        self.weight_scale_dict = weight_scale_dict
        self.weight_offset_dict = weight_offset_dict
        self.quant_weight_dict = quant_weight_dict


class ModelDeployQuantParams:
    def __init__(self,
                 quantized_weight_name, 
                 quant_weight_dict, 
                 input_scale_dict, 
                 input_offset_dict, 
                 weight_scale_dict, 
                 weight_offset_dict,
                 fuse_add=False):
        self.quantized_weight_name = quantized_weight_name
        self.quant_weight_dict = quant_weight_dict
        self.input_scale_dict = input_scale_dict
        self.input_offset_dict = input_offset_dict
        self.weight_scale_dict = weight_scale_dict
        self.weight_offset_dict = weight_offset_dict
        self.fuse_add = fuse_add


class ConvertLinearParams:
    def __init__(self,
                 onnx_model,
                 input_scale,
                 input_offset,
                 weight_scale,
                 weight_offset,
                 quant_weight):
        self.onnx_model = onnx_model
        self.input_scale = input_scale
        self.input_offset = input_offset
        self.weight_scale = weight_scale
        self.weight_offset = weight_offset
        self.quant_weight = quant_weight


def calculate_int_weight(weight_initializer,
                         weight_scale,
                         weight_offset,
                         bits):
    data_type = get_np_datatype().get(str(weight_initializer.data_type))
    weight = np.frombuffer(weight_initializer.raw_data, dtype=data_type)
    weight = weight.reshape(weight_initializer.dims)

    if len(weight.shape) == 4:
        weight_scale = weight_scale.view(-1, 1, 1, 1)
        weight_offset = weight_offset.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(weight.shape) == 2:
        weight_scale = weight_scale.view(-1, 1)
        weight_offset = weight_offset.view(-1, 1)

    if (np.array(weight_scale) != 0).all():
        quant_weight = weight / np.array(weight_scale) + np.array(weight_offset)
    else:
        raise ZeroDivisionError

    max_range = 2 ** (bits - 1)
    quant_weight = np.clip(quant_weight, -max_range, max_range - 1).astype(np.int8)
    new_weight = onnx.helper.make_tensor(weight_initializer.name,
                                         onnx.TensorProto.INT8,
                                         quant_weight.shape,
                                         quant_weight)
    return new_weight, quant_weight


def calculate_int_bias(bias_initializer, deq_scale):
    data_type = get_np_datatype().get(str(bias_initializer.data_type))
    bias = np.frombuffer(bias_initializer.raw_data, dtype=data_type)

    if bias.shape[0] == 0:
        bias = np.array(bias_initializer.float_data)

    bias = bias.reshape(bias_initializer.dims)
    left_limit = -pow(1.0 * 2, 32 - 1)
    right_limit = pow(1.0 * 2, 32 - 1) - 1
    quant_bias = np.round(np.true_divide(bias, deq_scale))
    # check the quant_bias in range of int32
    quant_bias = quant_bias.reshape(-1)
    cmp_ans = np.add(quant_bias < left_limit, quant_bias > right_limit)
    if cmp_ans.any():
        invalid_value = quant_bias[np.argmax(cmp_ans)]
        raise RuntimeError('Do bias quantize failed.')
    quant_bias = quant_bias.reshape(bias.shape).astype(np.int32)
    new_bias = onnx.helper.make_tensor(bias_initializer.name,
                                       onnx.TensorProto.INT32,
                                       quant_bias.shape,
                                       quant_bias)
    return new_bias


def convert_conv(graph, fp_node, quant_param, quant_index):
    input_scale = quant_param["input_scale"]
    weight_scale = quant_param["weight_scale"]
    quant_weight = quant_param["quant_weight"]

    attributes = {}
    # ATTRBUTE TYPE: INT: 2 INTS: 7 FLOAT: 1
    for attr in fp_node.attribute:
        if attr.type == 2:
            attributes[attr.name] = attr.i
        elif attr.type == 7:
            attributes[attr.name] = attr.ints
        elif attr.type == 1:
            attributes[attr.name] = attr.f
    deq_scale = np.multiply(np.array(weight_scale),
                        np.array(input_scale)).reshape([-1])
    node_input, node_output = transform_conv_input(graph, fp_node, quant_weight, deq_scale, quant_index)
    onnx_node_conv = helper.make_node(
        "Conv", inputs=node_input, name=fp_node.name,
        outputs=node_output,
        dilations=attributes.get("dilations"),
        group=attributes.get("group"),
        kernel_shape=attributes.get("kernel_shape"),
        strides=attributes.get("strides"),
        pads=attributes.get("pads")
    )

    return onnx_node_conv


def convert_matmul(graph, fp_node, quant_param, quant_index):
    quant_weight = quant_param["quant_weight"]
    
    # the weight of matmul should be transposed
    quant_weight = quant_weight.transpose(1, 0)
    node_input, node_output = transform_matmul_input(graph, fp_node, quant_weight, quant_index)
    onnx_node_matmul = helper.make_node(
        "MatMul", inputs=node_input, name=fp_node.name,
        outputs=node_output
    )

    return onnx_node_matmul


def convert_quant(quant_param, node_input, weight_name, quant_index):
    input_scale = quant_param["input_scale"]
    input_offset = quant_param["input_offset"]
    quant_input = [node_input[0]]
    quant_output = tansform_quant_output(node_input, weight_name, quant_index)

    if input_scale == 0:
        raise ZeroDivisionError
    else:
        onnx_node_quant = helper.make_node(
            "AscendQuant", inputs=quant_input,
            outputs=quant_output,
            scale=(1 / input_scale).item(),
            offset=input_offset.item(),
            quant_bit=8

        )
    return onnx_node_quant


def convert_dequant(graph, quant_param, deq_inputs, deq_outputs):
    input_scale = quant_param["input_scale"]
    weight_scale = quant_param["weight_scale"]
    x_scale = np.array(input_scale) * np.array(weight_scale)
    packed_weight_np_data = x_scale.squeeze()
    float32_scale_deq = np.array(packed_weight_np_data, np.float32)
    uint32_scale_deq = np.frombuffer(float32_scale_deq, np.uint32)
    uint64_result = np.zeros(float32_scale_deq.shape, np.uint64)
    # per-tensor
    if len(uint64_result.shape) == 0:
        uint64_result = np.expand_dims(uint64_result, axis=0)
    uint64_result |= np.uint64(uint32_scale_deq)
    deq_scale_name = deq_inputs[0] + "_x_scale"
    deq_scale = onnx.helper.make_tensor(deq_scale_name,
                                        onnx.TensorProto.UINT64,
                                        uint64_result.shape,
                                        uint64_result)
    graph.initializer.append(deq_scale)
    deq_inputs.append(deq_scale_name)

    onnx_node_as_dequant = helper.make_node(
        "AscendDequant", inputs=deq_inputs, 
        outputs=deq_outputs,
    )
    return onnx_node_as_dequant


def convert_add(graph, add_node, quant_param, node_output, quant_index):

    input_scale = quant_param["input_scale"]
    weight_scale = quant_param["weight_scale"]

    bias_index = None
    bias_name = add_node.input[0] 
    for idx, item in enumerate(graph.initializer):
        if bias_name == item.name:
            bias_index = idx

    deq_scale = np.multiply(np.array(weight_scale),
                            np.array(input_scale)).reshape([-1])

    new_bias = calculate_int_bias(graph.initializer[bias_index], deq_scale)
    del graph.initializer[bias_index]
    graph.initializer.append(new_bias)

    add_inputs = add_node.input
    add_outputs = [add_node.input[1]]
    add_inputs = transform_add_input(add_inputs, quant_index)
    # NOTE: bias must be the second input
    add_true_inputs = [add_inputs[1], add_inputs[0]]
    onnx_node_add = helper.make_node(
        "Add", inputs=add_true_inputs, 
        name=add_node.name,
        outputs=add_outputs,
    )
    return onnx_node_add


def find_index(nodes, weight_name):
    for idx, item in enumerate(nodes):
        if weight_name in item.input or weight_name + ".weight" in item.input:
            return idx
    raise LookupError


def init_quant_param(params):
    weight_name = params.weight_name
    input_scale_dict = params.input_scale_dict
    input_offset_dict = params.input_offset_dict
    weight_scale_dict = params.weight_scale_dict
    weight_offset_dict = params.weight_offset_dict
    quant_weight_dict = params.quant_weight_dict

    quant_param = {}
    quant_param["input_scale"] = input_scale_dict[weight_name]
    quant_param["input_offset"] = input_offset_dict[weight_name]
    quant_param["weight_scale"] = weight_scale_dict[weight_name].cpu()
    quant_param["weight_offset"] = weight_offset_dict[weight_name].cpu()
    quant_param["quant_weight"] = quant_weight_dict[weight_name].cpu()

    # some layers don't quantize input
    if quant_param.get("input_scale") is None:
        quant_param["input_scale"] = np.array([1])
        quant_param["input_offset"] = np.array([0])
    elif isinstance(quant_param.get("input_scale"), torch.Tensor):
        input_offset = quant_param.get("input_offset").float().cpu()
        input_scale = quant_param.get("input_scale").cpu()
        quant_param["input_offset"] = input_offset
        quant_param["input_scale"] = input_scale
    elif isinstance(quant_param.get("input_scale"), type(np.array([1]))):
        input_offset = quant_param.get("input_offset").astype(np.float32).cpu()
        input_scale = quant_param.get("input_scale").astype(np.float32).cpu()
        quant_param["input_offset"] = input_offset
        quant_param["input_scale"] = input_scale

    return quant_param


def quantize_model_deploy(graph, params):
    quantized_weight_name = params.quantized_weight_name
    quant_weight_dict = params.quant_weight_dict
    input_scale_dict = params.input_scale_dict
    input_offset_dict = params.input_offset_dict
    weight_scale_dict = params.weight_scale_dict
    weight_offset_dict = params.weight_offset_dict
    fuse_add = params.fuse_add

    logging.info("before graph initializer:%d", len(graph.initializer))
    delete_same_bias_name(graph)
    logging.info("graph initializer:%d", len(graph.initializer))

    nodes = graph.node
    quant_index = -1
    for weight_name in quantized_weight_name:
        quant_index = get_quant_index(quant_index)
        logging.info("enter this quantized module:%r", weight_name)

        index = find_index(nodes, weight_name)
        fp_node = nodes[index]
        node_input = copy.deepcopy(fp_node.input)
        node_output = copy.deepcopy(fp_node.output)

        init_quant_params = InitQuantParams(
            weight_name=weight_name,
            input_scale_dict=input_scale_dict,
            input_offset_dict=input_offset_dict,
            weight_scale_dict=weight_scale_dict,
            weight_offset_dict=weight_offset_dict,
            quant_weight_dict=quant_weight_dict
        )
        quant_param = init_quant_param(init_quant_params)

        onnx_node_quant = convert_quant(quant_param, node_input, weight_name, quant_index)
        onnx_node_add = None
        if fp_node.op_type == "Conv":
            onnx_node_compute = convert_conv(graph, fp_node, quant_param, quant_index)
        elif fp_node.op_type == "MatMul":
            onnx_node_compute = convert_matmul(graph, fp_node, quant_param, quant_index)
            onnx_node_add = None
            next_node = nodes[index + 1]
            if fuse_add and fp_node.op_type == "MatMul" and next_node.op_type == "Add":
                add_output = copy.deepcopy(next_node.output)
                onnx_node_add = convert_add(graph, next_node, quant_param, node_output, quant_index)
        
        if onnx_node_add is not None:
            deq_inputs = copy.deepcopy(onnx_node_add.output)
            deq_outputs = add_output
        else:
            deq_inputs = copy.deepcopy(onnx_node_compute.output)
            deq_outputs = node_output

        onnx_node_dequant = convert_dequant(graph, quant_param, deq_inputs, deq_outputs)

        nodes.remove(nodes[index])
        if onnx_node_add is not None:
            nodes.remove(nodes[index])
            nodes.insert(index, onnx_node_quant)
            nodes.insert(index + 1, onnx_node_compute)
            nodes.insert(index + 2, onnx_node_add)
            nodes.insert(index + 3, onnx_node_dequant)
        else:
            nodes.insert(index, onnx_node_quant)
            nodes.insert(index + 1, onnx_node_compute)
            nodes.insert(index + 2, onnx_node_dequant)
    return


def find_bias_index(quant_nodes, bias_name):
    all_bias = []
    for idx, item in enumerate(quant_nodes):
        if bias_name in item.input:
            all_bias.append(idx)
    return all_bias


def find_quant_node(quant_nodes, input_name):
    for _, item in enumerate(quant_nodes):
        if item.op_type == "Conv":
            continue
        if input_name in item.output:
            return item.input[1]
    raise LookupError


def find_all_quant_nodes(all_bias, nodes):
    all_quant_node = []
    for index in all_bias:
        if len(nodes[index].input) < 2:
            continue
        try:
            quant_weight_name = find_quant_node(nodes, nodes[index].input[1])
        except LookupError as err:
            logging.info("quant node not finded: %r", nodes[index].input[1])
        else:
            all_quant_node.append(quant_weight_name)
    return all_quant_node


def get_linear_quant_map(onnx_model, weight_scale):
    graph = onnx_model.graph
    nodes = onnx_model.graph.node
    bias_index = []
    all_quant_nodes = []
    quant_map = {}

    for _, item in enumerate(graph.initializer):
        name_value = ".".join(item.name.split(".")[:-1])
        if name_value not in weight_scale.keys():
            continue

        bias_name = "%s.bias" % (name_value)
        if weight_scale[name_value] is not None:
            all_bias = find_bias_index(nodes, bias_name)
            all_quant_node = find_all_quant_nodes(all_bias, nodes)
            bias_index.append(all_bias)
            all_quant_nodes.append(all_quant_node)
            if len(all_quant_node) != 0:
                quant_map[name_value] = all_quant_node
        else:
            logging.info("no value:%r, len of quant_map:%d, quant_map:%r", bias_name, len(quant_map), quant_map)
    return quant_map


def get_new_dict(param_dict, quant_map):
    new_dict = {}
    for key, value in param_dict.items():
        if key in quant_map.keys():
            all_node = quant_map[key]
            for item in all_node:
                new_dict[item] = value
        else:
            new_dict[key] = value
    return new_dict


def convert_linear_params(params):
    onnx_model = params.onnx_model
    input_scale = params.input_scale
    input_offset = params.input_offset
    weight_scale = params.weight_scale
    weight_offset = params.weight_offset
    quant_weight = params.quant_weight

    quant_map = get_linear_quant_map(onnx_model, weight_scale)
    weight_scale = get_new_dict(weight_scale, quant_map)
    weight_offset = get_new_dict(weight_offset, quant_map)
    input_scale = get_new_dict(input_scale, quant_map)
    input_offset = get_new_dict(input_offset, quant_map)
    quant_weight = get_new_dict(quant_weight, quant_map)
    return input_scale, input_offset, weight_scale, weight_offset, quant_weight
