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

import numpy as np
import onnx
from onnx import helper

from msmodelslim.onnx.squant_ptq.onnx_ptq_kia.weight_transform_onnx import (
    transform_conv_input,
    transform_gemm_input,
    transform_matmul_input,
    transform_add_input,
    tansform_quant_output,
    get_quant_index,
    get_np_datatype)
from msmodelslim import logger


class QuantParamsDict:
    def __init__(self, input_scale, input_offset, weight_scale, weight_offset, quant_weight, bias_name, node_name):
        self.input_scale = input_scale
        self.input_offset = input_offset
        self.weight_scale = weight_scale
        self.weight_offset = weight_offset
        self.quant_weight = quant_weight
        self.bias_name = bias_name
        self.node_name = node_name


def _calculate_int_bias(bias_initializer, deq_scale, quant_index):
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
        logger.error("invalid_value: %s", invalid_value)
        raise RuntimeError('Do bias quantize failed.')
    quant_bias = quant_bias.reshape(bias.shape).astype(np.int32)
    new_bias = onnx.helper.make_tensor(
        bias_initializer.name + "_node_" + str(quant_index),
        onnx.TensorProto.INT32,
        quant_bias.shape,
        quant_bias)
    return new_bias


def _get_node_type_attributes(node):
    attributes = {}
    # ATTRBUTE TYPE: INT: 2 INTS: 7 FLOAT: 1
    for attr in node.attribute:
        if attr.type == 2:
            attributes[attr.name] = attr.i
        elif attr.type == 7:
            attributes[attr.name] = attr.ints
        elif attr.type == 1:
            attributes[attr.name] = attr.f
    return attributes


def _convert_conv(graph, fp_node, quant_param, quant_index):
    input_scale = quant_param["input_scale"]
    weight_scale = quant_param["weight_scale"]
    quant_weight = quant_param["quant_weight"]
    bias_name = quant_param["bias_name"]

    attributes = _get_node_type_attributes(fp_node)
    deq_scale = np.multiply(np.array(weight_scale),
                            np.array(input_scale)).reshape([-1])
    node_input, node_output = transform_conv_input(graph, fp_node,
                                                   quant_weight, bias_name,
                                                   deq_scale, quant_index)
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


def _convert_matmul(graph, fp_node, quant_param, quant_index):
    quant_weight = quant_param["quant_weight"]

    node_input, node_output = transform_matmul_input(graph, fp_node,
                                                     quant_weight,
                                                     quant_index)
    onnx_node_matmul = helper.make_node(
        "MatMul", inputs=node_input, name=fp_node.name,
        outputs=node_output
    )

    return onnx_node_matmul


def _convert_gemm(graph, fp_node, quant_param, quant_index):
    input_scale = quant_param["input_scale"]
    weight_scale = quant_param["weight_scale"]
    quant_weight = quant_param["quant_weight"]
    bias_name = quant_param["bias_name"]

    attributes = _get_node_type_attributes(fp_node)
    deq_scale = np.multiply(np.array(weight_scale),
                            np.array(input_scale)).reshape([-1])

    if attributes.get("transB") == 1:
        quant_weight = quant_weight.transpose(1, 0)

    node_input, node_output = transform_gemm_input(graph, fp_node,
                                                   quant_weight,
                                                   bias_name,
                                                   deq_scale,
                                                   quant_index)

    onnx_node_matmul = helper.make_node(
        "Gemm", inputs=node_input, name=fp_node.name,
        outputs=node_output,
        alpha=attributes.get("alpha"),
        beta=attributes.get("beta"),
        transB=0
    )

    return onnx_node_matmul


def _convert_quant(quant_param, node_input, weight_name, quant_index):
    input_scale = quant_param["input_scale"]
    input_offset = quant_param["input_offset"]
    node_name = quant_param["node_name"]
    quant_input = [node_input[0]]
    quant_output = tansform_quant_output(node_input, weight_name,
                                         quant_index)

    quant_name = node_name + "_quant" + str(quant_index)
    onnx_node_quant = helper.make_node(
        op_type="AscendQuant", inputs=quant_input, name=quant_name,
        outputs=quant_output,
        scale=(1 / input_scale).item(),
        offset=input_offset.item(),
        quant_bit=8
    )
    return onnx_node_quant


def _convert_dequant(graph, quant_param, deq_inputs, deq_outputs, dequant_index):
    input_scale = quant_param["input_scale"]
    weight_scale = quant_param["weight_scale"]
    node_name = quant_param["node_name"]
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
    dequant_name = node_name + "_dequant" + str(dequant_index)
    onnx_node_as_dequant = helper.make_node(name=dequant_name,
        op_type="AscendDequant", inputs=deq_inputs,
        outputs=deq_outputs,
    )
    return onnx_node_as_dequant


def _convert_add(graph, add_node, quant_param, node_output, quant_index):
    input_scale = quant_param["input_scale"]
    weight_scale = quant_param["weight_scale"]

    bias_index = None
    # NOTE：部分bias name在input0，部分在input1
    bias_name = None
    input_name = None
    for idx, item in enumerate(graph.initializer):
        if item.name == add_node.input[0]:
            bias_index = idx
            bias_name = add_node.input[0]
            input_name = add_node.input[1]
            break
        elif item.name == add_node.input[1]:
            bias_index = idx
            bias_name = add_node.input[1]
            input_name = add_node.input[0]
            break
    deq_scale = np.multiply(np.array(weight_scale),
                            np.array(input_scale)).reshape([-1])
    new_bias = _calculate_int_bias(graph.initializer[bias_index], deq_scale,
                                  quant_index)
    graph.initializer.append(new_bias)

    add_outputs = [input_name]
    bias_name = transform_add_input(bias_name, quant_index)
    input_name = transform_add_input(input_name, quant_index)
    # NOTE: bias must be the second input
    add_true_inputs = [input_name, bias_name]
    onnx_node_add = helper.make_node(
        "Add", inputs=add_true_inputs,
        name=add_node.name,
        outputs=add_outputs,
    )
    return onnx_node_add


def _find_index(nodes, weight_name):
    for idx, item in enumerate(nodes):
        if weight_name in item.input or weight_name + ".weight" in item.input:
            return idx
    raise LookupError


def _init_quant_param(weight_name, quant_params_dict: QuantParamsDict):
    quant_param = {}
    quant_param["input_scale"] = quant_params_dict.input_scale[weight_name]
    quant_param["input_offset"] = quant_params_dict.input_offset[weight_name]
    quant_param["weight_scale"] = quant_params_dict.weight_scale[weight_name]
    quant_param["weight_offset"] = quant_params_dict.weight_offset[weight_name]
    quant_param["quant_weight"] = quant_params_dict.quant_weight[weight_name]
    quant_param["bias_name"] = quant_params_dict.bias_name[weight_name]
    quant_param["node_name"] = quant_params_dict.node_name[weight_name]

    # some layers don't quantize input
    if quant_param.get("input_scale") is None:
        quant_param["input_scale"] = np.array([1])
        quant_param["input_offset"] = np.array([0])
    elif isinstance(quant_param.get("input_scale"), type(np.array([1]))):
        input_offset = quant_param.get("input_offset").astype(np.float32)
        input_scale = quant_param.get("input_scale").astype(np.float32)
        quant_param["input_offset"] = input_offset
        quant_param["input_scale"] = input_scale

    return quant_param


def quantize_model_deploy(graph, quantized_weight_name,
                          quant_params_dict: QuantParamsDict, fuse_add=False):
    nodes = graph.node
    quant_index = -1
    dequant_index = 0
    for weight_name in quantized_weight_name:
        quant_index = get_quant_index(quant_index)
        index = _find_index(nodes, weight_name)
        fp_node = nodes[index]
        node_input = copy.deepcopy(fp_node.input)
        node_output = copy.deepcopy(fp_node.output)

        quant_param = _init_quant_param(weight_name, quant_params_dict)

        onnx_node_quant = _convert_quant(quant_param, node_input, weight_name, quant_index)
        onnx_node_add = None
        fp_node_op_type = fp_node.op_type
        if fp_node_op_type in ["Conv", "MatMul", "Gemm"]:
            logger.info(f"deploy quantized weight {weight_name} to type {fp_node_op_type}")
        if fp_node_op_type == "Conv":
            onnx_node_compute = _convert_conv(graph, fp_node, quant_param, quant_index)
        elif fp_node_op_type == "MatMul":
            onnx_node_compute = _convert_matmul(graph, fp_node, quant_param, quant_index)
            next_node = nodes[index + 1]
            if fuse_add and next_node.op_type == "Add":
                add_output = copy.deepcopy(next_node.output)
                onnx_node_add = _convert_add(graph, next_node, quant_param, node_output, quant_index)
        elif fp_node_op_type == "Gemm":
            onnx_node_compute = _convert_gemm(graph, fp_node, quant_param, quant_index)

        if onnx_node_add is not None:
            deq_inputs = copy.deepcopy(onnx_node_add.output)
            deq_outputs = add_output
        else:
            deq_inputs = copy.deepcopy(onnx_node_compute.output)
            deq_outputs = node_output
        dequant_index += 1
        onnx_node_dequant = _convert_dequant(graph, quant_param, deq_inputs, deq_outputs, dequant_index)

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


