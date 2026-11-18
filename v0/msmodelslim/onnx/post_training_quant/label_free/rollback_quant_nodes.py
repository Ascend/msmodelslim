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
import math

import onnx
import onnxruntime
import numpy as np

from msmodelslim import logger
from msmodelslim.onnx.post_training_quant.util import gen_model_inputs, get_model_params


def match_activations(float_model_path, quant_model_path, quantized_nodes=None, quant_config=None):
    float_model = onnx.load(float_model_path)
    float_session = get_session_for_intermediate_output(float_model)
    inputs = float_session.get_inputs()
    input_dict = gen_model_inputs(inputs, quant_config)

    float_inter_output = float_session.run([], input_dict)

    quant_model = onnx.load(quant_model_path)
    quant_model = preprocess_quant_model(quant_model)
    quant_session = get_session_for_intermediate_output(quant_model)
    quant_inter_output = quant_session.run([], input_dict)

    float_inter_output_name = []
    for node in float_model.graph.node:
        if node.name in quantized_nodes:
            float_inter_output_name.extend(node.output)

    quant_inter_output_name = []
    for node in quant_model.graph.node:
        if node.op_type == "DequantizeLinear":
            quant_inter_output_name.extend(node.output)

    inter_output_names = list(set(float_inter_output_name).intersection(set(quant_inter_output_name)))
    float_inter_output_dict = collect_inter_output_array(float_inter_output, float_session.get_outputs())
    quant_inter_output_dict = collect_inter_output_array(quant_inter_output, quant_session.get_outputs())

    activation_errors = {}
    for name in inter_output_names:
        err = compute_mse(float_inter_output_dict.get(name), quant_inter_output_dict.get(name))
        activation_errors.setdefault(name, err)
    return activation_errors


def preprocess_quant_model(model):
    """
    On some cpus, weights are quantized symmetrically in per-channel situation,
    their zero-point are not all zeros. We have to manually change it to zero.
    """
    graph = model.graph
    param_dict = get_model_params(graph.initializer)
    new_initializer = []
    for _, param in enumerate(graph.initializer):
        param_name = param.name
        param_value = param_dict.get(param_name)
        if ".weight_zero_point" in param_name and isinstance(param_value, np.ndarray) \
                and not param_value.any() == 0:
            param_value[param_value != 0] = 0
            new_param = onnx.helper.make_tensor(param_name,
                                                param.data_type,
                                                param_value.shape,
                                                param_value)
            logger.debug("%r param is set to zero.", param_name)
            new_initializer.append(new_param)
        else:
            new_initializer.append(param)
    new_graph = onnx.helper.make_graph(nodes=graph.node, name="ascend_onnx", inputs=graph.input,
                                       outputs=graph.output, initializer=new_initializer)
    onnx_model = onnx.helper.make_model(new_graph, opset_imports=[onnx.helper.make_opsetid(domain="", version=11)])
    return onnx_model


def get_rollback_nodes(model_path, activations_errors, amp_nums):
    model = onnx.load(model_path)
    sorted_errs = sorted(activations_errors.items(), key=lambda x: -x[1])
    activation_names = [sorted_errs[i][0] for i in range(amp_nums)]
    rollback_nodes = []
    for node in model.graph.node:
        for activation_name in activation_names:
            if activation_name in node.output:
                rollback_nodes.append(node.name)
    return rollback_nodes


def collect_inter_output_array(inter_outputs_arr, inter_outputs):
    inter_array = {}
    for i, inter_output in enumerate(inter_outputs):
        inter_array.setdefault(inter_output.name, inter_outputs_arr[i])
    return inter_array


def get_session_for_intermediate_output(model):
    for node in model.graph.node:
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    ort_session = onnxruntime.InferenceSession(model.SerializeToString())
    return ort_session


def compute_mse(array1, array2):
    if isinstance(array1, np.ndarray):
        xlist = [array1]
    else:
        xlist = array1
    if isinstance(array2, np.ndarray):
        ylist = [array2]
    else:
        ylist = array2
    if len(xlist) != len(ylist):
        raise RuntimeError("Unequal number of tensors to compare!")

    left = np.concatenate(xlist).flatten()
    right = np.concatenate(ylist).flatten()
    err = np.linalg.norm(left - right)
    return err
