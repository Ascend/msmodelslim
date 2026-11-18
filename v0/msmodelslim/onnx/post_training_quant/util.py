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
from enum import Enum
import random

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto
from onnx.mapping import STORAGE_TENSOR_TYPE_TO_FIELD

from msmodelslim import logger

random.seed(1)


LF_QUANTIZED_OP_TYPES = ["Conv", "Gemm", "MatMul"]  # quantized operator types in label-free mode

INPUT_DTYPE_DICT = {
    "tensor(float)": np.float32,
    "tensor(int64)": np.int64 if hasattr(np, "int64") else np.int,
    "tensor(int32)": np.int32,

}

ONNX_MODEL_SUFFIX = ".onnx"
ONNX_VERSION_LIST = list(range(6, 15))


class OnnxAttrbuteType(Enum):
    FLOAT = 1
    INT = 2
    STRING = 3
    FLOATS = 6
    INTS = 7
    STRINGS = 8


def get_quantized_nodes(model_path, quantize_nodes=None, exclude_nodes=None):
    quantize_nodes = quantize_nodes or []
    exclude_nodes = exclude_nodes or []
    model = onnx.load(model_path)
    params = set(param.name for param in model.graph.initializer)
    quantizable_nodes = []
    for node in model.graph.node:
        if quantize_nodes and node.name not in quantize_nodes:
            continue

        if node.name in exclude_nodes:
            continue

        # Quantize only those with a single input, means only operation which A is activation and B is weight.
        # For directly called functions like `F.conv2d`, will just skip.
        if node.op_type in LF_QUANTIZED_OP_TYPES and len(set(node.input) - params) == 1:
            quantizable_nodes.append(node.name)

    return quantizable_nodes


def optimize_graph(input_model_path, output_model_path):
    sess_options = ort.SessionOptions()
    # Set graph optimization level
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.optimized_model_filepath = output_model_path
    _ = ort.InferenceSession(input_model_path, sess_options)


def check_model(model_path):
    try:
        session = ort.InferenceSession(model_path)
    except Exception as exception:
        logger.error(exception)
    else:
        inputs = session.get_inputs()
        if len(inputs) == 0:
            raise ValueError("Invalid model file, please check it.")
        if hasattr(inputs[0], "shape") and len(inputs[0].shape) == 0:
            raise ValueError("The input %r shape of model is invalid, please check it." % inputs[0].name)


def get_node_attributes(node):
    attributes = {}
    for attr in node.attribute:
        if attr.type == OnnxAttrbuteType.INT.value:
            attributes[attr.name] = attr.i
        elif attr.type == OnnxAttrbuteType.INTS.value:
            attributes[attr.name] = attr.ints
        elif attr.type == OnnxAttrbuteType.FLOAT.value:
            attributes[attr.name] = attr.f
        elif attr.type == OnnxAttrbuteType.STRING.value:
            attributes[attr.name] = attr.s
    return attributes


def gen_model_inputs(inputs, quant_config):
    if quant_config and quant_config.input_shape and len(inputs) != len(quant_config.input_shape):
        raise ValueError(
            f"The input shape user specified ({quant_config.input_shape}) "
            f"does not align with the model inputs ({inputs})"
        )

    input_dict = {}
    for idx, input_x in enumerate(inputs):
        shape = list(input_x.shape)
        if quant_config and quant_config.is_dynamic_shape:     # dynamic shape
            if quant_config.input_shape[idx] is None:
                raise ValueError('For model with dynamic shape, please specify the shape of input'
                                 'to construct calib data')
            shape = list(quant_config.input_shape[idx])
        else:
            if shape and is_dynamic_batch(shape):
                shape[0] = 1
        array = np.random.random(shape).astype(INPUT_DTYPE_DICT.get(input_x.type))
        input_dict.setdefault(input_x.name, array)
    return input_dict


def check_and_get_calib_data(model_inputs, calib_data=None, quant_cfg=None) -> [dict]:
    valid_data = []
    if not calib_data:  # generate data randomly when calib_data is []
        logger.info('Generate random data for calibration')
        return [gen_model_inputs(inputs=model_inputs, quant_config=quant_cfg)]

    for index, data in enumerate(calib_data):
        data_list = data if isinstance(data, list) else [data]
        if len(data_list) != len(model_inputs):
            logger.warning("The number of %r data records in the calib_data is not equal to "
                           "the input of the model.", index)
            continue
        per_batch_data = {}
        for input_data, input_x in zip(data_list, model_inputs):
            if check_input_data(input_x, input_data, quant_cfg):
                per_batch_data.setdefault(input_x.name, input_data)
            else:
                logger.warning("The %r data records in calib_data is not valid", index)
        if len(per_batch_data.values()) == len(model_inputs):
            valid_data.append(per_batch_data)

    if not valid_data:
        logger.warning("There is no valid data in calib_data, we will generate data randomly.")
        return [gen_model_inputs(inputs=model_inputs, quant_config=quant_cfg)]
    return valid_data


def check_input_data(input_x, input_data, quant_cfg=None):
    if not isinstance(input_data, np.ndarray):
        return False
    input_x_shape = list(input_x.shape)
    input_data_shape = list(input_data.shape)
    if input_x_shape and not isinstance(input_data, np.ndarray):
        logger.warning("The input %r type is not valid.", input_x.name)
        return False
    if INPUT_DTYPE_DICT.get(input_x.type) != input_data.dtype.type:  # check dtype
        logger.warning("The input %r data type is not valid.", input_x.name)
        return False
    if input_x_shape and not check_input_shape(input_x_shape, input_data_shape, quant_cfg):
        return False
    return True


def check_input_shape(input_x_shape, input_data_shape, quant_cfg=None):
    if quant_cfg and quant_cfg.is_dynamic_shape:
        # dynamic shape
        if not check_dynamic_input_shape(input_x_shape, input_data_shape):
            return False
    else:
        if input_x_shape and is_dynamic_batch(input_x_shape):
            # dynamic batch
            if input_x_shape[1:] != input_data_shape[1:]:
                return False
        elif input_x_shape != input_data_shape:   # check shape
            return False
    return True


def check_dynamic_input_shape(input_x_shape, input_data_shape):
    for model_shape_i, input_shape_i in zip(input_x_shape, input_data_shape):
        if isinstance(model_shape_i, int) and model_shape_i > 0 and \
                model_shape_i != input_shape_i:
            return False
    return True


def is_dynamic_batch(shape):
    if not isinstance(shape[0], int) or (isinstance(shape[0], int) and shape[0] < 0):
        return True
    return False


def parse_tensor_proto(param: TensorProto):
    if param.dims:  # array
        value = onnx.numpy_helper.to_array(param)
    else:
        field = STORAGE_TENSOR_TYPE_TO_FIELD.get(param.data_type)
        if field and hasattr(param, field) and getattr(param, field):
            value = getattr(param, field)[0]
        else:
            value = onnx.numpy_helper.to_array(param).item()
    return value


def get_model_params(initializer):
    param_dict = {}
    for param in initializer:
        param_dict[param.name] = parse_tensor_proto(param)
    return param_dict


def inference(model, input_x=None):
    """
    onnx model inference
    """
    session = ort.InferenceSession(model.SerializeToString())
    if input_x is None:
        inputs = session.get_inputs()
        input_x = gen_model_inputs(inputs, None)
    output = session.run([], input_x)
    return output
