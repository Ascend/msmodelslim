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
import os
from os import path

import onnx
from onnx import helper

from ascend_utils.common.security import check_type, get_valid_read_path, get_valid_write_path, SafeWriteUmask
from msmodelslim.onnx.post_training_quant.util import optimize_graph, check_model, ONNX_VERSION_LIST, \
    ONNX_MODEL_SUFFIX
from msmodelslim import logger
from msmodelslim.onnx.post_training_quant.label_free.quantize_tool \
    import quantize_model as quantize_model_lf
from msmodelslim.onnx.post_training_quant.data_free.quantize_tool import \
    quantize_model as quantize_model_df
from msmodelslim.onnx.post_training_quant.config import QuantConfig


def run_quantize(input_model_path, output_model_path, quant_config):
    check_type(quant_config, QuantConfig, param_name="quant_config")
    input_model_path = get_valid_read_path(input_model_path, extensions=ONNX_MODEL_SUFFIX)
    check_model(input_model_path)
    output_model_path = get_valid_write_path(output_model_path, extensions=ONNX_MODEL_SUFFIX,
                                             check_user_stat=True)

    model_name, _ = path.splitext(path.split(input_model_path)[1])

    if quant_config.is_optimize_graph:
        logger.info("Start to optimize graph.")
        optimize_model_path = path.join(path.dirname(input_model_path), model_name + "_optimize.onnx")
        optimize_model_path = get_valid_write_path(optimize_model_path, extensions=ONNX_MODEL_SUFFIX)
        with SafeWriteUmask():
            optimize_graph(input_model_path, optimize_model_path)
            convert_version(optimize_model_path, optimize_model_path)
        logger.debug("Finish to optimize graph, the path is: %s", optimize_model_path)
        to_quantized_model_path = optimize_model_path
    else:
        to_quantized_model_path = input_model_path

    logger.info("Start to quantize model.")
    quantize_model_func = quantize_model_lf if quant_config.quant_mode else quantize_model_df
    with SafeWriteUmask():
        try:
            quantize_model_func(to_quantized_model_path, output_model_path, quant_config)
        except Exception as e:
            raise Exception("Please check your model and config.", e) from e

    if quant_config.is_optimize_graph:
        logger.debug("Remove temp optimized model file: %s", optimize_model_path)
        if path.exists(optimize_model_path):
            os.remove(optimize_model_path)
    logger.info("Finish to quantize model, the path is: %s", output_model_path)


def convert_version(input_model_path, output_model_path, version=11):
    input_model_path = get_valid_read_path(input_model_path, extensions=ONNX_MODEL_SUFFIX)
    output_model_path = get_valid_write_path(output_model_path, extensions=ONNX_MODEL_SUFFIX,
                                             check_user_stat=True)
    if version not in ONNX_VERSION_LIST:
        raise ValueError("version must be one of %s, please check it.", ONNX_VERSION_LIST)
    model = onnx.load(input_model_path)
    graph = model.graph

    graph = onnx.helper.make_graph(nodes=graph.node, name="ascend_onnx", inputs=graph.input,
                                   outputs=graph.output, initializer=graph.initializer)
    onnx_model = onnx.helper.make_model(graph, opset_imports=[helper.make_opsetid(domain="", version=version)])
    with SafeWriteUmask():
        onnx.save(onnx_model, output_model_path)
