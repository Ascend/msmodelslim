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

from mindspore.nn import Conv2d
from mindspore.train import serialization

from ascend_utils.mindspore.quant.ptq_quant.convert_deploy import convert_to_inference_network
from ascend_utils.mindspore.quant.ptq_quant.process_utils import convert_equact_to_relu
from ascend_utils.common.security import get_valid_write_path
from ascend_utils.common.security.mindspore import check_mindspore_cell
from ascend_utils.common.security.mindspore import check_mindspore_input
from msmodelslim import logger


def save_model(file_name, quantized_model, *input_data, file_format='AIR'):
    """
    Save the quantized model in the specified file format.
    :param file_name: The file name where the quantized model is saved.
    :param quantized_model: The quantized model to be saved.
    :param input_data: The input data is used to calibrate the quantized model to be exported.
    :param file_format: The file format in which the quantized model is saved.
    :raises ValueError: Error occurred when accessing the wrong file_name/quantized_model/input_data.
    """
    get_valid_write_path(file_name)
    check_mindspore_cell(quantized_model)
    check_mindspore_input(input_data)
    q_model = convert_equact_to_relu(quantized_model)
    convert_to_inference_network(q_model)

    for name, cell in q_model.cells_and_names():
        if isinstance(cell, Conv2d):  # shared conv2d situation
            cell.update_parameters_name(name)

    if file_format not in ['AIR', 'MINDIR']:
        raise ValueError("For 'save_model', 'file_format' must be one of ['AIR', 'MINDIR']")

    logger.info("Start to export %s file...", file_format)
    try:
        if isinstance(input_data, (list, tuple)):
            serialization.export(q_model,
                                 *input_data,
                                 file_name=file_name, file_format=file_format)
        else:
            serialization.export(q_model,
                                 input_data,
                                 file_name=file_name, file_format=file_format)
    except Exception as exception:
        logger.warning(exception)
        logger.warning("Fail to export %s file: %s", file_format, file_name)
    else:
        logger.info("Finish to export %s file: %s", file_format, file_name)