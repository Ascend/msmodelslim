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
# pylint: disable=pointless-string-statement

"""
Usage:
TEST_PATH=test/fuzz/onnx_ptq_api/run_quantize/file_name
python3 -m coverage run ${TEST_PATH}/fuzz_test.py ${TEST_PATH}/samples/ -atheris_runs=1000
"""

import sys  # noqa: E402
import os  # noqa: E402

import atheris  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from onnxruntime import quantization as _  # noqa: E402, F401
import sympy as _  # noqa: E402, F401, F811

# Need to import these first, otherwise `atheris.instrument_imports` will take a long time

with atheris.instrument_imports():
    from resources.sample_net_torch import TestOnnxQuantModel
    from msmodelslim.onnx.post_training_quant import QuantConfig, run_quantize
    from msmodelslim import logger

ONNX_MODEL_PATH = "./test.onnx"


def get_calib_data1():
    data_list = {"input": np.random.randn(3, 32, 32).astype(np.float32)}
    return [data_list]


def get_calib_data2():
    data_list = [
        np.random.randn(3, 32, 32).astype(np.float32),
        np.random.randn(3, 32, 32).astype(np.float32),
        [np.random.randn(3, 32, 32).astype(np.float32)],
    ]
    return data_list


def get_calib_data3():
    return [
        1,
    ]


def get_calib_data4():
    return []


@atheris.instrument_func
def fuzz_test(input_bytes):
    quant_model_path = input_bytes.decode('utf-8', 'ignore').strip()
    logger.info("quant_model_path: %s", quant_model_path)

    func_list = [get_calib_data1, get_calib_data2, get_calib_data3, get_calib_data4]
    get_calib_data = np.random.choice(func_list)
    logger.info("get_calib_data func: %s", get_calib_data)

    config_list = [
        QuantConfig(quant_mode=0),
        QuantConfig(quant_mode=0, calib_data=get_calib_data()),
        QuantConfig(calib_data=get_calib_data(), amp_num=1),
        QuantConfig(calib_data=get_calib_data()),
    ]
    quant_config = np.random.choice(config_list)

    try:
        run_quantize(ONNX_MODEL_PATH, quant_model_path, quant_config)
    except ValueError as value_error:
        logger.error(value_error)
    except TypeError as type_error:
        logger.error(type_error)


def gen_onnx_model():
    model = TestOnnxQuantModel()
    input_x = torch.randn((1, 3, 32, 32))
    torch.onnx.export(model, input_x, ONNX_MODEL_PATH, input_names=['input'], output_names=['output'])


if __name__ == '__main__':
    TEST_SAVE_PATH = "automl_fuzz_test_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    gen_onnx_model()
    atheris.Setup(sys.argv, fuzz_test)
    atheris.Fuzz()
    if os.path.exists(ONNX_MODEL_PATH):
        os.remove(ONNX_MODEL_PATH)
    if os.path.exists(TEST_SAVE_PATH):
        os.removedirs(TEST_SAVE_PATH)
