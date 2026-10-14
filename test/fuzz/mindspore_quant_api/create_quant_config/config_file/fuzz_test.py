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
# pylint: disable=pointless-string-statement,unspecified-encoding

"""
Usage:
TEST_PATH=$PWD/test/fuzz/mindspore_quant_api/create_quant_config/config_file
python3 -m coverage run ${TEST_PATH}/fuzz_test.py ${TEST_PATH}/samples/ -atheris_runs=1000
"""

import sys  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402

import atheris  # noqa: E402


@atheris.instrument_func
def fuzz_test(input_bytes):
    # Put mindspore related imports inside, avoiding ARM atheris `cannot allocate memory in static TLS block` error.
    import mindspore as ms

    ms.set_context(device_target='CPU')  # NPU will be rather slow

    from test.resources import sample_net_mindspore as sample_net
    from msmodelslim.mindspore.quant.ptq_quant.create_config import create_quant_config

    config_file = input_bytes.decode('utf-8', 'ignore').strip()
    logging.info("config_file: %s", config_file)

    model = sample_net.SampleModel()

    try:
        create_quant_config(config_file=config_file, model=model)
    except ValueError as value_error:
        logging.error(value_error)
        return

    with open(config_file, "r") as read_file:
        _ = json.load(read_file)


if __name__ == '__main__':
    import os

    TEST_SAVE_PATH = "automl_fuzz_test_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)

    atheris.Setup(sys.argv, fuzz_test)
    atheris.Fuzz()
