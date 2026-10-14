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
# pylint: disable=anomalous-backslash-in-string,pointless-string-statement

"""
Usage:
python3 -m coverage run $(pwd)/fuzz_test.py $(pwd)/samples/ -atheris_runs=100  # execute code
coverage report -i  # coverage rate
coverage html -d foo -i  # coverage rate + code execution in a html
"""

import sys  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402

import atheris  # noqa: E402

from test.fuzz.common.utils import random_change_dict_value  # noqa: E402

from msmodelslim.common.prune.transformer_prune.prune_model import PruneConfig  # noqa: E402


@atheris.instrument_func
def fuzz_test(input_bytes):
    fuzz_value = input_bytes.decode('utf-8', 'ignore').strip()

    patten = "uniter\.encoder\.encoder\.blocks\.(\d+)\."
    layer_id_map = {0: 1, 1: 3, 2: 5, 3: 7, 4: 9, 5: 11}

    test_case_id = random.randint(0, 1)
    if test_case_id == 0:
        random_change_dict_value(layer_id_map, fuzz_value)
    if test_case_id == 1:
        patten = fuzz_value

    config = PruneConfig()
    try:
        config.add_blocks_params(patten, layer_id_map)
    except ValueError as value_error:
        logging.error(value_error)
    except TypeError as type_error:
        logging.error(type_error)


if __name__ == '__main__':
    TEST_SAVE_PATH = "automl_fuzz_test_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)

    atheris.Setup(sys.argv, fuzz_test)
    atheris.Fuzz()
