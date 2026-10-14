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
TEST_PATH=$PWD/test/fuzz/low_rank_decompose/from_fixed/channel_fixed/
python3 -m coverage run ${TEST_PATH}/fuzz_test.py ${TEST_PATH}/samples/ -atheris_runs=1000
"""

import sys  # noqa: E402
import logging  # noqa: E402

import atheris  # noqa: E402

with atheris.instrument_imports():
    from msmodelslim.pytorch.low_rank_decompose import Decompose
    from resources.sample_net_torch import LrdSampleNetwork


@atheris.instrument_func
def fuzz_test(input_bytes):
    fdp = atheris.FuzzedDataProvider(input_bytes)
    channel_fixed = fdp.ConsumeInt(sys.maxsize)
    logging.info("channel_fixed: %s", channel_fixed)

    model = LrdSampleNetwork()
    decomposer = Decompose(model, "low_rank_decompose_test.json")

    try:
        decomposer.from_fixed(channel_fixed)
    except ValueError as value_error:
        logging.error(value_error)
        return

    decomposer.from_file()


if __name__ == '__main__':
    import os

    TEST_SAVE_PATH = "automl_fuzz_test_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)

    atheris.Setup(sys.argv, fuzz_test)
    atheris.Fuzz()
