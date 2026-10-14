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

import os
import sys  # noqa: F401
import shutil

from resources.sample_net_mindspore import MsSparseModel
import numpy as np  # noqa: F401

import mindspore as ms
from mindformers.models.configuration_utils import PretrainedConfig

from msmodelslim.mindspore.llm_ptq import QuantConfig, Calibrator


def test_mindspore_sparse_run():
    TEST_SAVE_PATH = "msmodelslim_llm_ptq_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)

    model_config = PretrainedConfig()
    model = MsSparseModel(model_config)
    tmp_weight_ckpt = "./test_ms_model.ckpt"
    tmp_sparse_weight_ckpt = "./test_ms_sparse_model.ckpt"
    ms.save_checkpoint(model.parameters_dict(), tmp_weight_ckpt)

    fraction = 0.011
    disable_names = [
        "lm_head",
        "model.layers.1.feed_forward.w1",
        "model.layers.1.feed_forward.w11",
        "model.layers.1.feed_forward.w3",
        "model.layers.2.feed_forward.w1",
        "model.layers.2.feed_forward.w11",
        "model.layers.2.feed_forward.w3",
        "model.layers.3.feed_forward.w1",
        "model.layers.3.feed_forward.w11",
        "model.layers.3.feed_forward.w3",
    ]

    quant_config = QuantConfig(disable_names=disable_names, fraction=fraction)
    calibrator = Calibrator(
        quant_config,
        model,
        tmp_weight_ckpt,
    )

    calibrator.run()

    if os.path.exists(tmp_sparse_weight_ckpt):
        os.remove(tmp_sparse_weight_ckpt)

    calibrator.save(tmp_sparse_weight_ckpt)

    fake_model = calibrator.fake_quantize_model()

    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)

    assert isinstance(fake_model, ms.nn.Cell)
