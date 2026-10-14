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
# pylint: disable=no-member

import mindspore as ms  # noqa: F401
import shutil  # noqa: F401
import sys  # noqa: F401
import os  # noqa: F401
from msmodelslim.mindspore.llm_ptq import QuantConfig as SparseQuantConfig


def test_sparse_quant_config():
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

    quant_config = SparseQuantConfig(disable_names=disable_names, fraction=fraction)

    assert isinstance(quant_config, SparseQuantConfig)
    assert quant_config.fraction == fraction
    assert len(quant_config.disable_names) == 10
