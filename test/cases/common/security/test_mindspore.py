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

from resources.sample_net_mindspore import TestNetMindSpore

import mindspore as ms
import numpy as np

from ascend_utils.common.security.mindspore import check_mindspore_cell, check_mindspore_input


def test_check_mindspore_module_given_valid_when_any_then_pass():
    model = TestNetMindSpore(class_num=10)
    check_mindspore_cell(model)


def test_check_mindspore_input_given_valid_when_any_then_pass():
    data = np.random.randn(3, 4).astype(np.float32)
    tensor = [ms.Tensor(data, dtype=ms.float32)]
    check_mindspore_input(tensor)
