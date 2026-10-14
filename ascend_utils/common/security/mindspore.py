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

import mindspore as ms

MAX_DEPTH_THRESHOLD = 100


def check_mindspore_cell(cell):
    if not isinstance(cell, ms.nn.Cell):
        raise TypeError("model must be a mindspore.nn.Cell instance. Not {}".format(type(cell)))


def check_mindspore_input(input_data):
    """
    Use recursion to check whether the input_data is mindspore.Tensor

    Args:
        input_data: can be list/tuple/Tensor
    """
    if not input_data or len(input_data) == 0:
        raise ValueError("input data cannot be empty")

    def recursive_check_mindspore_input(cur_data, depth=0):
        if depth >= MAX_DEPTH_THRESHOLD:
            raise ValueError("input data nested too deeply")
        depth = depth + 1
        if isinstance(cur_data, (list, tuple)):
            for value in cur_data:
                recursive_check_mindspore_input(value, depth)
        elif not isinstance(cur_data, ms.Tensor):
            raise TypeError("input data must be mindspore.Tensor. Not {}".format(type(cur_data)))

    recursive_check_mindspore_input(input_data)
