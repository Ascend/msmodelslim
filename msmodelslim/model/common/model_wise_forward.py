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
# pylint: disable=consider-merging-isinstance

from typing import List, Union, Tuple, Dict

import torch

from msmodelslim.core.base.protocol import ProcessRequest


def model_wise_forward_func(
    model: torch.nn.Module,
    inputs: Union[List, Tuple, Dict],
):
    yield ProcessRequest(
        "",
        model,
        inputs if isinstance(inputs, list) or isinstance(inputs, tuple) else [inputs],
        inputs if isinstance(inputs, dict) else {},
    )


def model_wise_visit_func(
    model: torch.nn.Module,
):
    yield ProcessRequest("", model, tuple(), {})
