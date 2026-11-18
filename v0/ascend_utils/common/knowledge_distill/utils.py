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


def replace_module(network, name, module, backend="mindspore"):
    tokens = name.split('.')
    sub_tokens = tokens[:-1]
    cur_network = network
    for token in sub_tokens:
        if not hasattr(cur_network, token):
            return
        cur_network = getattr(cur_network, token)
    setattr(cur_network, tokens[-1], module)
    if backend == "mindspore":
        module.update_parameters_name(name + '.')
    if tokens[-1].isdigit():
        idx = int(tokens[-1])
        cur_network[idx] = module
