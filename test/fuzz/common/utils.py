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
# pylint: disable=unnecessary-comprehension

import random


def random_change_dict_value(ori_dict, random_value):
    '''
    Recursively modify a random value of dict
    '''
    key_list = [key for key in ori_dict.keys()]
    change_key = random.choice(key_list)
    change_value = ori_dict[change_key]
    if isinstance(change_value, dict):
        random_change_dict_value(change_value, random_value)
    elif isinstance(change_value, list) and change_value:
        change_index = random.randint(0, len(change_value) - 1)
        if isinstance(change_value[change_index], dict):
            random_change_dict_value(change_value[change_index], random_value)
        else:
            change_value[change_index] = random_value
    else:
        ori_dict[change_key] = random_value
