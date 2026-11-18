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
# pylint: disable=chained-comparison

import re


STR_WHITE_LIST_REGEX = re.compile(r"[^_A-Za-z0-9\"'><=\[\])(,}{: /.~-]")


def type_to_str(value_type):
    return ' or '.join([ii.__name__ for ii in value_type]) if isinstance(value_type, tuple) else value_type.__name__


def check_type(value, value_type, param_name="value", additional_check_func=None, additional_msg=None):
    if not isinstance(value, value_type):
        raise TypeError('{} must be {}, not {}.'.format(param_name, type_to_str(value_type), type(value).__name__))
    if additional_check_func is not None:
        additional_msg = (" " + additional_msg) if additional_msg else ""
        if isinstance(value, (list, tuple)):
            if not all(list(map(additional_check_func, value))):
                raise ValueError("Element in {} is invalid.".format(param_name) + additional_msg)
        elif not additional_check_func(value):
            raise ValueError("Value of {} is invalid.".format(param_name) + additional_msg)
    is_int_when_tuple = isinstance(value_type, tuple) and int in value_type and bool not in value_type
    if value_type == int or is_int_when_tuple:  # noqa: E721
        if isinstance(value, bool):
            raise TypeError('{} must be {}, not {}.'.format(param_name, type_to_str(value_type), type(value).__name__))


def check_character(value, param_name="value"):
    max_depth = 100

    def check_character_recursion(inner_value, depth=0):
        if isinstance(inner_value, str):
            if re.search(STR_WHITE_LIST_REGEX, inner_value):
                raise ValueError("{} contains invalid characters.".format(param_name))
        elif isinstance(inner_value, (list, tuple)):
            if depth > max_depth:
                raise ValueError("Recursion depth of {} exceeds limitation.".format(param_name))

            for sub_value in inner_value:
                check_character_recursion(sub_value, depth=depth + 1)

    check_character_recursion(value)


def check_dict_character(dict_value, key_max_len=512, param_name="dict"):
    max_depth = 100

    def check_dict_character_recursion(inner_dict_value, depth=0):
        check_type(inner_dict_value, dict, param_name=param_name)

        for key, value in inner_dict_value.items():
            key = str(key)
            check_character(key, param_name=f"{param_name} key")
            if key_max_len > 0 and len(key) > key_max_len:
                raise ValueError("Length of {} key exceeds limitation {}.".format(param_name, key_max_len))
            if isinstance(value, dict):
                if depth > max_depth:
                    raise ValueError("Recursion depth of {} exceeds limitation.".format(param_name))
                check_dict_character_recursion(value, depth=depth + 1)
            else:
                check_character(value, param_name=param_name)

    check_dict_character_recursion(dict_value)
