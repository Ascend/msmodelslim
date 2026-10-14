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
# pylint: disable=logging-fstring-interpolation

import functools
import inspect
from typing import Callable, Tuple, Dict, Any

from msmodelslim import logger


def _get_target_id(target: Tuple):
    container, attr_name = target
    return f"{id(container)}:{attr_name}"


def _create_kwargs_wrapper(original_func: Callable, replacement_function: Callable) -> Callable:
    can_inspect = True
    try:
        func_sig = inspect.signature(original_func)
    except (TypeError, ValueError):
        can_inspect = False

    @functools.wraps(original_func)
    def kwargs_wrapper(*args, **kwargs):
        call_kwargs = {}
        if can_inspect:
            try:
                bound_args = func_sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                call_kwargs = bound_args.arguments
            except TypeError:
                call_kwargs = kwargs.copy()
                call_kwargs['_raw_args'] = args
        else:
            call_kwargs = kwargs.copy()
            if args:
                call_kwargs['_raw_args'] = args
        return replacement_function(**call_kwargs)

    return kwargs_wrapper


class FunctionHijacker:
    def __init__(self):
        self.original_functions: Dict[str, Any] = {}
        self.hijacked_targets: Dict[str, Tuple] = {}

    def hijack_function(self, target: Tuple, replacement_function: Callable):
        target_id = _get_target_id(target)
        container, attr_name = target

        if target_id not in self.original_functions:
            original_func = getattr(container, attr_name)
            self.original_functions[target_id] = original_func
            self.hijacked_targets[target_id] = target

            wrapped_replacement = _create_kwargs_wrapper(original_func, replacement_function)

            setattr(container, attr_name, wrapped_replacement)
            logger.debug(f"Hijacked function: {target}")
        else:
            wrapped_replacement = _create_kwargs_wrapper(self.original_functions[target_id], replacement_function)
            setattr(container, attr_name, wrapped_replacement)
            logger.debug(f"Updated hijacked function: {target}")

    def restore_function(self, target: Tuple):
        target_id = _get_target_id(target)
        if target_id in self.original_functions:
            container, attr_name = target
            original_func = self.original_functions[target_id]
            setattr(container, attr_name, original_func)
            del self.original_functions[target_id]
            del self.hijacked_targets[target_id]
            logger.debug(f"Restored function: {target}")
        else:
            logger.debug(f"Target {target} not found in hijacked_targets.")

    def restore_all(self):
        for target_id in list(self.hijacked_targets.keys()):
            target = self.hijacked_targets[target_id]
            self.restore_function(target)
        logger.debug("Restored all hijacked functions.")

    def get_original_function(self, target: Tuple) -> Callable:
        target_id = _get_target_id(target)
        return self.original_functions.get(target_id)


_hijacker = FunctionHijacker()


def hijack_function(target: Tuple, replacement_function: Callable):
    _hijacker.hijack_function(target, replacement_function)


def restore_function(target: Tuple):
    _hijacker.restore_function(target)


def restore_all_hijacked():
    _hijacker.restore_all()


def get_original_function(target: Tuple) -> Callable:
    return _hijacker.get_original_function(target)
