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
from typing import Callable, Tuple
from collections import defaultdict

from msmodelslim import logger


def _get_target_id(target: Tuple):
    container, attr_name = target
    return f"{id(container)}:{attr_name}"


class HookManager:
    """
    The core feature of this manager is parameter normalization: regardless of whether the target function
    is called with positional or keyword arguments, the registered hook functions will receive a `kwargs`
    dictionary containing all parameters. This greatly simplifies hook function implementation.
    """

    def __init__(self):
        self.before_hooks = defaultdict(list)
        self.after_hooks = defaultdict(list)
        self.error_hooks = defaultdict(list)
        self.original_functions = {}
        self.hooked_targets = {}

    def add_before_hook(self, target, hook_fn: Callable):
        """
        Add a hook to execute before function call.
        Hook function signature: hook_fn(func, kwargs)
        """
        self._add_hook(target, hook_fn, self.before_hooks)

    def add_after_hook(self, target, hook_fn: Callable):
        """
        Add a hook to execute after successful function return.
        Hook function signature: hook_fn(func, kwargs, result)
        """
        self._add_hook(target, hook_fn, self.after_hooks)

    def add_error_hook(self, target, hook_fn: Callable):
        """
        Add a hook to execute when function throws an exception.
        Hook function signature: hook_fn(func, kwargs, error)
        """
        self._add_hook(target, hook_fn, self.error_hooks)

    def restore_target(self, target):
        target_id = _get_target_id(target)
        if target_id in self.hooked_targets:
            original_func = self.original_functions[target_id]
            container, attr_name = self.hooked_targets[target_id]
            setattr(container, attr_name, original_func)

            # Clear all hooks for this target
            self.before_hooks.pop(target_id, None)
            self.after_hooks.pop(target_id, None)
            self.error_hooks.pop(target_id, None)

            del self.hooked_targets[target_id]
            del self.original_functions[target_id]
            logger.debug(f"Restored target: {target}")
        else:
            logger.debug(f"Target {target} not found in hooked_targets.")

    def restore_all(self):
        for target_id in list(self.hooked_targets.keys()):
            self.restore_target(self.hooked_targets[target_id])
        logger.debug("Restored all hooks.")

    def _add_hook(self, target, hook_fn: Callable, hooks_dict):
        target_id = _get_target_id(target)
        if target_id not in self.hooked_targets:
            self._wrap_target(target, target_id)
        hooks_dict[target_id].append(hook_fn)
        logger.debug(f"Added hook to {target}. Hooks registered: {len(hooks_dict[target_id])}")

    def _wrap_target(self, target: Tuple, target_id: str):
        container, attr_name = target
        original_func = getattr(container, attr_name)

        # Get function signature in advance
        can_inspect = True
        try:
            func_sig = inspect.signature(original_func)
        except (TypeError, ValueError):
            can_inspect = False

        self.original_functions[target_id] = original_func
        self.hooked_targets[target_id] = target

        @functools.wraps(original_func)
        def wrapped_function(*args, **kwargs):
            call_kwargs = {}
            if can_inspect:
                try:
                    # Bind all parameters to signature, normalize to kwargs
                    bound_args = func_sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    call_kwargs = bound_args.arguments
                except TypeError:
                    # Fallback if binding fails (e.g., for some variable argument functions)
                    call_kwargs = kwargs.copy()
                    call_kwargs['_raw_args'] = args
            else:
                # Best effort attempt for functions that can't be introspected
                call_kwargs = kwargs.copy()
                if args:
                    call_kwargs['_raw_args'] = args

            # Before hooks
            for hook in self.before_hooks.get(target_id, []):
                hook(original_func, call_kwargs)

            try:
                # Execute original function
                result = original_func(*args, **kwargs)
            except Exception as e:
                # Error hooks
                for hook in self.error_hooks.get(target_id, []):
                    hook(original_func, call_kwargs, e)
                raise

            # After hooks
            for hook in self.after_hooks.get(target_id, []):
                result = hook(original_func, call_kwargs, result)

            return result

        setattr(container, attr_name, wrapped_function)


# Global hook manager instance
_manager = HookManager()


def add_before_hook(target, hook_fn: Callable):
    _manager.add_before_hook(target, hook_fn)


def add_after_hook(target, hook_fn: Callable):
    _manager.add_after_hook(target, hook_fn)


def add_error_hook(target, hook_fn: Callable):
    _manager.add_error_hook(target, hook_fn)


def restore_target(target):
    _manager.restore_target(target)


def restore_all_hooks():
    _manager.restore_all()
