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
# pylint: disable=consider-using-from-import

import functools
import inspect
import fnmatch
from typing import List, Dict, Any, Union, Callable, Tuple
import contextvars

import torch
import torch.nn as nn

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.manager import TimestepManager
from msmodelslim.utils.logging import logger

MAX_RECURSION_DEPTH = 20


class InputCapture:
    """Handles capturing and storing function inputs and outputs."""

    _captured_inputs_var = contextvars.ContextVar("captured_inputs", default=[])

    @classmethod
    def reset(cls) -> None:
        """Reset all captured inputs."""
        cls._captured_inputs_var.set([])

    @classmethod
    def get_all(cls) -> List[Dict[str, Any]]:
        """Get all captured inputs."""
        return cls._captured_inputs_var.get()

    @classmethod
    def add_record(cls, record: Dict[str, Any]) -> None:
        """Add a new record to the captured inputs."""
        inputs = cls._captured_inputs_var.get()
        inputs.append(record)
        cls._captured_inputs_var.set(inputs)

    @classmethod
    def capture_forward_inputs(
        cls,
        func: Callable,
        capture_mode: str = 'args',
    ) -> Callable:
        """
        Decorator to capture inputs to a forward function.

        Args:
            func: Forward function to decorate
            capture_mode: 'args', 'kwargs', 'timestep'

        Returns:
            Wrapped function
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature and bind arguments
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Handle 'self' for methods
            is_method = 'self' in sig.parameters
            captured_args = list(bound.args[1:]) if is_method else list(bound.args)

            captured_kwargs = bound.arguments.copy()
            if is_method and 'self' in captured_kwargs:
                del captured_kwargs['self']

            # Apply capture mode
            if capture_mode == 'args':
                captured_kwargs = {}
                record = captured_args
            elif capture_mode == 'kwargs':
                captured_args = []
                record = captured_kwargs
            elif capture_mode == 'timestep':
                record = {
                    "tag": "",
                    "timestep_idx": TimestepManager.get_timestep_idx(),
                    "module_name": func.__qualname__,
                    "args": captured_args,
                    "kwargs": captured_kwargs,
                }
            else:
                raise ValueError(f"Invalid capture_mode: {capture_mode}. Must be 'args' or 'kwargs' or 'timestep'")

            # Execute original function
            result = func(*args, **kwargs)

            # Store record
            record = to_device(record, device='cpu')
            cls.add_record(record)

            return result

        return wrapper


class DumperManager(nn.Module):
    """Module that listens to and captures forward pass inputs and outputs."""

    def __init__(
        self,
        module: nn.Module,
        capture_mode: str = 'args',
    ):
        """
        Initialize a listener for the given module.

        Args:
            module: Module to listen to
            capture_mode: 'args' or 'kwargs' or 'timestep'
        """
        super().__init__()
        self.module = module
        self.capture_mode = capture_mode
        self.old_forward = None

        if capture_mode not in {'args', 'kwargs', 'timestep'}:
            raise ValueError(f"Invalid capture_mode: {capture_mode}. Must be 'args' or 'kwargs' or 'timestep'")

        self._add_hook(self.module)

    def save(self, path: str = '__output.pth') -> List[Dict[str, Any]]:
        """Save captured data and restore original forward method."""
        data = InputCapture.get_all()
        torch.save(data, path)

        # Restore original forward method
        if self.old_forward:
            self.module.forward = self.old_forward
            self.old_forward = None

        logger.info('Captured data saved to: %r', path)
        return data

    def reset(self) -> None:
        """Reset captured inputs."""
        InputCapture.reset()

    def _add_hook(self, module: nn.Module) -> Callable:
        """Add forward hook to the module."""
        self.old_forward = module.forward
        wrapper = InputCapture.capture_forward_inputs(
            self.old_forward,
            capture_mode=self.capture_mode,
        )
        module.forward = wrapper
        return wrapper


def get_rank():
    """
    Get the rank of the current process.

    Returns:
        int: Non-negative rank (in default group) if distributed is initialized; -1 otherwise.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return -1


def get_disable_layer_names(
    model: nn.Module, layer_include: Union[List[str], Tuple[str], str], layer_exclude: Union[List[str], Tuple[str], str]
) -> List[str]:
    """
    Get the names of layers to be disabled based on inclusion and exclusion patterns using fnmatch.

    Args:
        model: The neural network module
        layer_include: Patterns for layers to include. Can be a string, list or tuple of strings.
        layer_exclude: Patterns for layers to exclude. Can be a string, list or tuple of strings.

    Returns:
        List of layer names that should be disabled for quantization.
    """
    # Convert single string patterns to list for uniform processing
    if isinstance(layer_include, str):
        layer_include = [layer_include]
    if isinstance(layer_exclude, str):
        layer_exclude = [layer_exclude]

    all_layer_names = []
    quant_layer_names = set()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            all_layer_names.append(name)

        # Check inclusion patterns
        if layer_include and not any(fnmatch.fnmatch(name, pattern) for pattern in layer_include):
            continue
        # Check exclusion patterns
        if layer_exclude and any(fnmatch.fnmatch(name, pattern) for pattern in layer_exclude):
            continue

        quant_layer_names.add(name)

    disable_layer_names = [name for name in all_layer_names if name not in quant_layer_names]
    return disable_layer_names


def to_device(data, device, depth=0):
    """recursive function to move data to the specified device"""
    if depth > MAX_RECURSION_DEPTH:
        raise RecursionError(f"Maximum recursion depth {MAX_RECURSION_DEPTH} exceeded")

    if isinstance(data, dict):
        return {k: to_device(v, device, depth=depth + 1) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(item, device, depth=depth + 1) for item in data]
    elif isinstance(data, tuple):
        return tuple(to_device(item, device, depth=depth + 1) for item in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


def get_rank_suffix_file(base_name, ext, is_distributed, rank):
    """
    生成带rank后缀的文件名，分布式环境下添加_rank标识，非分布式环境直接使用基础名称

    参数:
        base_name (str): 文件名基础部分（不含后缀）
        ext (str): 文件后缀（不含小数点）
        is_distributed (bool): 是否为分布式环境
        rank (int): 当前进程的rank值（分布式环境下有效）

    返回:
        str: 处理后的完整文件名（含后缀）
    """
    if is_distributed:
        return f"{base_name}_{rank}.{ext}"
    return f"{base_name}.{ext}"
