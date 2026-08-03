#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

import os
from typing import Any, List, Dict, Callable
import functools
import inspect

import torch
from torch import nn

from ascend_utils.common.security.pytorch import safe_torch_load
from msmodelslim.utils.security import get_valid_read_path, get_write_directory, SafeWriteUmask
from msmodelslim.utils.exception import SchemaValidateError, SecurityError
from msmodelslim.utils.exception_decorator import exception_handler
from msmodelslim.utils.logging import get_logger

MAX_RECURSION_DEPTH = 20
MAX_READ_FILE_SIZE_32G = 34359738368  # 32G, 32 * 1024 * 1024 * 1024


@exception_handler("", err_cls=SecurityError, ms_err_cls=SecurityError)
def load_cached_data(pth_file_path, generate_func, model, dump_config):
    """内部缓存加载函数"""
    try:
        # 检查文件是否存在
        if not os.path.exists(pth_file_path):
            raise FileNotFoundError(f"File {repr(pth_file_path)} does not exist")

        # 加载pth文件
        pth_file_path = get_valid_read_path(pth_file_path, size_max=MAX_READ_FILE_SIZE_32G)
        data = safe_torch_load(pth_file_path)
        get_logger().info("Successfully loaded data from %r", pth_file_path)
        return data
    except FileNotFoundError as e:
        # 检测目录，若不存在则创建
        pth_file_dir = get_write_directory(os.path.dirname(pth_file_path))
        pth_file_name = os.path.basename(pth_file_path)
        pth_file_path = os.path.join(pth_file_dir, pth_file_name)
        get_logger().info("Failed to load %r: %s, will regenerate", pth_file_path, e)

        # 触发缓存未命中时的处理逻辑
        # 配置dump参数
        dumper_manager = DumperManager(model, capture_mode=dump_config.capture_mode)

        # 执行推理获取激活值
        generate_func()

        # 返回dump的数据
        dumper_manager.save(pth_file_path)

        data = safe_torch_load(pth_file_path)
        get_logger().info("Successfully dumped and loaded data from %r", pth_file_path)
        return data


@exception_handler("", err_cls=SecurityError, ms_err_cls=SecurityError)
def load_cached_data_for_models(
    pth_file_path_list: Dict[str, str], generate_func: Callable, models: Dict[str, nn.Module], dump_config
) -> Dict[str, Any]:
    """内部缓存加载函数，兼容MoE结构模型"""
    calib_data = {}
    to_regenerate = False

    for expert_name, _ in models.items():
        if os.path.exists(pth_file_path_list[expert_name]):
            calib_data[expert_name] = safe_torch_load(pth_file_path_list[expert_name])
            get_logger().info("Loaded calib data from %s", pth_file_path_list[expert_name])
        else:
            calib_data[expert_name] = None
            to_regenerate = True
            break

    #  如果任一缓存不存在，重新生成
    if to_regenerate:
        get_logger().info("======== Calib data missing, regenerating... ========")

        dumper = {}
        for expert_name, _ in models.items():
            dumper[expert_name] = DumperManager(models[expert_name], capture_mode=dump_config.capture_mode)

        generate_func()

        for expert_name, _ in models.items():
            calib_data[expert_name] = dumper[expert_name].save(pth_file_path_list[expert_name])

        get_logger().info("======== Calib data generated successfully ========")

    return calib_data


class InputCapture:
    """Handles capturing and storing function inputs and outputs."""

    def __init__(self):
        self._captured_inputs: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Reset all captured inputs."""
        self._captured_inputs = []

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all captured inputs."""
        return self._captured_inputs

    def add_record(self, record: Dict[str, Any]) -> None:
        """Add a new record to the captured inputs."""
        self._captured_inputs.append(record)

    def capture_forward_inputs(
        self,
        func: Callable,
        capture_mode: str = 'args',
    ) -> Callable:
        """
        Decorator to capture inputs to a forward function.

        Args:
            func: Forward function to decorate
            capture_mode: 'args'

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
            # 可拓展其他模式
            if capture_mode == 'args':
                captured_kwargs = {}
                record = captured_args
            else:
                raise SchemaValidateError("Invalid capture_mode: %r. Must be 'args' " % capture_mode)

            # Execute original function
            result = func(*args, **kwargs)

            # Store record
            record = to_device(record, device='cpu')
            self.add_record(record)

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
            capture_mode: 'args'
        """
        super().__init__()
        self.module = module
        self.capture_mode = capture_mode
        self.input_capture = InputCapture()
        self.old_forward = None

        if capture_mode not in {'args'}:
            raise SchemaValidateError("Invalid capture_mode: %r. Must be 'args' " % capture_mode)

        self._add_hook(self.module)

    def save(self, path: str = '__output.pth') -> List[Dict[str, Any]]:
        """Save captured data and restore original forward method."""
        data = self.input_capture.get_all()
        try:
            with SafeWriteUmask():
                torch.save(data, path)
        finally:
            # Restore original forward even if save fails (disk full, permission, etc.)
            if self.old_forward:
                self.module.forward = self.old_forward
                self.old_forward = None

        get_logger().info('Captured data saved to: %r', path)
        return data

    def reset(self) -> None:
        """Reset captured inputs."""
        self.input_capture.reset()

    def _add_hook(self, module: nn.Module) -> Callable:
        """Add forward hook to the module."""
        self.old_forward = module.forward
        wrapper = self.input_capture.capture_forward_inputs(
            self.old_forward,
            capture_mode=self.capture_mode,
        )
        module.forward = wrapper
        return wrapper


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
