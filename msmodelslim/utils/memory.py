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

from typing import Any, Optional

import torch
from torch import nn

from msmodelslim.utils.logging import get_logger

MAX_RECURSION_DEPTH = 10


def _align_input_to_module_device(module: Optional[nn.Module], input_data: Any, device: Any = None) -> Any:
    """
    将模块的输入数据对齐到模块所在的设备上的内部函数。

    参数:
        module: 目标模块
        input_data: 输入数据，通常是 (args, kwargs) 的元组
        device: 目标设备

    返回:
        对齐后的输入数据
    """
    if module is None or input_data is None:
        return input_data

    if device:
        module_device = device
    else:
        # 获取模块所在的设备
        try:
            module_device = next(module.parameters()).device
        except StopIteration:
            # 如果模块没有参数，返回原始输入
            return input_data

    # 统计移动的数据总量
    total_moved_bytes = 0
    moved_tensors_count = 0

    def _to_device(data: Any, device: torch.device, depth: int = 0) -> Any:
        """递归地将数据移动到指定设备"""
        nonlocal total_moved_bytes, moved_tensors_count
        if depth > MAX_RECURSION_DEPTH:
            raise RecursionError(f"Maximum recursion depth {MAX_RECURSION_DEPTH} exceeded")

        if isinstance(data, dict):
            return {k: _to_device(v, device, depth=depth + 1) for k, v in data.items()}
        elif isinstance(data, list):
            return [_to_device(item, device, depth=depth + 1) for item in data]
        elif isinstance(data, tuple):
            return tuple(_to_device(item, device, depth=depth + 1) for item in data)
        elif isinstance(data, torch.Tensor):
            # 检查张量是否已经在目标设备上
            if data.device != device:
                # 统计移动的数据量
                tensor_bytes = data.element_size() * data.numel()
                total_moved_bytes += tensor_bytes
                moved_tensors_count += 1
                return data.to(device)
            else:
                return data
        else:
            return data

    result = _to_device(input_data, module_device)

    # 在 debug 日志中提示移动的数据总量
    if moved_tensors_count > 0:
        get_logger().debug(
            "Device alignment hook for %r: moved %d tensors to device %r, total size: %r, "
            "device memory: allocated=%r, reserved=%r",
            module.__class__.__name__,
            moved_tensors_count,
            module_device,
            format_memory_size(total_moved_bytes),
            format_memory_size(get_device_allocated_memory()),
            format_memory_size(get_device_reserved_memory()),
        )
    else:
        get_logger().debug(
            "Device alignment hook for %r: no tensors moved (already on device %r), "
            "device memory: allocated=%r, reserved=%r",
            module.__class__.__name__,
            module_device,
            format_memory_size(get_device_allocated_memory()),
            format_memory_size(get_device_reserved_memory()),
        )

    return result


def align_input_to_module_device_hook(module: Optional[nn.Module], args: Any, kwargs: Any) -> Any:
    """
    将模块的输入数据对齐到模块所在的设备上的 hook 函数。

    该函数作为 forward pre-hook 使用，在模块 forward 前自动将输入数据移动到模块所在设备。

    参数:
        module: 目标模块
        *args: 位置参数（当with_kwargs=False时，第一个参数是input_data）
        **kwargs: 关键字参数

    返回:
        对齐后的输入数据
    """
    input_data = (
        args,
        kwargs,
    )
    aligned_input_data = _align_input_to_module_device(module, input_data)
    return aligned_input_data[0], aligned_input_data[1]


_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _cast_float_to_dtype(data: Any, target_dtype: torch.dtype) -> Any:
    """递归地将浮点张量转换到 ``target_dtype``；int 与其它类型保持不变。

    与 ``_align_input_to_module_device`` 中的 ``_to_device`` 对称：只动浮点张量，
    整型张量（如 ``input_ids``）原样返回。
    """
    if isinstance(data, torch.Tensor):
        if data.is_floating_point() and data.dtype != target_dtype:
            return data.to(target_dtype)
        return data
    if isinstance(data, dict):
        return {k: _cast_float_to_dtype(v, target_dtype) for k, v in data.items()}
    if isinstance(data, list):
        return [_cast_float_to_dtype(v, target_dtype) for v in data]
    if isinstance(data, tuple):
        return tuple(_cast_float_to_dtype(v, target_dtype) for v in data)
    return data


def align_input_dtype_to_module_hook(module: Optional[nn.Module], args: Any, kwargs: Any) -> Any:
    """Forward pre-hook：将模块的浮点输入对齐到模块参数的 dtype。

    与 ``align_input_to_module_device_hook`` 配对使用：前者对齐设备，本函数对齐 dtype。
    伪量化 decoder Linear 反量化输出 float32，而共享 norm / lm_head 权重为 BFloat16，
    原生 ``model.forward`` 不做 dtype 转换，会导致 ``F.linear`` 报 dtype 不匹配。
    整型输入（如 ``input_ids``）通过 ``is_floating_point`` 跳过。

    Args:
        module: 目标模块
        args: 位置参数
        kwargs: 关键字参数

    Returns:
        对齐 dtype 后的 (args, kwargs)
    """
    if module is None:
        return args, kwargs
    try:
        target_dtype = next(module.parameters()).dtype
    except StopIteration:
        # 模块无参数，返回原始输入
        return args, kwargs
    if target_dtype not in _FLOAT_DTYPES:
        return args, kwargs
    return _cast_float_to_dtype(args, target_dtype), _cast_float_to_dtype(kwargs, target_dtype)


def offload_data_to_cpu_device_hook(module: Optional[nn.Module], _: Any, output: Any) -> Any:
    """
    将模块的输出数据对齐到模块所在的设备上的 hook 函数。

    该函数作为 forward post-hook 使用，在模块 forward 后自动将输出数据移动到 cpu 设备。

    参数:
        module: 目标模块
        _: 位置参数（input）
        output: 位置参数（output）

    返回:
        对齐后的输入数据
    """
    processed_output = _align_input_to_module_device(module, output, torch.device('cpu'))
    return processed_output


def register_device_alignment_hook(
    module: Optional[nn.Module], with_kwargs: bool = False, name: Optional[str] = None, post_offload: bool = False
) -> Optional[torch.utils.hooks.RemovableHandle]:
    """
    为模块注册设备对齐 hook。

    该函数会为指定的模块注册一个 forward pre-hook，在每次 forward 前自动将输入数据
    对齐到模块所在的设备上。

    参数:
        module: 需要注册 hook 的模块
        with_kwargs: 是否使用kwargs格式的hook，默认为False
        name: 模块名称，用于日志输出，默认为None
        post_offload: 是否注册post-hook将数据移回CPU，默认为False

    返回:
        hook handle: 注册的 hook 句柄字典，用于后续移除 hook
    """
    if module is None:
        return None

    # 检查是否已经注册过 hook
    if hasattr(module, '_device_alignment_hooks_registered'):
        return {
            'pre_hook': module._device_alignment_pre_hook_handle,
            'post_hook': (
                module._device_alignment_post_hook_handle
                if hasattr(module, '_device_alignment_post_hook_handle')
                else None
            ),
        }

    # 注册 forward pre-hook
    pre_hook_handle = module.register_forward_pre_hook(align_input_to_module_device_hook, with_kwargs=with_kwargs)
    # 只有当post_offload为True时才注册forward post-hook
    post_hook_handle = None
    if post_offload:
        post_hook_handle = module.register_forward_hook(offload_data_to_cpu_device_hook, with_kwargs=False)

    # 保存句柄和注册标记
    module._device_alignment_hooks_registered = True  # 复数形式标记
    module._device_alignment_pre_hook_handle = pre_hook_handle  # pre句柄
    # 仅在post-hook被注册时才保存其句柄
    if post_hook_handle is not None:
        module._device_alignment_post_hook_handle = post_hook_handle  # post句柄

    # 使用提供的name或模块类名进行日志输出
    module_name = name if name is not None else module.__class__.__name__
    get_logger().debug("Registered device alignment hook for %r, post_offload: %s", module_name, post_offload)

    return {'pre_hook': pre_hook_handle, 'post_hook': post_hook_handle}


def unregister_device_alignment_hook(module: Optional[nn.Module], name: Optional[str] = None) -> None:
    """
    移除模块的设备对齐 hook。

    参数:
        module: 需要移除 hook 的模块
        name: 模块名称，用于日志输出，默认为None
    """
    if module is None or not hasattr(module, '_device_alignment_hooks_registered'):
        return

    # 移除 pre-hook（容错处理：防止句柄意外丢失）
    if hasattr(module, '_device_alignment_pre_hook_handle'):
        try:
            module._device_alignment_pre_hook_handle.remove()
        except Exception as e:
            get_logger().warning("Failed to remove pre-hook: %s", e)
        delattr(module, '_device_alignment_pre_hook_handle')

    # 移除 post-hook（仅当存在时）
    if hasattr(module, '_device_alignment_post_hook_handle'):
        try:
            module._device_alignment_post_hook_handle.remove()
        except Exception as e:
            get_logger().warning("Failed to remove post-hook: %s", e)
        delattr(module, '_device_alignment_post_hook_handle')

    # 清除注册标记
    delattr(module, '_device_alignment_hooks_registered')

    # 使用提供的name或模块类名进行日志输出
    module_name = name if name is not None else module.__class__.__name__
    get_logger().debug("Unregistered device alignment hook for %r", module_name)


def get_module_param_size(module: nn.Module) -> int:
    """
    计算模块的参数大小（字节数）

    Args:
        module: 待计算的模块

    Returns:
        模块参数总大小（字节数）
    """
    total_size: int = 0
    for param in module.parameters():
        total_size += param.numel() * param.element_size()
    return total_size


def get_device_allocated_memory() -> int:
    """
    获取设备已分配内存大小，自动检测设备类型

    Args:
        device_id: 设备ID，默认为0

    Returns:
        已分配内存大小，单位为字节
    """
    # 自动检测设备类型
    if hasattr(torch, 'npu') and torch.npu.is_available():
        device_id = torch.npu.current_device()
        return torch.npu.memory_allocated(device_id)
    elif hasattr(torch, 'cuda') and torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        return torch.cuda.memory_allocated(device_id)
    return 0


def get_device_reserved_memory() -> int:
    """
    获取设备已申请内存大小，自动检测设备类型

    Args:
        device_id: 设备ID，默认为0

    Returns:
        已申请内存大小，单位为字节
    """
    # 自动检测设备类型
    if hasattr(torch, 'npu') and torch.npu.is_available():
        device_id = torch.npu.current_device()
        return torch.npu.memory_reserved(device_id)
    elif hasattr(torch, 'cuda') and torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        return torch.cuda.memory_reserved(device_id)
    return 0


def format_memory_size(size_bytes: int) -> str:
    """
    格式化内存大小显示

    Args:
        size_bytes: 字节数

    Returns:
        格式化后的内存大小字符串
    """
    if size_bytes < 1024:
        return "%dB" % size_bytes
    elif size_bytes < 1024 * 1024:
        return "%.2fKB" % (size_bytes / 1024)
    elif size_bytes < 1024 * 1024 * 1024:
        return "%.2fMB" % (size_bytes / (1024 * 1024))
    else:
        return "%.2fGB" % (size_bytes / (1024 * 1024 * 1024))
