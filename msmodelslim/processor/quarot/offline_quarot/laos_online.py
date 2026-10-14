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
# pylint: disable=consider-using-from-import,logging-fstring-interpolation

from typing import Optional, Tuple

import torch
import torch.nn as nn

import msmodelslim.ir as qir
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.utils.exception import SchemaValidateError  # noqa: F401
from msmodelslim.utils.logging import get_logger
from .quarot_interface import LAOSOnlineRotationInterface
from ..common.quarot_utils import (
    get_decompose_dim,
    online_rotate_o_proj_input,
    online_rotate_down_proj,
    QuaRotMode,
    create_rot,
)


def _get_available_device(index: int = 0) -> torch.device:
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return torch.device("npu:{}".format(index))
    elif hasattr(torch, 'cuda') and torch.cuda.is_available():
        return torch.device("cuda:{}".format(index))
    else:
        return torch.device("cpu")


def _get_full_module_name(target_module: nn.Module, request: BatchProcessRequest) -> str:
    """
    通过遍历request.module获取完整的模块名称。

    Args:
        target_module: 目标模块
        request: 处理请求，包含模块前缀信息

    Returns:
        完整的模块名称

    Raises:
        UnsupportedError: 如果找不到完整的模块名称
    """
    for name, module in request.module.named_modules():
        if module is target_module:
            # 拼接完整路径：request.name + 相对路径
            full_name = f"{request.name}.{name}" if name else request.name
            return full_name

    # 如果找不到，抛出UnsupportedError
    raise UnsupportedError(f"Cannot find full module name for {target_module}")


def _convert_hookir_to_wrapper(module: nn.Module) -> None:
    """
    将模块中的HookIR转换为Wrapper

    Args:
        module: 要处理的模块
    """
    # 遍历模块中的所有子模块
    for name, sub_module in module.named_modules():
        if hasattr(sub_module, '_forward_pre_hooks'):
            # 遍历模块的所有前向钩子
            for hook in sub_module._forward_pre_hooks.values():
                # 检查是否是HookIR类型
                if isinstance(hook, qir.HookIR):
                    # 将hook_ir转换为wrapper
                    wrapper = hook.wrapper_module(sub_module)
                    # 将wrapper替换模块
                    module.set_submodule(name, wrapper)
                    get_logger().info(f"Converted {type(hook)} to wrapper for module: {name}")


class LAOSOnlineRotationProcessor:
    def __init__(self, model: nn.Module, config, adapter: LAOSOnlineRotationInterface, **kwargs) -> None:
        if not isinstance(adapter, LAOSOnlineRotationInterface):
            raise UnsupportedError(
                f'{adapter.__class__.__name__} does not support LAOSOnlineRotationInterface',
                action='Please provide a valid model adapter which implements LAOSOnlineRotationInterface',
            )
        self.config = config
        self.model = model
        self.adapter = adapter
        self.rot_online_down_proj: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.rot_online_o_proj: Optional[torch.Tensor] = None
        self.online_rotation_info: Optional[qir.QuarotOnlineRotationInfo] = None

        self.num_attn_heads: int = self.adapter.get_num_attention_heads()

    def pre_run(self) -> None:
        # 创建在线旋转矩阵
        device = _get_available_device()

        get_logger().info(f"Creating online rotation matrices on device: {device}")
        rot1, rot2, self.rot_online_o_proj = self._add_online_rotations(self.model)
        identity = torch.eye(int(self.adapter.get_head_dim())).to(self.rot_online_o_proj)
        self.rot_online_down_proj = (rot1, rot2)
        self.online_rotation_info = qir.QuarotOnlineRotationInfo(
            self.rot_online_o_proj, identity, rot1, rot2, self.config.max_tp_size
        )
        get_logger().info(f"Online rotation matrices created on device: {device}")

    def preprocess(self, request: BatchProcessRequest) -> None:
        # 获取ov_pair和up_down_pairs
        ov_pairs = self.adapter.get_layer_wise_ov_pair(request.module)
        up_down_pairs = self.adapter.get_layer_wise_up_down_pair(request.module)

        # 插入在线旋转矩阵
        layer_idx = int(request.name.split('.')[-1])

        if self.rot_online_o_proj is not None:
            online_rotate_o_proj_input(ov_pairs, rot_online=self.rot_online_o_proj, num_attn_heads=self.num_attn_heads)

        if self.rot_online_down_proj and layer_idx in self.config.down_proj_online_layers:
            online_rotate_down_proj(up_down_pairs, *self.rot_online_down_proj)

        # 注册在线旋转的hook
        if self.rot_online_down_proj and layer_idx in self.config.down_proj_online_layers:
            for _, down_proj in up_down_pairs.items():
                # 获取完整的模块名称
                full_module_name = _get_full_module_name(down_proj, request)
                # 使用完整的layer_name注册Kronecker HookIR实例作为hook（用于down_proj）
                hook_ir = qir.QuarotKroneckerRotationHookIR(full_module_name, self.online_rotation_info)
                hook_handle = down_proj.register_forward_pre_hook(hook_ir)
                hook_ir.set_hook_handle(hook_handle)

        if self.rot_online_o_proj is not None:
            for o_proj, _ in ov_pairs.items():
                # 获取完整的模块名称
                full_module_name = _get_full_module_name(o_proj, request)
                # 使用完整的layer_name注册普通HookIR实例作为hook（用于o_proj）
                hook_ir = qir.QuarotHeadsRotationHookIR(full_module_name, self.online_rotation_info)
                hook_handle = o_proj.register_forward_pre_hook(hook_ir)
                hook_ir.set_hook_handle(hook_handle)

    def post_run(self) -> None:
        """遍历模型中的HookIR，如果是自己添加的HookIR则进行转换"""
        _convert_hookir_to_wrapper(self.model)

    def _add_online_rotations(self, model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not hasattr(model.config, 'intermediate_size'):
            raise ValueError("Model config must contain 'intermediate_size' for online rotation")
        size_1, size_2 = get_decompose_dim(model.config.intermediate_size)
        rot1 = create_rot(mode=QuaRotMode.HADAMARD, size=size_1, block_size=self.config.max_tp_size).to(
            device=model.device
        )
        rot2 = create_rot(mode=QuaRotMode.HADAMARD, size=size_2, block_size=-1).to(model.device)

        rot_online_o_proj = create_rot(
            mode=QuaRotMode.HADAMARD, size=self.num_attn_heads, block_size=self.config.max_tp_size
        ).to(model.device)

        return rot1, rot2, rot_online_o_proj
