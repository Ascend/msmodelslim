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
# pylint: disable=attribute-defined-outside-init,bad-classmethod-argument

import torch
from torch import nn
from typing import Any, Dict, Tuple
from typing import List, Optional  # noqa: F401
from .wrapper import WrapperIR, HookIR


class FlatQuantOnlineWrapper(WrapperIR):
    """
    按Kronecker Product方式进行旋转的包装器。

    该类继承自WrapperIR，包装AutoFakeQuantLinear实例，使用全局共享的旋转矩阵，
    通过Kronecker Product组合后进行旋转运算。
    """

    def __init__(self, module: nn.Module):
        """
        初始化FlatQuantOnlineWrapper包装器。

        Args:
            module: 被包装的AutoFakeQuantLinear实例
            layer_name: 层名称，用于保存时标识
            rotation_info: 旋转矩阵信息
        """
        super().__init__(module)
        self.sigmoid = torch.nn.Sigmoid()
        self.clip_factor = None
        self.left_trans = None
        self.right_trans = None
        self.save_trans = None

    @staticmethod
    def is_atomic() -> bool:
        """
        如果该伪量化IR是原子性的，则返回True，否则返回False。
        原子性伪量化IR是指该IR应当被视为一个整体，不能被拆分，哪怕其内部包含其他伪量化IR。
        """
        return True

    def _add_clip(self, clip_factor: torch.Tensor):
        self.clip_factor = nn.Parameter(clip_factor, requires_grad=False)

    def _add_flat(self, save_trans: Dict[str, torch.Tensor]):
        if save_trans is None:
            return
        self.save_trans = save_trans

        left_trans = save_trans.get("left_trans", None)
        right_trans = save_trans.get("right_trans", None)
        if left_trans is not None:
            self.left_trans = torch.nn.Parameter(left_trans, requires_grad=False)
            self.save_trans["left_trans"] = self.left_trans
        if right_trans is not None:
            self.right_trans = torch.nn.Parameter(right_trans, requires_grad=False)
            self.save_trans["right_trans"] = self.right_trans

    def _apply_clip(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        可学习激活值裁剪(lac)

        Args:
            hidden_states: 激活值

        Returns:
            经过可学习激活值裁剪，裁剪之后的激活值。
        """
        init_shape = hidden_states.shape  # noqa: F841
        reshaped_hidden_states = hidden_states.reshape((-1, hidden_states.shape[-1]))

        h_max, h_min = reshaped_hidden_states.amax(1, keepdim=True), reshaped_hidden_states.amin(1, keepdim=True)
        tmp = torch.zeros_like(h_max)
        h_max, h_min = torch.maximum(h_max, tmp), torch.minimum(h_min, tmp)
        h_max = h_max * self.sigmoid(self.clip_factor)
        h_min = h_min * self.sigmoid(self.clip_factor)

        h_clamped = hidden_states.clamp(min=h_min, max=h_max)

        return h_clamped

    def _apply_flat(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        仿射变换，对激活值进行Kronecker积

        Args:
            hidden_states: 激活值

        Returns:
            经过仿射变换之后的激活值。
        """

        if self.right_trans is not None:
            init_shape = hidden_states.shape
            matrix = self.right_trans.to(hidden_states)
            hidden_states = hidden_states.reshape(-1, matrix.shape[0])
            hidden_states = hidden_states @ matrix
            hidden_states = hidden_states.reshape(init_shape)
        if self.left_trans is not None:
            self.size = self.left_trans.size()[0]
            init_shape = hidden_states.shape
            matrix = self.left_trans.T.to(hidden_states)
            hidden_states = hidden_states.reshape(-1, self.size, init_shape[-1] // self.size)
            hidden_states = matrix @ hidden_states
            hidden_states = hidden_states.reshape(init_shape)
        return hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播，在AutoFakeQuantLinear前添加Flat运算(可能包括 lac，smooth以及仿射变换)。

        Args:
            hidden_states: 输入张量

        Returns:
            经过Flat运算和线性变换的输出张量
        """
        hidden_states = self._apply_flat(hidden_states)
        if self.clip_factor is not None:
            hidden_states = self._apply_clip(hidden_states)
        return self.wrapped_module(hidden_states)

    @classmethod
    def create(self, module):
        return FlatQuantOnlineWrapper(module)

    def extra_repr(self) -> str:
        """
        返回额外的字符串表示，描述Kronecker旋转矩阵信息。

        Returns:
            包含Kronecker旋转矩阵信息的字符串
        """
        if self.left_trans is not None and self.right_trans is not None:
            return f"kronecker_rotation(Left:{self.left_trans.shape[0]}x{self.left_trans.shape[1]}, Right:{self.right_trans.shape[0]}x{self.right_trans.shape[1]})"
        return "No affine transformation was performed."


class FlatQuantOnlineHookIR(HookIR):
    """
    FlatQuant HookIR实现，用于在线的Kronecker旋转。

    该类实现了HookIR抽象基类，将hook信息转换为FlatQuantOnlineWrapper。
    """

    def __init__(self, clip_factor, save_trans):
        """
        初始化FlatQuantOnlineHookIR。

        Args:

        """
        super().__init__()
        self.clip_factor = clip_factor
        self.save_trans = save_trans

    def __call__(
        self,
        module: nn.Module,
        args: Tuple[Any, ...],
    ) -> Tuple[Any, ...]:
        """
        实现Callable接口，作为hook函数被调用。

        Args:
            module: 被hook的模块
            args: 模块的输入

        Returns:
            处理后的输入
        """
        x = args[0]
        return x

    def wrapper_module(self, module: nn.Module) -> WrapperIR:
        """
        实现HookIR抽象方法，返回FlatQuantOnlineWrapper。

        Args:
            module: 要包装的模块

        Returns:
            FlatQuantOnlineWrapper实例
        """
        self.remove_hook()
        flat_ir = FlatQuantOnlineWrapper(module)
        if self.clip_factor:
            flat_ir._add_clip(self.clip_factor)
        flat_ir._add_flat(self.save_trans)
        return flat_ir
