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

from typing import Any, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .wrapper import WrapperIR, HookIR


class SVDResidualWrapper(WrapperIR):
    """
    SVD残差包装器：包装Linear层，存储SVD分解后的低秩矩阵

    该类继承自WrapperIR，用于包装已进行SVD分解的Linear层，
    存储分解后的低秩矩阵 `svd_lowrank_l1` 与 `svd_lowrank_l2`。
    - `svd_lowrank_l1 = V^T` [rank, in_dim]
    - `svd_lowrank_l2 = U * S` [out_dim, rank]
    """

    def __init__(self, module: nn.Module, svd_lowrank_l1: torch.Tensor, svd_lowrank_l2: torch.Tensor):
        """
        初始化SVD残差包装器

        Args:
            module: 被包装的Linear层模块
            svd_lowrank_l1: V^T，形状 [rank, in_dim]
            svd_lowrank_l2: U * S，形状 [out_dim, rank]
        """
        super().__init__(module)
        self.svd_lowrank_l1 = nn.Parameter(svd_lowrank_l1, requires_grad=False)
        self.svd_lowrank_l2 = nn.Parameter(svd_lowrank_l2, requires_grad=False)

    @staticmethod
    def is_atomic() -> bool:
        """
        判断该IR是否为原子模块

        Returns:
            True: 该IR是原子模块，不需要递归处理内部模块
        """
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（双通路）：
        1) 主通路：使用残差权重进行线性变换；
        2) 旁路：通过 SVD 低秩分解矩阵进行低秩线性变换；
        3) 汇总：主通路与旁路输出相加。

        Args:
            x: 输入张量

        Returns:
            与原始权重近似等价的输出张量
        """
        # 主通路：残差权重进行线性变换
        residual_out = self.wrapped_module(x)

        # 旁路（低秩路径）：x -> F.linear(x, V^T) = x@V -> F.linear(x@V, U*S) = (x@V)@(U*S)^T
        lowrank_hidden = F.linear(x, self.svd_lowrank_l1, bias=None)
        lowrank_out = F.linear(lowrank_hidden, self.svd_lowrank_l2, bias=None)

        # 汇总输出：residual_out + lowrank_out ~= F.linear(x, original_weight, bias)
        return residual_out + lowrank_out


class SVDResidualHookIR(HookIR):
    """
    SVD残差钩子IR：将hook信息转换为SVDResidualWrapper

    该类实现了HookIR抽象基类，用于在模型前向传播时将Linear层包装为SVDResidualWrapper。
    """

    def __init__(self, svd_lowrank_l1: torch.Tensor, svd_lowrank_l2: torch.Tensor):
        """
        初始化SVDResidualHookIR

        Args:
            svd_lowrank_l1: SVD分解的l1矩阵 (V^T)，形状 [rank, in_dim]
            svd_lowrank_l2: SVD分解的l2矩阵 (U * S)，形状 [out_dim, rank]
        """
        super().__init__()
        self.svd_lowrank_l1 = svd_lowrank_l1
        self.svd_lowrank_l2 = svd_lowrank_l2

    def __call__(
        self,
        module: nn.Module,
        args: Tuple[Any, ...],
    ) -> Tuple[Any, ...]:
        """
        实现Callable接口，作为hook函数被调用

        当前实现直接返回输入，不做任何修改。
        实际的SVD重构计算在SVDResidualWrapper中完成。

        Args:
            module: 被hook的模块
            args: 模块的输入元组

        Returns:
            处理后的输入元组（当前实现直接返回原输入）
        """
        # 当前实现不修改输入，直接返回
        # SVD重构计算在wrapper_module中完成
        return args

    def wrapper_module(self, module: nn.Module) -> WrapperIR:
        """
        实现HookIR抽象方法，返回SVDResidualWrapper

        将Linear层包装为SVDResidualWrapper，以便在前向传播时
        使用SVD分解的低秩矩阵进行权重重构。

        Args:
            module: 要包装的Linear层模块

        Returns:
            SVDResidualWrapper实例
        """
        self.remove_hook()
        return SVDResidualWrapper(module, self.svd_lowrank_l1, self.svd_lowrank_l2)
