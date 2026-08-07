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

from enum import Enum

import torch
import torch.distributed as dist
from torch import nn


def is_rank_zero() -> bool:
    """
    判断当前进程是否是 rank 0

    在多卡量化时，通常只需要 rank 0 进行信息提取、日志记录、模型保存等操作。

    Returns:
        bool: 如果是 rank 0 或未初始化分布式环境，返回 True；否则返回 False

    Examples:
        >>> if is_rank_zero():
        >>>     # 只在 rank 0 上执行
        >>>     print("Saving model...")
        >>>     model.save()
    """
    if not dist.is_available():
        return True
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


class Scope(Enum):
    """
    定义模块作用域的枚举类
    LOCAL: 本地模块
    LOCAL_ONLY: 仅本地模块（不与其他进程共享）
    SHARED: 在多个进程间共享的模块
    ALL: 所有模块
    """

    LOCAL = 0
    LOCAL_ONLY = 1
    SHARED = 2
    ALL = 3


class DistHelper:
    """
    Expert Parallel(专家并行)模型处理类
    用于管理分布式训练中的模型模块，处理本地模块和共享模块的分配
    """

    def __init__(self, model: nn.Module, prefix: str = ""):
        """
        初始化EPModelHandle
        Args:
            model: 需要处理的PyTorch模型
        """
        self._model = model
        # 获取本地所有非None模块的名称集合
        self._local_modules = {name for name, module in self._model.named_modules(prefix=prefix) if module is not None}

        # 创建用于收集所有进程模块信息的列表
        gathered_modules = [None] * dist.get_world_size()
        # 在所有进程间同步模块信息
        dist.all_gather_object(gathered_modules, self._local_modules)

        # 计算所有进程共有的模块（交集）
        self._shared_modules = set.intersection(*gathered_modules)
        # 计算所有进程的模块总和（并集）
        self._all_modules = set.union(*gathered_modules)
        # 计算仅在本地存在的模块（差集）
        self._local_only_modules = self._local_modules - self._shared_modules

    @staticmethod
    def gather_variable_shapes(local_tensor: torch.Tensor):
        """
        支持不同 shape 张量的 all_gather 实现。

        Explicit ``device=`` on factories: Ascend HCCL requires NPU tensors;
        relying solely on ``torch.device`` context is fragile across torch_npu builds.
        """
        device = local_tensor.device
        world_size = dist.get_world_size()
        local_tensor = local_tensor.contiguous()

        # 同步张量形状
        local_shape = torch.tensor(local_tensor.shape, dtype=torch.long, device=device)
        shape_list = [torch.zeros_like(local_shape) for _ in range(world_size)]
        dist.all_gather(shape_list, local_shape)

        # 初始化存储（与 local 同 device / dtype）
        tensor_list = [
            torch.zeros(
                *s.tolist(),
                dtype=local_tensor.dtype,
                device=device,
            )
            for s in shape_list
        ]

        # 收集数据
        dist.all_gather(tensor_list, local_tensor)
        return tensor_list

    def get_rank(self):
        """
        获取当前进程的rank
        Returns:
            int: 当前进程的rank
        """
        _ = self
        return dist.get_rank()

    def local_modules(self):
        """
        生成器函数：遍历所有本地模块
        Yields:
            本地模块实例
        """
        for name in self._local_modules:
            yield self._model.get_submodule(name)

    def local_only_modules(self):
        """
        生成器函数：遍历仅存在于本地的模块
        Yields:
            仅本地模块实例
        """
        for name in self._local_only_modules:
            yield self._model.get_submodule(name)

    def shared_modules(self):
        """
        生成器函数：遍历所有共享模块
        Yields:
            共享模块实例
        """
        for name in self._shared_modules:
            yield self._model.get_submodule(name)

    def all_modules(self):
        """
        生成器函数：遍历所有进程中的所有模块
        Yields:
            所有模块实例
        """
        for name in self._all_modules:
            yield self._model.get_submodule(name)

    def is_local(self, name: str):
        """
        检查指定名称的模块是否为本地模块
        Args:
            name: 模块名称
        Returns:
            bool: 是否为本地模块
        """
        return name in self._local_modules

    def is_local_only(self, name: str):
        """
        检查指定名称的模块是否仅存在于本地
        Args:
            name: 模块名称
        Returns:
            bool: 是否仅为本地模块
        """
        return name in self._local_only_modules

    def is_shared(self, name: str):
        """
        检查指定名称的模块是否为共享模块
        Args:
            name: 模块名称
        Returns:
            bool: 是否为共享模块
        """
        return name in self._shared_modules

    def is_all(self, name: str):
        """
        检查指定名称的模块是否存在于所有进程中
        Args:
            name: 模块名称
        Returns:
            bool: 是否存在于所有进程
        """
        return name in self._all_modules

    def get_shared_modules_slice(self, prefix: str = ""):
        """
        将共享模块平均分配给每个rank
        Returns:
            list: 当前rank负责的共享模块列表
        """
        shared_modules_list = sorted([f"{prefix}.{name}" if prefix != "" else name for name in self._shared_modules])
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # 按照world_size的间隔取模块
        return shared_modules_list[rank::world_size]
