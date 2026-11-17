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

from abc import abstractmethod
from typing import Optional, Callable, Set, Iterator, Tuple

from torch import nn
from torch.utils.hooks import RemovableHandle

from msmodelslim.utils.logging import get_logger


class WrapperIR(nn.Module):
    """
    包装类IR基类，用于包装其他nn.Module。

    该类是一个nn.Module，但持有一个被包装的nn.Module，提供统一的接口
    来访问和操作被包装的模块。
    """

    def __init__(self, wrapped_module: nn.Module):
        """
        初始化包装类IR。

        Args:
            wrapped_module: 被包装的nn.Module实例
        """
        super().__init__()
        self.wrapped_module = wrapped_module

    @staticmethod
    def is_atomic() -> bool:
        """
        如果该包装类IR是原子性的，则返回True，否则返回False。
        原子性包装类IR是指该IR应当被视为一个整体，不能被拆分。

        在保存时，原子性包装器会作为整体处理，不会单独处理被包装的模块。
        非原子性包装器会分别处理被包装模块和包装器自身。

        Returns:
            False，表示该包装类IR是非原子性的
        """
        return False

    def named_modules(
        self, memo: Optional[Set[nn.Module]] = None, prefix: str = '', remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Module]]:
        """
        重写named_modules方法，如果is_atomic()返回True，则只返回自身。

        Args:
            memo: 用于避免重复访问的模块集合
            prefix: 模块名称前缀
            remove_duplicate: 是否移除重复的模块

        Yields:
            模块名称和模块实例的元组
        """
        yield prefix, self


class HookIR(Callable):
    """
    HookIR基类，持有一个hook句柄，可以随时移除hook
    """

    def __init__(self):
        """初始化HookIR，创建hook句柄"""
        self.hook_handle: Optional[RemovableHandle] = None

    def set_hook_handle(self, hook_handle: RemovableHandle):
        """
        设置hook句柄

        Args:
            hook_handle: hook的句柄
        """
        self.hook_handle = hook_handle

    def remove_hook(self):
        """
        移除hook
        """
        if self.hook_handle is not None:
            try:
                self.hook_handle.remove()
                get_logger().debug("Removed hook with handle: %s", self.hook_handle)
            except Exception as e:
                get_logger().warning("Failed to remove hook: %s", e)

    @abstractmethod
    def wrapper_module(self, module: nn.Module) -> WrapperIR:
        pass
