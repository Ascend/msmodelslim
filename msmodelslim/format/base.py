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

from __future__ import annotations

from typing import Callable, Dict, Literal, Set, Type

from abc import abstractmethod
from pydantic import BaseModel, Field
from torch import nn

import msmodelslim.ir as qir
from msmodelslim.format.interface import ExportContext, IFormat
from msmodelslim.utils.logging import get_logger

ModuleHandler = Callable[[str, nn.Module], None]


class QuantFormatConfig(BaseModel):
    """Base config for quantized export formats; subclasses are distinguished by ``type``."""

    type: Literal["_auto_save"] = "_auto_save"
    save_directory: str = Field(default=".", exclude=True)

    def set_save_directory(self, save_directory: str) -> None:
        """Set the output directory for export artifacts."""
        self.save_directory = str(save_directory)


class QuantFormatBase(IFormat):
    """Base class: module traversal + handler map; IO writers are owned by subclasses."""

    def __init__(self, config: QuantFormatConfig, ctx: ExportContext) -> None:
        self.config = config
        self.ctx = ctx
        self.processed_modules: Set[nn.Module] = set()
        self._module_handler_map = self.build_module_handler_map()

    def process_module_tensors(self, prefix: str, module: nn.Module) -> None:
        for name, sub_module in module.named_modules(memo=self.processed_modules, prefix=prefix):
            self._process_module_maybe_wrapper_ir(name, sub_module)

    def finalize_export(self, model: nn.Module) -> None:
        """Subclasses release format-specific writers and metadata here."""
        pass

    def merge_ranks(self) -> None:
        pass

    @abstractmethod
    def build_module_handler_map(self) -> Dict[Type[nn.Module], ModuleHandler]:
        """子类实现：模块类型到落盘 handler 的映射表。"""
        pass

    @abstractmethod
    def on_float_module(self, prefix: str, module: nn.Module) -> None:
        """Export parameters for modules without a dedicated quant handler."""
        pass

    def _process_module_maybe_wrapper_ir(self, prefix: str, module: nn.Module) -> None:
        if isinstance(module, qir.WrapperIR):
            self._on_wrapper_ir(prefix, module)
        else:
            self._process_module(prefix, module)

    def _on_wrapper_ir(self, prefix: str, module: qir.WrapperIR):
        """
        处理WrapperIR类型的模块。

        WrapperIR是一个包装器，它持有一个被包装的nn.Module。在保存时，
        根据包装器的原子性决定处理策略。

        处理策略：
        - 如果包装器不是原子性的（is_atomic()返回False），先处理被包装模块，再处理包装器自身
        - 如果包装器是原子性的（is_atomic()返回True），只处理包装器自身，跳过被包装模块

        这样设计的好处：
        - 支持原子性和非原子性包装器的不同处理需求
        - 原子性包装器作为整体处理，避免重复处理
        - 非原子性包装器可以分别处理被包装模块和包装器自身

        Args:
            prefix: 模块名称前缀
            module: WrapperIR模块实例
        """
        # 根据原子性决定处理策略
        wrapped_module = module.wrapped_module
        if not module.is_atomic():
            # 非原子性：先处理被包装模块，再处理包装器自身
            self._process_module(prefix, wrapped_module)
        # 处理包装器自身
        self._process_module(prefix, module)

    def _process_module(self, prefix: str, module: nn.Module):
        """
        使用process_map处理模块的通用方法。

        Args:
            prefix: 模块名称前缀
            module: 要处理的模块
        """

        get_logger().debug("Processing module %r for %r", type(module).__name__, prefix)

        if type(module) in self._module_handler_map:
            self._module_handler_map[type(module)](prefix, module)
        else:
            self.on_float_module(prefix, module)

        self.processed_modules.add(module)


__all__ = ["QuantFormatConfig", "QuantFormatBase", "ModuleHandler"]
