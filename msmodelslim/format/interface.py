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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from torch import nn


@dataclass
class ExportContext:
    """导出运行时环境。"""

    save_directory: Path
    source_model_path: Path | None = None
    rank: int = 0
    world_size: int = 1


class IFormat(ABC):
    """
    量化模型落盘格式协议。

    基本用法::

        model = nn.Module()
        export_format = create_quant_format(...)  # 由配置与 ExportContext 构造

        export_format.prepare_export()
        for name, module in model.named_modules():
            export_format.process_module_tensors(name, module)
        export_format.finalize_export(model)

        if export_format.support_distributed() and dist.is_initialized():
            export_format.merge_ranks()
    """

    def prepare_export(self) -> None:
        """初始化导出（打开 writer、重置状态）。"""
        pass

    @abstractmethod
    def process_module_tensors(self, prefix: str, module: nn.Module) -> None:
        """导出 ``module`` 子树内的量化张量（不含全局元数据）。"""
        pass

    @abstractmethod
    def finalize_export(self, model: nn.Module) -> None:
        """收尾：关闭 writer，写入全模型元数据。"""
        pass


class IFormatLoader(ABC):
    """Protocol for loading a quantized model export and rebuilding FakeQuant IR.

    The load-side counterpart of ``IFormat``. A concrete loader (e.g. ``AscendV1Format``)
    reads the export directory, applies FakeQuant IR structure to the empty-weights shell
    (``apply_ir``), and hydrates weights per scope (``hydrate``).

    Basic usage::

        loader: IFormatLoader = build_default_format_chain().handle(path, device="cpu")
        loader.bind_adapter(adapter)
        loader.apply_ir(model)                          # full tree, one pass
        loader.hydrate(model, skip_prefixes=decoders)    # shared modules
        # ... per decoder layer:
        loader.hydrate(model, prefix="model.layers.0")
    """

    def bind_adapter(self, adapter: object) -> None:
        """Optional runtime adapter (e.g. FA3 inject). Default: ignore."""

    @abstractmethod
    def apply_ir(self, model: nn.Module) -> None:
        """Apply FakeQuant IR to every described Linear / activation in the full tree.

        Called once during shell build. Meta FakeQuant modules do not occupy real storage,
        so a full-tree pass is safe.
        """

    @abstractmethod
    def hydrate(self, model: nn.Module, prefix: Optional[str] = None, skip_prefixes: Optional[set] = None) -> None:
        """Load FakeQuant / FLOAT weights for modules in scope.

        ``prefix`` limits to one subtree (e.g. one decoder layer); ``skip_prefixes`` excludes
        subtrees (e.g. all decoder layers for shared hydrate); both ``None`` means full tree.
        """


__all__ = ["IFormat", "ExportContext", "IFormatLoader"]
