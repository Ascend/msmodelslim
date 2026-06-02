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

from pathlib import Path
from typing import Any, Literal

from msmodelslim.format.base import QuantFormatConfig
from msmodelslim.format.interface import ExportContext
from msmodelslim.format.registry import QuantFormatFactory
from pydantic import Field, SerializeAsAny
from torch import nn

import msmodelslim.ir as qir
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.ir.qal import QABCRegistry
from msmodelslim.processor.base import AutoSessionProcessor, AutoProcessorConfig
from msmodelslim.utils.logging import get_logger, logger_setter


def _convert_hookir_to_wrapper(module: nn.Module) -> None:
    """
    将模块中的HookIR转换为Wrapper

    Args:
        module: 要处理的模块
    """
    for name, sub_module in module.named_modules():
        if hasattr(sub_module, "_forward_pre_hooks"):
            for hook in sub_module._forward_pre_hooks.values():
                if isinstance(hook, qir.HookIR):
                    wrapper = hook.wrapper_module(sub_module)
                    module.set_submodule(name, wrapper)
                    get_logger().info("Converted %s to wrapper for module: %s", type(hook), name)


class QuantSaveProcessorConfig(AutoProcessorConfig):
    type: Literal["saver"] = "saver"
    format: SerializeAsAny[QuantFormatConfig]
    save_directory: str = Field(default="", exclude=True)

    def set_save_directory(self, save_directory: str):
        self.save_directory = str(save_directory)


@QABCRegistry.register(dispatch_key=QuantSaveProcessorConfig, abc_class=AutoSessionProcessor)
@logger_setter(prefix="msmodelslim.processor.save")
class QuantSaveProcessor(AutoSessionProcessor):
    """
    统一保存会话处理器：目录准备、HookIR 转换、遍历与生命周期；
    与 :class:`~msmodelslim.format.interface.IFormat` 三段式协议对齐。
    """

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        adapter: object,
        **kwargs: Any,
    ):
        super().__init__(model)
        save_dir = str(getattr(config, "save_directory", "") or "")
        source_model_path = str(getattr(adapter, "model_path", "") or "")
        config.format.set_save_directory(save_dir)
        ctx = ExportContext(
            Path(save_dir),
            source_model_path=Path(source_model_path) if source_model_path else None,
        )

        self._quant_format_factory = QuantFormatFactory()
        self._format = self._quant_format_factory.create(config.format, ctx)

    def pre_run(self) -> None:
        self._format.prepare_export()

    def postprocess(self, request: BatchProcessRequest) -> None:
        prefix, module = request.name, request.module
        _convert_hookir_to_wrapper(module)
        self._format.process_module_tensors(prefix, module)

    def post_run(self) -> None:
        self._format.finalize_export(self.model)


__all__ = [
    "QuantSaveProcessor",
    "QuantSaveProcessorConfig",
    "_convert_hookir_to_wrapper",
]
