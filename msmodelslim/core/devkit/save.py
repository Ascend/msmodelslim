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

from msmodelslim.core.quantization_format.base import QuantFormatConfig
from msmodelslim.core.quantization_format.interface import ExportContext
from msmodelslim.core.quantization_format.registry import QuantFormatFactory
from pydantic import Field, SerializeAsAny
from torch import nn

import msmodelslim.core.quantization_schema as qir
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.quantization_schema.qal import QABCRegistry
from msmodelslim.core.quantization_algorithm.base import AutoSessionProcessor, AutoProcessorConfig
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
    """统一保存处理器配置。

    说明：
    - 该处理器位于 `spec.process[]`。`type: saver` 是内部保存处理器的分派标识，
      不是模型导出格式；用户可见的导出格式由 `format.type` 决定。
    - 根级 `spec.save[]` 支持 `ascendv1_saver`、`compressed_tensors` 和
      `mindie_format_saver`；`ascendv1_saver` 与 `mindie_format_saver` 会路由到各自的
      保存处理器，`compressed_tensors` 则使用该统一保存处理器。
    - 直接使用 `QuantSaveProcessor` 时，`format` 当前仅支持 `compressed_tensors`。
      通常由框架根据根级 `spec.save[]` 自动注入 `format`，普通量化任务无需在
      `spec.process[]` 中手工编写 `type: saver`。

    注意事项：
    - 如需选择导出格式，请在根级 `spec.save[]` 中配置；只有直接构造统一保存处理器时，
      才需要在 `format` 中显式指定 `type: compressed_tensors` 及其扩展字段。
    """

    type: Literal["saver"] = Field(
        default="saver",
        description="处理器分派标识，固定为 `saver`；该值不是导出格式，导出格式由 `format.type` 指定。",
    )
    format: SerializeAsAny[QuantFormatConfig] = Field(
        description="导出格式配置（单对象）；由 `QuantSaveProcessor` 直接使用时当前必须为 "
        "`compressed_tensors`，通常由框架从 `spec.save[]` 自动注入。"
    )
    save_directory: str = Field(default="", exclude=True)

    def set_save_directory(self, save_directory: str):
        self.save_directory = str(save_directory)


@QABCRegistry.register(dispatch_key=QuantSaveProcessorConfig, abc_class=AutoSessionProcessor)
@logger_setter(prefix="msmodelslim.core.devkit.save")
class QuantSaveProcessor(AutoSessionProcessor):
    """
    统一保存会话处理器：目录准备、HookIR 转换、遍历与生命周期；
    与 :class:`~msmodelslim.core.quantization_format.interface.IFormat` 三段式协议对齐。
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
