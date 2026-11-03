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

``CheckpointSaverFormat``：遵守 ``IFormat``，落盘：

1. FULL ``state_dict`` → safetensors（来自 ``ExportContext.state_dict``，由调用方在构造 ctx 时注入）
2. ``qat_checkpoint.json``（``modules`` 元数据）
3. HF sidecar（从 ``source_model_path`` 拷贝 tokenizer 等；再用 model config 覆盖）

``process_module_tensors`` 只扫图收集 meta，不写权重；``finalize_export(model)`` 保持 ``IFormat`` 签名。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Type

import torch
from torch import nn

from msmodelslim.format.base import ModuleHandler, QuantFormatBase, QuantFormatConfig
from msmodelslim.format.checkpoint_format.checkpoint_json_writer_factory_infra import (
    CheckpointJsonWriterFactoryInfra,
    CheckpointJsonWriterInfra,
)
from msmodelslim.format.checkpoint_format.checkpoint_tensors_writer_factory_infra import (
    CheckpointSafetensorsWriterFactoryInfra,
    CheckpointSafetensorsWriterInfra,
)
from msmodelslim.format.checkpoint_format.constants import (
    FORMAT_NAME,
    FORMAT_VERSION,
    STRUCTURE_FILE,
    WEIGHT_FILE,
    WEIGHT_INDEX_FILE,
    WEIGHT_PREFIX,
    canonical_module_name,
)
from msmodelslim.format.interface import ExportContext
from msmodelslim.processor.paretoq.impl import ParetoQFakeQuantLinear
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.security import safe_copy_file, set_file_stat

# Re-export for existing importers.
_STRUCTURE_FILE = STRUCTURE_FILE
_WEIGHT_FILE = WEIGHT_FILE

_HF_COPY_SUFFIXES = (".json", ".py", ".txt", ".jinja")
_HF_SKIP_SUFFIXES = ("index.json",)


class CheckpointSaverFormat(QuantFormatBase):
    """浮点 QAT checkpoint 落盘：``state_dict`` → safetensors；模块遍历 → json + HF sidecar。"""

    def __init__(
        self,
        ctx: ExportContext,
        safetensors_writer_factory_infra: CheckpointSafetensorsWriterFactoryInfra,
        json_writer_factory_infra: CheckpointJsonWriterFactoryInfra,
        *,
        qparam_names: List[str],
        algorithm: Optional[List[Dict[str, Any]]] = None,
        part_file_size: int = 0,
    ) -> None:
        # QuantFormatBase 需要 config 占位；清单字段直接挂在本类上。
        super().__init__(QuantFormatConfig(), ctx)
        self._safetensors_writer_factory_infra = safetensors_writer_factory_infra
        self._json_writer_factory_infra = json_writer_factory_infra
        self.algorithm: List[Dict[str, Any]] = list(algorithm or [])
        self.qparam_names: List[str] = list(qparam_names)
        self.part_file_size = int(part_file_size or 0)
        self.safetensors_writer: Optional[CheckpointSafetensorsWriterInfra] = None
        self.json_writer: Optional[CheckpointJsonWriterInfra] = None
        self._modules_meta: Dict[str, Dict[str, Any]] = {}
        self._logger = get_logger()

    def prepare_export(self) -> None:
        save_directory = str(self.ctx.save_directory)
        os.makedirs(save_directory, exist_ok=True)
        self._modules_meta = {}
        self.processed_modules = set()
        # part_file_size>0 → 分片 + index；0 → 单文件 model.safetensors
        self.safetensors_writer = self._safetensors_writer_factory_infra.create_safetensors_writer(
            self.part_file_size,
            save_directory,
            WEIGHT_PREFIX,
        )
        self.json_writer = self._json_writer_factory_infra.create_json_writer(
            save_directory,
            STRUCTURE_FILE,
        )
        self._logger.info(
            "CheckpointSaverFormat.prepare_export: dir=%s part_file_size=%s",
            save_directory,
            self.part_file_size,
        )

    def build_module_handler_map(self) -> Dict[Type[nn.Module], ModuleHandler]:
        return {ParetoQFakeQuantLinear: self.on_paretoq_linear}

    def on_paretoq_linear(self, prefix: str, module: ParetoQFakeQuantLinear) -> None:
        """只记录还原壳所需元数据；不写权重。"""
        self._modules_meta[canonical_module_name(prefix)] = {
            "class": type(module).__name__,
            "w_bits": int(module.w_bits),
            "deploy_act": getattr(module, "deploy_act", None),
            "weight_layerwise": bool(getattr(module, "weight_layerwise", False)),
        }

    def on_float_module(self, prefix: str, module: nn.Module) -> None:
        """普通浮点模块无清单项；权重只来自 ``ExportContext.state_dict``。"""
        del prefix, module

    def finalize_export(self, model: nn.Module) -> None:
        """写出权重、``qat_checkpoint.json``，以及 HF sidecar。

        权重必须来自 ``ExportContext.state_dict``（FSDP gather 结果）。
        ``model`` 仅满足 ``IFormat`` 签名；meta 已在 ``process_module_tensors`` 收集。
        """
        del model
        state_dict = self.ctx.state_dict
        if not state_dict:
            raise RuntimeError(
                "finalize_export requires ExportContext.state_dict "
                "(inject the gathered full state_dict when constructing ExportContext)"
            )
        try:
            if self.safetensors_writer is None or self.json_writer is None:
                raise RuntimeError("Call prepare_export() before finalize_export")

            for key, value in state_dict.items():
                if isinstance(value, torch.Tensor):
                    self.safetensors_writer.write(canonical_module_name(key), value)

            weight_file = WEIGHT_INDEX_FILE if self.part_file_size > 0 else WEIGHT_FILE
            self.json_writer.dump(
                {
                    "format": FORMAT_NAME,
                    "version": FORMAT_VERSION,
                    "algorithm": list(self.algorithm),
                    "qparam_names": list(self.qparam_names),
                    "modules": dict(self._modules_meta),
                    "tensors": {"weight_file": weight_file},
                },
                indent=2,
            )
            self._save_hf_sidecars()
            self._logger.info(
                "CheckpointSaverFormat.finalize_export: wrote %s modules=%d tensors=%d",
                os.path.join(str(self.ctx.save_directory), STRUCTURE_FILE),
                len(self._modules_meta),
                len(state_dict),
            )
        finally:
            if self.safetensors_writer is not None:
                self.safetensors_writer.close()
                self.safetensors_writer = None
            self.json_writer = None

    def _save_hf_sidecars(self) -> None:
        """Copy HF sidecar files from ``source_model_path``, then refresh config from model."""
        save_directory = str(self.ctx.save_directory)
        if self.ctx.source_model_path is not None:
            self._copy_hf_files(str(self.ctx.source_model_path), save_directory)
        else:
            self._logger.warning(
                "CheckpointSaverFormat: source_model_path is None; tokenizer / HF sidecar files will not be copied",
            )

    def _copy_hf_files(self, input_path: str, output_path: str) -> None:
        """Mirror ``CompressedTensorsQuantFormat._copy_hf_files`` for tokenizer / HF metadata."""
        if not os.path.isdir(input_path):
            self._logger.warning(
                "CheckpointSaverFormat: source_model_path is not a directory: %s",
                input_path,
            )
            return
        for file in os.listdir(input_path):
            if file == STRUCTURE_FILE:
                continue
            if not any(file.endswith(suffix) for suffix in _HF_COPY_SUFFIXES):
                continue
            if any(file.endswith(suffix) for suffix in _HF_SKIP_SUFFIXES):
                continue
            ori_file = os.path.join(input_path, file)
            dest_file = os.path.join(output_path, file)
            set_file_stat(dest_file, "600")
            safe_copy_file(src_path=ori_file, dest_path=dest_file)
            set_file_stat(dest_file, "600")


__all__ = [
    "CheckpointSaverFormat",
    "_STRUCTURE_FILE",
    "_WEIGHT_FILE",
]
