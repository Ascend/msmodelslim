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

import os
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Type

import torch
from torch import nn

import msmodelslim.ir as qir
from msmodelslim import logger
from msmodelslim.format.base import QuantFormatBase, QuantFormatConfig
from msmodelslim.format.compressed_tensors_format.config.base import QUANTIZATION_CONFIG_NAME
from msmodelslim.format.compressed_tensors_format.compressed_tensors_json_reader_factory_infra import (
    CompressedTensorJsonReaderFactoryInfra,
)
from msmodelslim.format.compressed_tensors_format.compressed_tensors_json_writer_factory_infra import (
    CompressedTensorJsonWriterFactoryInfra,
)
from msmodelslim.format.compressed_tensors_format.compressed_tensors_safetensors_writer_factory_infra import (
    CompressedTensorSafetensorsWriterFactoryInfra,
    CompressedTensorSafetensorsWriterInfra,
)
from msmodelslim.format.compressed_tensors_format.quantization.quant_config import QuantizationConfig
from msmodelslim.format.interface import ExportContext
from msmodelslim.utils.exception import ConfigError, InvalidModelError, SchemaValidateError
from msmodelslim.utils.security import (
    SafeWriteUmask,
    get_valid_read_path,
    get_valid_write_path,
    safe_copy_file,
)

_HF_COPY_SUFFIXES = (".json", ".py", ".txt", ".jinja")
_HF_SKIP_SUFFIXES = ("index.json",)
_SAVE_PREFIX = "model"
_WRITE_UMASK = 0o377


class CompressedTensorsQuantFormatConfig(QuantFormatConfig):
    type: Literal["compressed_tensors"] = "compressed_tensors"
    part_file_size: int = 4


class CompressedTensorsQuantFormat(QuantFormatBase):
    """compressed-tensors 导出：层间写 safetensors，收尾从量化模型反向推导并写 ``config.json``。"""

    def __init__(
        self,
        config: QuantFormatConfig,
        ctx: ExportContext,
        safetensors_writer_factory_infra: CompressedTensorSafetensorsWriterFactoryInfra,
        json_writer_factory_infra: CompressedTensorJsonWriterFactoryInfra,
        json_reader_factory_infra: CompressedTensorJsonReaderFactoryInfra,
    ) -> None:
        super().__init__(config, ctx)
        if not isinstance(safetensors_writer_factory_infra, CompressedTensorSafetensorsWriterFactoryInfra):
            raise SchemaValidateError(
                "safetensors_writer_factory_infra must be CompressedTensorSafetensorsWriterFactoryInfra, "
                f"got {type(safetensors_writer_factory_infra).__name__}",
                action="Pass a valid safetensors writer factory when creating CompressedTensorsQuantFormat.",
            )
        if not isinstance(json_writer_factory_infra, CompressedTensorJsonWriterFactoryInfra):
            raise SchemaValidateError(
                "json_writer_factory_infra must be CompressedTensorJsonWriterFactoryInfra, "
                f"got {type(json_writer_factory_infra).__name__}",
                action="Pass DefaultJsonWriterFactory when creating CompressedTensorsQuantFormat.",
            )
        if not isinstance(json_reader_factory_infra, CompressedTensorJsonReaderFactoryInfra):
            raise SchemaValidateError(
                "json_reader_factory_infra must be CompressedTensorJsonReaderFactoryInfra, "
                f"got {type(json_reader_factory_infra).__name__}",
                action="Pass DefaultJsonReaderFactory when creating CompressedTensorsQuantFormat.",
            )
        self._safetensors_writer_factory_infra = safetensors_writer_factory_infra
        self._json_writer_factory_infra = json_writer_factory_infra
        self._json_reader_factory_infra = json_reader_factory_infra
        self.safetensors_writer: Optional[CompressedTensorSafetensorsWriterInfra] = None

    def prepare_export(self) -> None:
        part_file_size = int(getattr(self.config, "part_file_size", 4))
        save_directory = str(self.ctx.save_directory)
        logger.info(
            "Preparing compressed-tensors export: save_directory=%s, part_file_size=%s, prefix=%s",
            save_directory,
            part_file_size,
            _SAVE_PREFIX,
        )
        self.safetensors_writer = self._safetensors_writer_factory_infra.create_safetensors_writer(
            part_file_size,
            save_directory,
            _SAVE_PREFIX,
        )

    def build_module_handler_map(self) -> Dict[Type[nn.Module], Callable[[str, nn.Module], None]]:
        return {
            qir.W8A8StaticFakeQuantLinear: self.on_w8a8_static,
            qir.W8A8DynamicPerChannelFakeQuantLinear: self.on_w8a8_dynamic_per_channel,
            nn.Linear: self.on_float_linear,
            nn.Module: self.on_float_module,
        }

    def finalize_export(self, model: nn.Module) -> None:
        try:
            if self.ctx.source_model_path is not None:
                self._copy_hf_files(str(self.ctx.source_model_path), str(self.ctx.save_directory))
            self._update_config_json(model)
            logger.info("Compressed-tensors export completed: %s", self.config.save_directory)
        finally:
            if self.safetensors_writer is not None:
                self.safetensors_writer.close()
                self.safetensors_writer = None

    def on_w8a8_static(self, prefix: str, module: qir.W8A8StaticFakeQuantLinear) -> None:
        with torch.device(module.weight.device):
            self.safetensors_writer.write(prefix + ".weight", module.weight.to(torch.int8))
            self.safetensors_writer.write(prefix + ".weight_scale", module.weight_scale.unsqueeze(1))

            if module.bias is not None:
                self.safetensors_writer.write(prefix + ".bias", module.bias)

            self.safetensors_writer.write(prefix + ".input_scale", module.input_scale.to(torch.float32))
            if torch.any(module.input_offset != 0):
                self.safetensors_writer.write(
                    prefix + ".input_zero_point",
                    module.input_offset,
                )

    def on_w8a8_dynamic_per_channel(self, prefix: str, module: qir.W8A8DynamicPerChannelFakeQuantLinear) -> None:
        with torch.device(module.weight.device):
            ws = module.weight_scale
            if ws.dim() == 1:
                ws = ws.unsqueeze(-1)

            self.safetensors_writer.write(prefix + ".weight", module.weight.to(torch.int8))
            self.safetensors_writer.write(prefix + ".weight_scale", ws)

            if module.bias is not None:
                self.safetensors_writer.write(prefix + ".bias", module.bias)

    def on_float_linear(self, prefix: str, module: nn.Linear) -> None:
        logger.debug("Exporting float Linear layer (unquantized): %s", prefix)
        return self.on_float_module(prefix, module)

    def on_float_module(self, prefix: str, module: nn.Module) -> None:
        for name, param in module.named_parameters(recurse=False, prefix=prefix):
            self.safetensors_writer.write(name, param.detach())

    def _copy_hf_files(self, input_path: str, output_path: str) -> None:
        for file in os.listdir(input_path):
            if not any(file.endswith(suffix) for suffix in _HF_COPY_SUFFIXES):
                continue
            if any(file.endswith(suffix) for suffix in _HF_SKIP_SUFFIXES):
                continue
            ori_file = os.path.join(input_path, file)
            dest_file = os.path.join(output_path, file)
            with SafeWriteUmask(umask=_WRITE_UMASK):
                safe_copy_file(src_path=ori_file, dest_path=dest_file)

    def _update_config_json(self, model: nn.Module) -> None:
        config_path = os.path.join(self.config.save_directory, "config.json")
        if not self._ensure_config_json_exists(config_path):
            raise ConfigError(
                f"config.json not found under {self.config.save_directory}",
                action="Provide source_model_path with a valid config.json, or place config.json in save_directory.",
            )

        read_path = get_valid_read_path(config_path)
        reader = self._json_reader_factory_infra.create_json_reader(read_path)
        config_data = reader.load()
        if not isinstance(config_data, dict):
            raise ConfigError(
                "Invalid config.json content, expected dict",
                action="Ensure config.json is a valid JSON object.",
            )

        config_data[QUANTIZATION_CONFIG_NAME] = self._build_quantization_config(model)
        write_path = get_valid_write_path(config_path, extensions=[".json"])
        writer = self._json_writer_factory_infra.create_json_writer(
            os.path.dirname(write_path),
            os.path.basename(write_path),
        )
        writer.dump(config_data, indent=2)
        logger.info("Updated compressed-tensors quantization_config in %s", write_path)

    def _ensure_config_json_exists(self, config_path: str) -> bool:
        if os.path.exists(config_path):
            return True

        if self.ctx.source_model_path is not None:
            src_config_path = Path(self.ctx.source_model_path) / "config.json"
            if src_config_path.is_file():
                write_path = get_valid_write_path(config_path, extensions=[".json"])
                src_reader = self._json_reader_factory_infra.create_json_reader(str(src_config_path))
                writer = self._json_writer_factory_infra.create_json_writer(
                    os.path.dirname(write_path),
                    os.path.basename(write_path),
                )
                writer.dump(src_reader.load(), indent=2)
        return os.path.exists(config_path)

    def _build_quantization_config(self, model: nn.Module) -> Dict[str, Any]:
        qconfig = QuantizationConfig.from_model(model)
        if qconfig is None:
            raise InvalidModelError(
                "No quantized QIR module found in model",
                action="Ensure the model contains at least one supported QIR fake-quant module before export.",
            )
        return qconfig.to_quantization_config_dict()


__all__ = [
    "CompressedTensorsQuantFormatConfig",
    "CompressedTensorsQuantFormat",
]
