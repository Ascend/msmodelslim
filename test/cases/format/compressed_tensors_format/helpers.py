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

Test helpers for compressed_tensors_format unit tests.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from torch import nn

import msmodelslim.ir as qir
from msmodelslim.format.compressed_tensors_format.compressed_tensors_json_reader_factory_infra import (
    CompressedTensorJsonReaderFactoryInfra,
    CompressedTensorJsonReaderInfra,
)
from msmodelslim.format.compressed_tensors_format.compressed_tensors_json_writer_factory_infra import (
    CompressedTensorJsonWriterFactoryInfra,
    CompressedTensorJsonWriterInfra,
)
from msmodelslim.format.compressed_tensors_format.compressed_tensors_safetensors_writer_factory_infra import (
    CompressedTensorSafetensorsWriterFactoryInfra,
)
from msmodelslim.ir.qal import QDType, QParam, QScheme, QScope, QStorage

if TYPE_CHECKING:
    from msmodelslim.format.compressed_tensors_format.compressed_tensors import (
        CompressedTensorsQuantFormat,
    )
    from msmodelslim.format.interface import ExportContext


class MockSafetensorsWriter:
    """In-memory safetensors writer for export tests."""

    def __init__(self) -> None:
        self.tensors: Dict[str, torch.Tensor] = {}
        self.closed = False

    def write(self, key: str, value: torch.Tensor) -> None:
        self.tensors[key] = value

    def close(self) -> None:
        self.closed = True


class MockSafetensorsWriterCreatorInfra(CompressedTensorSafetensorsWriterFactoryInfra):
    """Factory that returns MockSafetensorsWriter instances."""

    def __init__(self) -> None:
        self.created: List[MockSafetensorsWriter] = []
        self.last_args: tuple = ()

    def create_safetensors_writer(
        self, part_file_size: int, save_directory: str, save_prefix: str
    ) -> MockSafetensorsWriter:
        self.last_args = (part_file_size, save_directory, save_prefix)
        writer = MockSafetensorsWriter()
        self.created.append(writer)
        return writer


MockSafetensorsWriterFactoryInfra = MockSafetensorsWriterCreatorInfra


class MockJsonWriter(CompressedTensorJsonWriterInfra):
    def __init__(self, save_directory: str, file_name: str) -> None:
        self.save_directory = save_directory
        self.file_name = file_name

    def dump(self, data: dict[str, Any], *, indent: int = 2) -> None:
        path = os.path.join(self.save_directory, self.file_name)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=indent)


class MockJsonWriterFactoryInfra(CompressedTensorJsonWriterFactoryInfra):
    def create_json_writer(self, save_directory: str, file_name: str) -> MockJsonWriter:
        return MockJsonWriter(save_directory, file_name)


class MockJsonReader(CompressedTensorJsonReaderInfra):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self._override_data: Optional[Any] = None

    def set_data(self, data: Any) -> None:
        self._override_data = data

    def load(self) -> dict[str, Any]:
        if self._override_data is not None:
            return self._override_data
        with open(self.file_path, encoding="utf-8") as file:
            return json.load(file)


class MockJsonReaderFactoryInfra(CompressedTensorJsonReaderFactoryInfra):
    def create_json_reader(self, file_path: str) -> MockJsonReader:
        return MockJsonReader(file_path)


def make_w8a8_static_module(out_features: int = 4, in_features: int = 8) -> qir.W8A8StaticFakeQuantLinear:
    input_scale = torch.tensor([0.5], dtype=torch.float32)
    input_offset = torch.tensor([0.0], dtype=torch.float32)
    weight_scale = torch.ones(out_features, dtype=torch.float32) * 0.1
    weight = torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8)
    bias = torch.zeros(out_features, dtype=torch.float32)

    x_q_param = QParam(
        scheme=QScheme(scope=QScope.PER_TENSOR, dtype=QDType.INT8, symmetric=False),
        ext={"scale": input_scale, "offset": input_offset},
    )
    w_q_param = QParam(
        scheme=QScheme(scope=QScope.PER_CHANNEL, dtype=QDType.INT8, symmetric=True),
        ext={"scale": weight_scale},
    )
    w_q = QStorage(dtype=QDType.INT8, value=weight)
    return qir.W8A8StaticFakeQuantLinear(x_q_param, w_q_param, w_q, bias)


def make_w8a8_dynamic_module(out_features: int = 4, in_features: int = 8) -> qir.W8A8DynamicPerChannelFakeQuantLinear:
    weight_scale = torch.ones(out_features, dtype=torch.float32) * 0.1
    weight = torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8)
    bias = torch.zeros(out_features, dtype=torch.float32)

    x_q_param = QParam(
        scheme=QScheme(scope=QScope.PER_TOKEN, dtype=QDType.INT8, symmetric=True),
        ext={"scale": torch.ones(1, dtype=torch.float32)},
    )
    w_q_param = QParam(
        scheme=QScheme(scope=QScope.PER_CHANNEL, dtype=QDType.INT8, symmetric=True),
        ext={"scale": weight_scale},
    )
    w_q = QStorage(dtype=QDType.INT8, value=weight)
    return qir.W8A8DynamicPerChannelFakeQuantLinear(x_q_param, w_q_param, w_q, bias)


class QuantizedModel(nn.Module):
    """Minimal model wrapping a single QIR static layer."""

    def __init__(self, module: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.linear = module if module is not None else make_w8a8_static_module()


class MixedQuantFloatModel(nn.Module):
    """Model with one quantized layer and one float Linear for ignore inference."""

    def __init__(self) -> None:
        super().__init__()
        self.quant = make_w8a8_static_module()
        self.float_linear = nn.Linear(8, 4)


def make_compressed_tensors_quant_format(
    ctx: "ExportContext",
    writer_infra: MockSafetensorsWriterCreatorInfra,
    *,
    json_writer_factory_infra: CompressedTensorJsonWriterFactoryInfra | None = None,
    json_reader_factory_infra: CompressedTensorJsonReaderFactoryInfra | None = None,
    prepare: bool = False,
) -> "CompressedTensorsQuantFormat":
    """Construct ``CompressedTensorsQuantFormat`` for tests (shared factory)."""
    from msmodelslim.format.compressed_tensors_format.compressed_tensors import (
        CompressedTensorsQuantFormat,
        CompressedTensorsQuantFormatConfig,
    )

    if json_writer_factory_infra is None:
        json_writer_factory_infra = MockJsonWriterFactoryInfra()
    if json_reader_factory_infra is None:
        json_reader_factory_infra = MockJsonReaderFactoryInfra()

    config = CompressedTensorsQuantFormatConfig(save_directory=str(ctx.save_directory))
    fmt = CompressedTensorsQuantFormat(
        config=config,
        ctx=ctx,
        safetensors_writer_factory_infra=writer_infra,
        json_writer_factory_infra=json_writer_factory_infra,
        json_reader_factory_infra=json_reader_factory_infra,
    )
    if prepare:
        fmt.prepare_export()
    return fmt
