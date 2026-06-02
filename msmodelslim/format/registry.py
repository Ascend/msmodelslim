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

from typing import Any, List, Union

from pydantic import Field, TypeAdapter
from typing_extensions import Annotated

from msmodelslim.format.ascendV1_format.ascendV1 import AscendV1QuantFormatConfig
from msmodelslim.format.compressed_tensors_format.compressed_tensors import (
    CompressedTensorsQuantFormat,
    CompressedTensorsQuantFormatConfig,
)
from msmodelslim.format.compressed_tensors_format.compressed_tensors_json_reader_factory_infra import (
    CompressedTensorJsonReaderFactoryInfra,
)
from msmodelslim.format.compressed_tensors_format.compressed_tensors_json_writer_factory_infra import (
    CompressedTensorJsonWriterFactoryInfra,
)
from msmodelslim.format.compressed_tensors_format.compressed_tensors_safetensors_writer_factory_infra import (
    CompressedTensorSafetensorsWriterFactoryInfra,
)
from msmodelslim.format.base import QuantFormatConfig
from msmodelslim.format.interface import ExportContext, IFormat
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.format.mindie_format.mindie import MindIEQuantFormatConfig

QuantFormatConfigUnion = Annotated[
    Union[
        CompressedTensorsQuantFormatConfig,
        AscendV1QuantFormatConfig,
        MindIEQuantFormatConfig,
    ],
    Field(discriminator="type"),
]

QuantFormatConfigList = List[QuantFormatConfigUnion]

_format_config_adapter = TypeAdapter(QuantFormatConfigUnion)


def parse_format_config(data: dict[str, Any]) -> QuantFormatConfig:
    """Parse a format config dict via Pydantic tagged-union dispatch on ``type``."""
    return _format_config_adapter.validate_python(data)


class QuantFormatFactory:
    """注册内置格式，并持有默认 IO factory 以构造 ``IFormat`` 实例。"""

    def __init__(
        self,
        safetensors_writer_factory_infra: CompressedTensorSafetensorsWriterFactoryInfra | None = None,
        json_writer_factory_infra: CompressedTensorJsonWriterFactoryInfra | None = None,
        json_reader_factory_infra: CompressedTensorJsonReaderFactoryInfra | None = None,
    ) -> None:
        if safetensors_writer_factory_infra is None:
            from msmodelslim.infra.io.default_safetensors_writer_factory import (
                DefaultSafetensorsWriterFactory,
            )

            safetensors_writer_factory_infra = DefaultSafetensorsWriterFactory()
        if json_writer_factory_infra is None:
            from msmodelslim.infra.io.default_json_writer_factory import (
                DefaultJsonWriterFactory,
            )

            json_writer_factory_infra = DefaultJsonWriterFactory()
        if json_reader_factory_infra is None:
            from msmodelslim.infra.io.default_json_reader_factory import (
                DefaultJsonReaderFactory,
            )

            json_reader_factory_infra = DefaultJsonReaderFactory()

        self._safetensors_writer_factory_infra = safetensors_writer_factory_infra
        self._json_writer_factory_infra = json_writer_factory_infra
        self._json_reader_factory_infra = json_reader_factory_infra

    def create_compressed_tensors_quant_format(
        self,
        config: CompressedTensorsQuantFormatConfig,
        ctx: ExportContext,
    ) -> CompressedTensorsQuantFormat:
        return CompressedTensorsQuantFormat(
            config,
            ctx,
            self._safetensors_writer_factory_infra,
            self._json_writer_factory_infra,
            self._json_reader_factory_infra,
        )

    def create(self, config: QuantFormatConfig, ctx: ExportContext) -> IFormat:
        if isinstance(config, CompressedTensorsQuantFormatConfig):
            return self.create_compressed_tensors_quant_format(config, ctx)
        raise SchemaValidateError(
            f"Unsupported quant format config type: {type(config).__name__}",
            action="Use a supported format config such as compressed_tensors.",
        )


__all__ = [
    "QuantFormatFactory",
    "QuantFormatConfigUnion",
    "QuantFormatConfigList",
    "parse_format_config",
]
