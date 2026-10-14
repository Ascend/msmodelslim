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

Unit tests for msmodelslim.format.registry.
"""

from __future__ import annotations

import pytest

from msmodelslim.format.ascendV1_format.ascendV1 import AscendV1QuantFormatConfig
from msmodelslim.format.compressed_tensors_format.compressed_tensors import (
    CompressedTensorsQuantFormat,
    CompressedTensorsQuantFormatConfig,
)
from msmodelslim.format.interface import ExportContext
from msmodelslim.format.mindie_format.mindie import MindIEQuantFormatConfig
from msmodelslim.format.registry import QuantFormatFactory, parse_format_config
from msmodelslim.utils.exception import SchemaValidateError


class TestParseFormatConfig:
    """Tests for parse_format_config."""

    def test_parse_format_config_return_compressed_tensors_when_type_set(self):
        config = parse_format_config({"type": "compressed_tensors"})

        assert isinstance(config, CompressedTensorsQuantFormatConfig)
        assert config.type == "compressed_tensors"

    def test_parse_format_config_return_ascendv1_when_type_set(self):
        config = parse_format_config({"type": "ascendv1_saver"})

        assert isinstance(config, AscendV1QuantFormatConfig)
        assert config.type == "ascendv1_saver"

    def test_parse_format_config_return_mindie_when_type_set(self):
        config = parse_format_config({"type": "mindie_format_saver"})

        assert isinstance(config, MindIEQuantFormatConfig)
        assert config.type == "mindie_format_saver"


class TestQuantFormatFactory:
    """Tests for QuantFormatFactory."""

    def test_create_return_compressed_tensors_format_when_config_valid(self, tmp_path):
        ctx = ExportContext(save_directory=tmp_path)
        config = CompressedTensorsQuantFormatConfig(save_directory=str(tmp_path))

        fmt = QuantFormatFactory().create(config, ctx)

        assert isinstance(fmt, CompressedTensorsQuantFormat)

    def test_create_raise_schema_validate_error_when_config_unsupported(self, tmp_path):
        ctx = ExportContext(save_directory=tmp_path)
        config = AscendV1QuantFormatConfig()

        with pytest.raises(SchemaValidateError, match="Unsupported quant format config type"):
            QuantFormatFactory().create(config, ctx)

    def test_factory_inject_default_io_factories_when_not_provided(self):
        from msmodelslim.infra.io.default_json_reader_factory import DefaultJsonReaderFactory
        from msmodelslim.infra.io.default_json_writer_factory import DefaultJsonWriterFactory
        from msmodelslim.infra.io.default_safetensors_writer_factory import DefaultSafetensorsWriterFactory

        factory = QuantFormatFactory()

        assert isinstance(factory._safetensors_writer_factory_infra, DefaultSafetensorsWriterFactory)
        assert isinstance(factory._json_writer_factory_infra, DefaultJsonWriterFactory)
        assert isinstance(factory._json_reader_factory_infra, DefaultJsonReaderFactory)
