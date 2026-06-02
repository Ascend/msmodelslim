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

Unit tests for msmodelslim.format.compressed_tensors_format.compressed_tensors.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import pytest
import torch
from torch import nn

import msmodelslim.ir as qir
from msmodelslim.format.compressed_tensors_format.compressed_tensors import (
    CompressedTensorsQuantFormat,
    CompressedTensorsQuantFormatConfig,
)
from msmodelslim.format.compressed_tensors_format.config.base import (
    QUANTIZATION_CONFIG_NAME,
)
from msmodelslim.format.interface import ExportContext
from msmodelslim.utils.exception import (
    ConfigError,
    InvalidModelError,
    SchemaValidateError,
)

from test.cases.format.compressed_tensors_format.helpers import (
    MockJsonReader,
    MockJsonReaderFactoryInfra,
    MockJsonWriterFactoryInfra,
    QuantizedModel,
    make_compressed_tensors_quant_format,
    make_w8a8_dynamic_module,
    make_w8a8_static_module,
)


class TestCompressedTensorsQuantFormatConfig:
    """Tests for CompressedTensorsQuantFormatConfig."""

    def test_config_type_is_compressed_tensors_when_default(self):
        config = CompressedTensorsQuantFormatConfig()
        assert config.type == "compressed_tensors"

    def test_config_part_file_size_is_four_when_default(self):
        config = CompressedTensorsQuantFormatConfig()
        assert config.part_file_size == 4


class TestCompressedTensorsQuantFormatInit:
    """Tests for CompressedTensorsQuantFormat.__init__."""

    def test_init_via_quant_format_factory_injects_defaults(self, export_ctx):
        from msmodelslim.format.registry import QuantFormatFactory
        from msmodelslim.infra.io.default_json_reader_factory import (
            DefaultJsonReaderFactory,
        )
        from msmodelslim.infra.io.default_json_writer_factory import (
            DefaultJsonWriterFactory,
        )
        from msmodelslim.infra.io.default_safetensors_writer_factory import (
            DefaultSafetensorsWriterFactory,
        )

        config = CompressedTensorsQuantFormatConfig(save_directory=str(export_ctx.save_directory))
        fmt = QuantFormatFactory().create(config, export_ctx)

        assert isinstance(fmt._safetensors_writer_factory_infra, DefaultSafetensorsWriterFactory)
        assert isinstance(fmt._json_writer_factory_infra, DefaultJsonWriterFactory)
        assert isinstance(fmt._json_reader_factory_infra, DefaultJsonReaderFactory)

    def test_init_raise_schema_validate_error_when_safetensors_writer_factory_invalid(self, export_ctx, writer_infra):
        config = CompressedTensorsQuantFormatConfig(save_directory=str(export_ctx.save_directory))
        with pytest.raises(SchemaValidateError, match="safetensors_writer_factory_infra must be"):
            CompressedTensorsQuantFormat(
                config=config,
                ctx=export_ctx,
                safetensors_writer_factory_infra=object(),  # type: ignore[arg-type]
                json_writer_factory_infra=MockJsonWriterFactoryInfra(),
                json_reader_factory_infra=MockJsonReaderFactoryInfra(),
            )

    def test_init_raise_schema_validate_error_when_json_writer_factory_invalid(self, export_ctx, writer_infra):
        config = CompressedTensorsQuantFormatConfig(save_directory=str(export_ctx.save_directory))
        with pytest.raises(SchemaValidateError, match="json_writer_factory_infra must be"):
            CompressedTensorsQuantFormat(
                config=config,
                ctx=export_ctx,
                safetensors_writer_factory_infra=writer_infra,
                json_writer_factory_infra=object(),  # type: ignore[arg-type]
                json_reader_factory_infra=MockJsonReaderFactoryInfra(),
            )

    def test_init_raise_schema_validate_error_when_json_reader_factory_invalid(self, export_ctx, writer_infra):
        config = CompressedTensorsQuantFormatConfig(save_directory=str(export_ctx.save_directory))
        with pytest.raises(SchemaValidateError, match="json_reader_factory_infra must be"):
            CompressedTensorsQuantFormat(
                config=config,
                ctx=export_ctx,
                safetensors_writer_factory_infra=writer_infra,
                json_writer_factory_infra=MockJsonWriterFactoryInfra(),
                json_reader_factory_infra=object(),  # type: ignore[arg-type]
            )


class TestCompressedTensorsQuantFormatPrepareExport:
    """Tests for CompressedTensorsQuantFormat.prepare_export."""

    def test_prepare_export_create_writer_when_infra_configured(self, export_ctx, writer_infra):
        fmt = make_compressed_tensors_quant_format(export_ctx, writer_infra, prepare=True)

        assert fmt.safetensors_writer is not None
        assert writer_infra.last_args == (4, str(export_ctx.save_directory), "model")


class TestCompressedTensorsQuantFormatOnW8A8Static:
    """Tests for CompressedTensorsQuantFormat.on_w8a8_static."""

    def test_on_w8a8_static_write_tensors_when_module_valid(self, quant_format):
        module = make_w8a8_static_module()
        prefix = "model.layers.0.self_attn.q_proj"

        quant_format.on_w8a8_static(prefix, module)

        writer = quant_format.safetensors_writer
        assert f"{prefix}.weight" in writer.tensors
        assert f"{prefix}.weight_scale" in writer.tensors
        assert f"{prefix}.input_scale" in writer.tensors
        assert writer.tensors[f"{prefix}.weight"].dtype == torch.int8

    def test_on_w8a8_static_raise_invalid_model_error_when_weight_scale_none(self, quant_format):
        module = make_w8a8_static_module()
        module.weight_scale = None

        with pytest.raises(AttributeError):
            quant_format.on_w8a8_static("layer", module)

    def test_on_w8a8_static_write_zero_point_when_input_offset_nonzero(self, quant_format):
        module = make_w8a8_static_module()
        module.input_offset.data.fill_(3)

        quant_format.on_w8a8_static("layer", module)

        assert "layer.input_zero_point" in quant_format.safetensors_writer.tensors


class TestCompressedTensorsQuantFormatOnW8A8DynamicPerChannel:
    """Tests for CompressedTensorsQuantFormat.on_w8a8_dynamic_per_channel."""

    def test_on_w8a8_dynamic_write_unsqueezed_scale_when_scale_is_1d(self, quant_format):
        module = make_w8a8_dynamic_module()
        prefix = "model.layers.0.mlp.down_proj"

        quant_format.on_w8a8_dynamic_per_channel(prefix, module)

        scale = quant_format.safetensors_writer.tensors[f"{prefix}.weight_scale"]
        assert scale.dim() == 2

    def test_on_w8a8_dynamic_raise_invalid_model_error_when_weight_scale_none(self, quant_format):
        module = make_w8a8_dynamic_module()
        module.weight_scale = None

        with pytest.raises(AttributeError):
            quant_format.on_w8a8_dynamic_per_channel("layer", module)


class TestCompressedTensorsQuantFormatOnFloatModule:
    """Tests for float module export handlers."""

    def test_on_float_module_write_parameters_when_linear_unquantized(self, quant_format):
        linear = nn.Linear(4, 2, bias=False)
        quant_format.on_float_module("float_layer", linear)

        assert "float_layer.weight" in quant_format.safetensors_writer.tensors

    def test_on_float_linear_delegate_to_float_module_when_called(self, quant_format):
        linear = nn.Linear(4, 2, bias=False)

        quant_format.on_float_linear("float_layer", linear)

        assert "float_layer.weight" in quant_format.safetensors_writer.tensors


class TestCompressedTensorsQuantFormatCopyHfFiles:
    """Tests for CompressedTensorsQuantFormat._copy_hf_files."""

    def test_copy_hf_files_copy_allowed_suffixes_when_source_has_mixed_files(self, quant_format, temp_dir):
        src_dir = os.path.join(temp_dir, "src")
        os.makedirs(src_dir)
        with open(os.path.join(src_dir, "config.json"), "w", encoding="utf-8") as f:
            f.write("{}")
        with open(os.path.join(src_dir, "weights.bin"), "w", encoding="utf-8") as f:
            f.write("skip")

        quant_format._copy_hf_files(src_dir, temp_dir)

        assert os.path.isfile(os.path.join(temp_dir, "config.json"))
        assert not os.path.isfile(os.path.join(temp_dir, "weights.bin"))

    def test_copy_hf_files_skip_index_json_when_present(self, quant_format, temp_dir):
        src_dir = os.path.join(temp_dir, "src")
        os.makedirs(src_dir)
        with open(os.path.join(src_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
            f.write("{}")

        quant_format._copy_hf_files(src_dir, temp_dir)

        assert not os.path.isfile(os.path.join(temp_dir, "model.safetensors.index.json"))


class TestCompressedTensorsQuantFormatEnsureConfigJsonExists:
    """Tests for CompressedTensorsQuantFormat._ensure_config_json_exists."""

    def test_ensure_config_json_exists_return_true_when_file_already_present(self, quant_format, temp_dir):
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("{}")

        assert quant_format._ensure_config_json_exists(config_path) is True

    def test_ensure_config_json_exists_copy_from_source_when_missing(self, temp_dir, writer_infra):
        src_dir = os.path.join(temp_dir, "source")
        os.makedirs(src_dir)
        with open(os.path.join(src_dir, "config.json"), "w", encoding="utf-8") as f:
            f.write('{"model_type": "test"}')

        save_dir = os.path.join(temp_dir, "save")
        os.makedirs(save_dir)
        ctx = ExportContext(save_directory=Path(save_dir), source_model_path=Path(src_dir))
        fmt = make_compressed_tensors_quant_format(ctx, writer_infra)

        assert fmt._ensure_config_json_exists(os.path.join(save_dir, "config.json")) is True


class TestCompressedTensorsQuantFormatUpdateConfigJson:
    """Tests for CompressedTensorsQuantFormat._update_config_json."""

    def test_update_config_json_inject_quantization_config_when_config_valid(self, quant_format, temp_dir):
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"model_type": "llama"}, f)

        quant_format._update_config_json(QuantizedModel())

        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)
        assert QUANTIZATION_CONFIG_NAME in data
        assert data["model_type"] == "llama"

    def test_update_config_json_raise_config_error_when_config_missing(self, quant_format):
        with pytest.raises(ConfigError, match="config.json not found"):
            quant_format._update_config_json(QuantizedModel())

    def test_update_config_json_raise_config_error_when_content_not_dict(self, export_ctx, writer_infra, temp_dir):
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("[]")

        json_reader_factory = MockJsonReaderFactoryInfra()
        bad_reader = MockJsonReader(config_path)
        bad_reader.set_data([])
        json_reader_factory.create_json_reader = lambda file_path: bad_reader
        fmt = make_compressed_tensors_quant_format(
            export_ctx,
            writer_infra,
            json_reader_factory_infra=json_reader_factory,
        )

        with pytest.raises(ConfigError, match="Invalid config.json content"):
            fmt._update_config_json(QuantizedModel())


class TestCompressedTensorsQuantFormatBuildQuantizationConfig:
    """Tests for CompressedTensorsQuantFormat._build_quantization_config."""

    def test_build_quantization_config_return_dict_when_model_has_qir_module(self, quant_format):
        result = quant_format._build_quantization_config(QuantizedModel())

        assert isinstance(result, dict)
        assert "config_groups" in result

    def test_build_quantization_config_raise_invalid_model_error_when_no_qir_module(self, quant_format):
        empty_model = nn.Sequential(nn.Linear(4, 2))

        with pytest.raises(InvalidModelError, match="No quantized QIR module found"):
            quant_format._build_quantization_config(empty_model)


class TestCompressedTensorsQuantFormatFinalizeExport:
    """Tests for CompressedTensorsQuantFormat.finalize_export."""

    def test_finalize_export_close_writer_when_export_completes(self, quant_format, temp_dir):
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"model_type": "test"}, f)

        writer = quant_format.safetensors_writer
        quant_format.finalize_export(QuantizedModel())

        assert writer.closed is True
        assert quant_format.safetensors_writer is None

    def test_finalize_export_close_writer_when_update_config_raises(self, quant_format):
        writer = quant_format.safetensors_writer

        with pytest.raises(ConfigError):
            quant_format.finalize_export(QuantizedModel())

        assert writer.closed is True
        assert quant_format.safetensors_writer is None


class TestCompressedTensorsQuantFormatBuildModuleHandlerMap:
    """Tests for CompressedTensorsQuantFormat.build_module_handler_map."""

    def test_build_module_handler_map_contain_qir_handlers_when_called(self, export_ctx, writer_infra):
        fmt = make_compressed_tensors_quant_format(export_ctx, writer_infra)
        handler_map = fmt.build_module_handler_map()

        assert qir.W8A8StaticFakeQuantLinear in handler_map
        assert qir.W8A8DynamicPerChannelFakeQuantLinear in handler_map
        assert nn.Linear in handler_map
