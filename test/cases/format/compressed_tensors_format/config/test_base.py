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

Unit tests for msmodelslim.format.compressed_tensors_format.config.base.
"""

from __future__ import annotations

from msmodelslim.format.compressed_tensors_format.config.base import (
    COMPRESSION_VERSION_NAME,
    QUANTIZATION_CONFIG_NAME,
    QUANTIZATION_METHOD_NAME,
    SPARSITY_CONFIG_NAME,
    TRANSFORM_CONFIG_NAME,
    QuantizationFormat,
)


class TestQuantizationFormat:
    """Tests for QuantizationFormat enum."""

    def test_quantization_format_int_quantized_value_when_accessed(self):
        assert QuantizationFormat.int_quantized.value == "int-quantized"

    def test_quantization_format_mixed_precision_value_when_accessed(self):
        assert QuantizationFormat.mixed_precision.value == "mixed-precision"

    def test_quantization_format_members_are_unique_when_enumerated(self):
        values = [member.value for member in QuantizationFormat]
        assert len(values) == len(set(values))


class TestConfigConstants:
    """Tests for compressed-tensors config field name constants."""

    def test_config_constants_match_expected_names_when_imported(self):
        assert QUANTIZATION_CONFIG_NAME == "quantization_config"
        assert SPARSITY_CONFIG_NAME == "sparsity_config"
        assert TRANSFORM_CONFIG_NAME == "transform_config"
        assert COMPRESSION_VERSION_NAME == "version"
        assert QUANTIZATION_METHOD_NAME == "quant_method"
