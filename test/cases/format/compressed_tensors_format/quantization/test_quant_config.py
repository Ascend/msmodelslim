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

Unit tests for msmodelslim.format.compressed_tensors_format.quantization.quant_config.
"""

from __future__ import annotations

from torch import nn

from msmodelslim.format.compressed_tensors_format.config.base import (
    COMPRESSION_VERSION_NAME,
    DEFAULT_COMPRESSION_VERSION,
    QUANTIZATION_METHOD_NAME,
    QUANTIZATION_CONFIG_NAME,
)
from msmodelslim.format.compressed_tensors_format.quantization.quant_config import (
    DEFAULT_QUANTIZATION_METHOD,
    QuantizationConfig,
    QuantizationStatus,
)
from msmodelslim.format.compressed_tensors_format.quantization.quant_scheme import (
    preset_name_to_scheme,
)

from test.cases.format.compressed_tensors_format.helpers import QuantizedModel


class TestQuantizationStatus:
    """Tests for QuantizationStatus enum."""

    def test_quantization_status_compressed_value_when_accessed(self):
        assert QuantizationStatus.COMPRESSED.value == "compressed"


class TestQuantizationConfig:
    """Tests for QuantizationConfig."""

    def test_quantization_config_resolve_preset_when_config_groups_has_target_list(
        self,
    ):
        config = QuantizationConfig(
            config_groups={"W8A8_STATIC": ["Linear"]},
        )

        scheme = config.config_groups["W8A8_STATIC"]
        assert scheme.targets == ["Linear"]
        assert scheme.weights.num_bits == 8

    def test_quantization_config_from_model_return_config_when_model_has_qir_module(
        self,
    ):
        model = QuantizedModel()

        config = QuantizationConfig.from_model(model)

        assert config is not None
        assert config.quantization_status == QuantizationStatus.COMPRESSED
        assert len(config.config_groups) >= 1

    def test_quantization_config_from_model_return_none_when_model_has_no_qir_module(
        self,
    ):
        model = nn.Sequential(nn.Linear(4, 2))

        assert QuantizationConfig.from_model(model) is None

    def test_quantization_config_to_dict_include_metadata_when_called(self):
        scheme = preset_name_to_scheme("W8A8_STATIC", ["Linear"])
        config = QuantizationConfig(config_groups={"group_0": scheme})

        result = config.to_quantization_config_dict()

        assert result[QUANTIZATION_METHOD_NAME] == DEFAULT_QUANTIZATION_METHOD
        assert result[COMPRESSION_VERSION_NAME] == DEFAULT_COMPRESSION_VERSION
        assert QUANTIZATION_CONFIG_NAME not in result
        assert "config_groups" in result
