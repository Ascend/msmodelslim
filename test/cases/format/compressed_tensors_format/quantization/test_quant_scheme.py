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

Unit tests for msmodelslim.format.compressed_tensors_format.quantization.quant_scheme.
"""

from __future__ import annotations

import pytest
from torch import nn

from msmodelslim.format.compressed_tensors_format.config.base import QuantizationFormat
from msmodelslim.format.compressed_tensors_format.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from msmodelslim.format.compressed_tensors_format.quantization.quant_scheme import (
    QuantizationScheme,
    preset_name_to_scheme,
    scheme_for_qir_module,
)
from msmodelslim.utils.exception import SchemaValidateError

from test.cases.format.compressed_tensors_format.helpers import (
    make_w8a8_dynamic_module,
    make_w8a8_static_module,
)


class TestQuantizationScheme:
    """Tests for QuantizationScheme validation."""

    def test_quantization_scheme_create_success_when_w8a8_static_preset_valid(self):
        scheme = preset_name_to_scheme("W8A8_STATIC", ["Linear"])

        assert scheme.targets == ["Linear"]
        assert scheme.weights.strategy == QuantizationStrategy.CHANNEL
        assert scheme.input_activations.strategy == QuantizationStrategy.TENSOR

    def test_quantization_scheme_raise_value_error_when_input_actorder_set(self):
        with pytest.raises(SchemaValidateError, match="actorder"):
            QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.INT,
                    strategy=QuantizationStrategy.CHANNEL,
                ),
                input_activations=QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.INT,
                    strategy=QuantizationStrategy.TENSOR,
                    actorder=True,
                ),
            )

    def test_quantization_scheme_raise_value_error_when_format_is_mixed_precision(self):
        with pytest.raises(SchemaValidateError, match="mixed-precision"):
            QuantizationScheme(
                targets=["Linear"],
                format=QuantizationFormat.mixed_precision,
            )


class TestPresetNameToScheme:
    """Tests for preset_name_to_scheme."""

    def test_preset_name_to_scheme_return_scheme_when_name_is_lowercase(self):
        scheme = preset_name_to_scheme("w8a8_static", ["Linear"])

        assert scheme.weights.num_bits == 8

    def test_preset_name_to_scheme_raise_schema_validate_error_when_name_unknown(self):
        with pytest.raises(SchemaValidateError, match="Unknown preset scheme name"):
            preset_name_to_scheme("NOT_A_PRESET", ["Linear"])


class TestSchemeForQirModule:
    """Tests for scheme_for_qir_module."""

    def test_scheme_for_qir_module_return_scheme_when_static_qir_module(self):
        module = make_w8a8_static_module()

        scheme = scheme_for_qir_module(module)

        assert scheme is not None
        assert scheme.input_activations.dynamic is False

    def test_scheme_for_qir_module_return_scheme_when_dynamic_qir_module(self):
        module = make_w8a8_dynamic_module()

        scheme = scheme_for_qir_module(module)

        assert scheme is not None
        assert scheme.input_activations.dynamic is True

    def test_scheme_for_qir_module_return_none_when_module_unsupported(self):
        module = nn.Linear(4, 2)

        assert scheme_for_qir_module(module) is None
