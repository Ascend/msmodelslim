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

Unit tests for msmodelslim.format.compressed_tensors_format.quantization.quant_scheme.
"""

from __future__ import annotations

# pylint: disable=no-name-in-module

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
        with pytest.raises(SchemaValidateError, match="activation ordering"):
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

    def test_preset_name_to_scheme_return_dynamic_scheme_when_w8a8_dynamic(self):
        scheme = preset_name_to_scheme("W8A8_DYNAMIC", ["Linear"])

        assert scheme.input_activations.dynamic is True
        assert scheme.input_activations.strategy == QuantizationStrategy.TOKEN

    def test_quantization_scheme_raise_not_implemented_when_group_dynamic_activation(self):
        with pytest.raises(NotImplementedError, match="group-wise activation quantization"):
            QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.INT,
                    strategy=QuantizationStrategy.GROUP,
                    group_size=128,
                ),
                input_activations=QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.INT,
                    strategy=QuantizationStrategy.GROUP,
                    group_size=128,
                    dynamic=True,
                ),
            )

    def test_quantization_scheme_raise_not_implemented_when_unsupported_activation_strategy(self):
        with pytest.raises(NotImplementedError, match="not supported for activation quantization"):
            QuantizationScheme(
                targets=["Linear"],
                input_activations=QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.INT,
                    strategy=QuantizationStrategy.CHANNEL,
                ),
            )

    def test_quantization_scheme_raise_value_error_when_output_actorder_set(self):
        with pytest.raises(SchemaValidateError, match="actorder to output activations"):
            QuantizationScheme(
                targets=["Linear"],
                output_activations=QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.INT,
                    strategy=QuantizationStrategy.GROUP,
                    group_size=128,
                    actorder=True,
                ),
            )

    def test_quantization_scheme_warn_when_group_sizes_differ(self, mocker):
        warn_mock = mocker.patch(
            "msmodelslim.format.compressed_tensors_format.quantization.quant_scheme.logger.warning"
        )

        QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=8,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                group_size=128,
            ),
            input_activations=QuantizationArgs(
                num_bits=8,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                group_size=64,
            ),
        )

        assert any("different group sizes" in str(call.args[0]) for call in warn_mock.call_args_list)
