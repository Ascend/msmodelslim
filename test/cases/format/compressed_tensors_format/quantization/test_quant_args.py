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

Unit tests for msmodelslim.format.compressed_tensors_format.quantization.quant_args.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from msmodelslim.format.compressed_tensors_format.quantization.quant_args import (
    ActivationOrdering,
    DynamicType,
    FP8_E4M3_DATA,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from msmodelslim.utils.exception import SchemaValidateError


class TestQuantizationArgs:
    """Tests for QuantizationArgs validation and helpers."""

    def test_quantization_args_infer_tensor_strategy_when_group_size_unset(self):
        args = QuantizationArgs(num_bits=8, type=QuantizationType.INT)

        assert args.strategy == QuantizationStrategy.TENSOR

    def test_quantization_args_infer_channel_strategy_when_group_size_is_minus_one(
        self,
    ):
        args = QuantizationArgs(num_bits=8, type=QuantizationType.INT, group_size=-1)

        assert args.strategy == QuantizationStrategy.CHANNEL

    def test_quantization_args_raise_schema_validate_error_when_token_static(self):
        with pytest.raises(SchemaValidateError, match="static token quantization"):
            QuantizationArgs(
                num_bits=8,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.TOKEN,
                dynamic=False,
            )

    def test_quantization_args_raise_schema_validate_error_when_group_size_invalid(
        self,
    ):
        with pytest.raises(SchemaValidateError, match="Invalid group size"):
            QuantizationArgs(num_bits=8, type=QuantizationType.INT, group_size=-2)

    def test_quantization_args_pytorch_dtype_return_int8_when_int8_bits(self):
        args = QuantizationArgs(num_bits=8, type=QuantizationType.INT, strategy=QuantizationStrategy.TENSOR)

        assert args.pytorch_dtype() == torch.int8

    def test_quantization_args_block_structure_parse_string_when_valid(self):
        args = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.BLOCK,
            block_structure="128x128",
        )

        assert args.block_structure == [128, 128]

    def test_quantization_args_raise_schema_validate_error_when_block_structure_invalid(
        self,
    ):
        with pytest.raises(SchemaValidateError, match="Invalid block_structure"):
            QuantizationArgs(
                num_bits=8,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.BLOCK,
                block_structure="bad",
            )

    def test_quantization_args_infer_group_strategy_when_group_size_positive(self):
        args = QuantizationArgs(num_bits=8, type=QuantizationType.INT, group_size=128)

        assert args.strategy == QuantizationStrategy.GROUP

    def test_quantization_args_validate_type_lowercase_string_when_passed(self):
        args = QuantizationArgs(num_bits=8, type="int")

        assert args.type == QuantizationType.INT

    def test_quantization_args_validate_strategy_lowercase_string_when_passed(self):
        args = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy="tensor",
        )

        assert args.strategy == QuantizationStrategy.TENSOR

    def test_quantization_args_validate_actorder_bool_true_when_passed(self):
        args = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.GROUP,
            group_size=128,
            actorder=True,
        )

        assert args.actorder == ActivationOrdering.GROUP

    def test_quantization_args_validate_actorder_string_when_passed(self):
        args = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.GROUP,
            group_size=128,
            actorder="weight",
        )

        assert args.actorder == ActivationOrdering.WEIGHT

    def test_quantization_args_validate_dynamic_string_local_when_passed(self):
        args = QuantizationArgs(
            num_bits=4,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.TENSOR_GROUP,
            group_size=32,
            dynamic="local",
        )

        assert args.dynamic == DynamicType.LOCAL
        assert args.observer == "minmax"

    def test_quantization_args_block_structure_parse_list_when_valid(self):
        args = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.BLOCK,
            block_structure=[64, 64],
        )

        assert args.block_structure == [64, 64]

    def test_quantization_args_raise_schema_validate_error_when_block_structure_list_invalid(
        self,
    ):
        with pytest.raises(SchemaValidateError, match="Invalid block_structure"):
            QuantizationArgs(
                num_bits=8,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.BLOCK,
                block_structure=[64],
            )

    def test_quantization_args_raise_schema_validate_error_when_block_strategy_missing_structure(
        self,
    ):
        with pytest.raises(SchemaValidateError, match="Block strategy requires block structure"):
            QuantizationArgs(
                num_bits=8,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.BLOCK,
            )

    def test_quantization_args_raise_schema_validate_error_when_block_structure_without_strategy(
        self,
    ):
        with pytest.raises(SchemaValidateError, match="Block structure requires block strategy"):
            QuantizationArgs(
                num_bits=8,
                type=QuantizationType.INT,
                block_structure=[64, 64],
            )

    def test_quantization_args_raise_schema_validate_error_when_group_strategy_missing_group_size(
        self,
    ):
        with pytest.raises(SchemaValidateError, match="requires group_size"):
            QuantizationArgs(
                num_bits=8,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
            )

    def test_quantization_args_raise_schema_validate_error_when_group_size_without_group_strategy(
        self,
    ):
        with pytest.raises(SchemaValidateError, match="group_size requires strategy"):
            QuantizationArgs(
                num_bits=8,
                type=QuantizationType.INT,
                group_size=128,
                strategy=QuantizationStrategy.TENSOR,
            )

    def test_quantization_args_raise_schema_validate_error_when_actorder_on_tensor_strategy(
        self,
    ):
        with pytest.raises(SchemaValidateError, match="activation ordering"):
            QuantizationArgs(
                num_bits=8,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.TENSOR,
                actorder=True,
            )

    def test_quantization_args_raise_schema_validate_error_when_dynamic_unsupported_strategy(
        self,
    ):
        with pytest.raises(SchemaValidateError, match="must be used for dynamic quant"):
            QuantizationArgs(
                num_bits=8,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.CHANNEL,
                dynamic=True,
            )

    def test_quantization_args_raise_schema_validate_error_when_local_not_tensor_group(self):
        with pytest.raises(SchemaValidateError, match="local is only supported"):
            QuantizationArgs(
                num_bits=8,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.TENSOR,
                dynamic="local",
            )

    def test_quantization_args_warn_and_clear_observer_when_dynamic_true_and_observer_set(
        self,
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            args = QuantizationArgs(
                num_bits=8,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.TOKEN,
                dynamic=True,
                observer="minmax",
            )

        assert args.observer is None
        assert any("No observer is used for dynamic quant" in str(w.message) for w in caught)

    def test_quantization_args_set_default_observer_when_static(self):
        args = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.TENSOR,
            dynamic=False,
        )

        assert args.observer == "memoryless_minmax"

    def test_quantization_args_infer_zp_dtype_fp8_when_4bit_float(self):
        args = QuantizationArgs(
            num_bits=4,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.TENSOR_GROUP,
            group_size=32,
            dynamic="local",
        )

        assert args.zp_dtype is FP8_E4M3_DATA.dtype

    def test_quantization_args_serialize_dtype_return_none_when_symmetric(self):
        args = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.TENSOR,
            symmetric=True,
        )

        assert args.serialize_dtype(torch.int8) is None

    def test_quantization_args_serialize_dtype_return_string_when_asymmetric(self):
        args = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.TENSOR,
            symmetric=False,
            zp_dtype=torch.int8,
        )

        assert args.serialize_dtype(torch.int8) == "torch.int8"

    def test_quantization_args_pytorch_dtype_return_int16_when_16_bits(self):
        args = QuantizationArgs(num_bits=16, type=QuantizationType.INT, strategy=QuantizationStrategy.TENSOR)

        assert args.pytorch_dtype() == torch.int16

    def test_quantization_args_pytorch_dtype_return_int32_when_32_bits(self):
        args = QuantizationArgs(num_bits=32, type=QuantizationType.INT, strategy=QuantizationStrategy.TENSOR)

        assert args.pytorch_dtype() == torch.int32

    def test_quantization_args_pytorch_dtype_return_float8_when_float8_bits(self):
        args = QuantizationArgs(
            num_bits=8, type=QuantizationType.FLOAT, strategy=QuantizationStrategy.TENSOR_GROUP, group_size=32
        )

        dtype = args.pytorch_dtype()
        assert dtype == getattr(torch, "float8_e4m3fn", torch.float16)

    def test_quantization_args_raise_not_implemented_when_float_non_8_bits(self):
        args = QuantizationArgs(
            num_bits=4,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.TENSOR_GROUP,
            group_size=32,
            dynamic="local",
        )

        with pytest.raises(NotImplementedError, match="Only num_bits in \\(8\\)"):
            args.pytorch_dtype()

    def test_activation_ordering_get_aliases_when_called(self):
        aliases = ActivationOrdering.get_aliases()

        assert aliases["dynamic"] == "group"
        assert aliases["static"] == "weight"
