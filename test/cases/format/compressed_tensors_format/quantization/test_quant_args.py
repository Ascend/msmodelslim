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

Unit tests for msmodelslim.format.compressed_tensors_format.quantization.quant_args.
"""

from __future__ import annotations

import pytest

from msmodelslim.format.compressed_tensors_format.quantization.quant_args import (
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

        import torch

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
