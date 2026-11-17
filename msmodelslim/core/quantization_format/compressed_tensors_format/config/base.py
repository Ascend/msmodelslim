#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions of this file (QuantizationFormat) are derived from
# compressed_tensors.config.base.CompressionFormat
# (https://github.com/neuralmagic/compressed-tensors).
# See msmodelslim/Third_Party_Open_Source_Software_Notice.
#
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.

from enum import Enum, unique

QUANTIZATION_CONFIG_NAME = "quantization_config"
SPARSITY_CONFIG_NAME = "sparsity_config"
TRANSFORM_CONFIG_NAME = "transform_config"
COMPRESSION_VERSION_NAME = "version"
QUANTIZATION_METHOD_NAME = "quant_method"
# Written to config.json ``version``; aligned with compressed-tensors schema.
DEFAULT_COMPRESSION_VERSION = "0.13.0"


@unique
class QuantizationFormat(str, Enum):
    dense = "dense"
    sparse_bitmask = "sparse-bitmask"
    sparse_24_bitmask = "sparse-24-bitmask"
    int_quantized = "int-quantized"
    float_quantized = "float-quantized"
    naive_quantized = "naive-quantized"
    pack_quantized = "pack-quantized"
    marlin_24 = "marlin-24"
    mixed_precision = "mixed-precision"
    nvfp4_pack_quantized = "nvfp4-pack-quantized"
    mxfp4_pack_quantized = "mxfp4-pack-quantized"
    mxfp8_quantized = "mxfp8-quantized"


__all__ = [
    "QuantizationFormat",
    "QUANTIZATION_CONFIG_NAME",
    "SPARSITY_CONFIG_NAME",
    "TRANSFORM_CONFIG_NAME",
    "COMPRESSION_VERSION_NAME",
    "QUANTIZATION_METHOD_NAME",
    "DEFAULT_COMPRESSION_VERSION",
]
