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
"""

from enum import Enum


class DeviceType(str, Enum):
    NPU = "npu"  # 昇腾NPU
    CPU = "cpu"  # CPU


class QuantType(str, Enum):
    W4A4 = "w4a4"  # 权重INT4量化，激活值INT4量化
    W4A8 = "w4a8"  # 权重INT4量化，激活值INT8量化
    W4A4C8 = "w4a4c8"  # 权重4bit量化，激活值4bit量化，KVCache 8bit量化
    W4A4F8 = "w4a4f8"  # 权重4bit量化，激活值4bit量化，FA3量化
    W4A8C8 = "w4a8c8"  # 权重INT4量化，激活值INT8量化，KVCache INT8量化
    W8A16 = "w8a16"  # 权重INT8量化，激活值不量化
    W8A8 = "w8a8"  # 权重INT8量化，激活值INT8量化
    W8A8S = "w8a8s"  # 权重INT8稀疏量化，激活值INT8量化
    W8A8C8 = "w8a8c8"  # 权重INT8量化，激活值INT8量化，KVCache INT8量化
    W8A8F8 = "w8a8f8"  # 权重INT8量化，激活值INT8量化，FA3量化
    W16A16S = "w16a16s"  # 权重浮点稀疏


class RunnerType(str, Enum):
    """Runner类型枚举"""

    AUTO = "auto"
    MODEL_WISE = "model_wise"
    LAYER_WISE = "layer_wise"
    DP_LAYER_WISE = "dp_layer_wise"
