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

dtype_ops：按数据格式分文件，按量化粒度分接口。
公开入口是模块级函数；每种格式一个类，类不进 __all__。
函数名把 sym / asym 放在最后。quantize 默认 float32。

__init__.py  只导出模块级函数
  ├─ int8_ops   Int8
  ├─ int4_ops   Int4
  ├─ fp8_ops    Fp8
  ├─ mxfp8_ops  Mxfp8
  └─ mxfp4_ops  Mxfp4

粒度后缀：INT / FP8 为 per_tensor / per_channel / per_group，再接 _sym / _asym。
FP8 没有 per_group，也没有非对称。MX 不加粒度后缀，只接 _sym。
FP8 / MXFP8 的 fmt 默认 "e4m3"，可选 "e5m2"，选元素编码而非存储类型。
INT8 / INT4 / MXFP4 无 fmt；MXFP4 元素固定 E2M1。
"""

from .fp8_ops import (
    fp8_calc_qparam_per_channel_sym,
    fp8_calc_qparam_per_tensor_sym,
    fp8_dequantize_per_channel_sym,
    fp8_dequantize_per_tensor_sym,
    fp8_quantize_dynamic_per_channel_sym,
    fp8_quantize_dynamic_per_tensor_sym,
    fp8_quantize_per_channel_sym,
    fp8_quantize_per_tensor_sym,
)
from .int4_ops import (
    int4_calc_qparam_per_channel_asym,
    int4_calc_qparam_per_channel_sym,
    int4_calc_qparam_per_group_asym,
    int4_calc_qparam_per_group_sym,
    int4_calc_qparam_per_tensor_asym,
    int4_calc_qparam_per_tensor_sym,
    int4_dequantize_per_channel_asym,
    int4_dequantize_per_channel_sym,
    int4_dequantize_per_group_asym,
    int4_dequantize_per_group_sym,
    int4_dequantize_per_tensor_asym,
    int4_dequantize_per_tensor_sym,
    int4_quantize_dynamic_per_channel_asym,
    int4_quantize_dynamic_per_channel_sym,
    int4_quantize_dynamic_per_group_asym,
    int4_quantize_dynamic_per_group_sym,
    int4_quantize_dynamic_per_tensor_asym,
    int4_quantize_dynamic_per_tensor_sym,
    int4_quantize_per_channel_asym,
    int4_quantize_per_channel_sym,
    int4_quantize_per_group_asym,
    int4_quantize_per_group_sym,
    int4_quantize_per_tensor_asym,
    int4_quantize_per_tensor_sym,
)
from .int8_ops import (
    int8_calc_qparam_per_channel_asym,
    int8_calc_qparam_per_channel_sym,
    int8_calc_qparam_per_group_asym,
    int8_calc_qparam_per_group_sym,
    int8_calc_qparam_per_tensor_asym,
    int8_calc_qparam_per_tensor_sym,
    int8_dequantize_per_channel_asym,
    int8_dequantize_per_channel_sym,
    int8_dequantize_per_group_asym,
    int8_dequantize_per_group_sym,
    int8_dequantize_per_tensor_asym,
    int8_dequantize_per_tensor_sym,
    int8_quantize_dynamic_per_channel_asym,
    int8_quantize_dynamic_per_channel_sym,
    int8_quantize_dynamic_per_group_asym,
    int8_quantize_dynamic_per_group_sym,
    int8_quantize_dynamic_per_tensor_asym,
    int8_quantize_dynamic_per_tensor_sym,
    int8_quantize_per_channel_asym,
    int8_quantize_per_channel_sym,
    int8_quantize_per_group_asym,
    int8_quantize_per_group_sym,
    int8_quantize_per_tensor_asym,
    int8_quantize_per_tensor_sym,
)
from .mxfp4_ops import (
    mxfp4_calc_qparam_sym,
    mxfp4_dequantize_sym,
    mxfp4_quantize_dynamic_sym,
    mxfp4_quantize_sym,
)
from .mxfp8_ops import (
    mxfp8_calc_qparam_sym,
    mxfp8_dequantize_sym,
    mxfp8_quantize_dynamic_sym,
    mxfp8_quantize_sym,
)

__all__ = [
    "int8_quantize_per_tensor_sym",
    "int8_quantize_per_channel_sym",
    "int8_quantize_per_group_sym",
    "int8_dequantize_per_tensor_sym",
    "int8_dequantize_per_channel_sym",
    "int8_dequantize_per_group_sym",
    "int8_quantize_dynamic_per_tensor_sym",
    "int8_quantize_dynamic_per_channel_sym",
    "int8_quantize_dynamic_per_group_sym",
    "int8_calc_qparam_per_tensor_sym",
    "int8_calc_qparam_per_channel_sym",
    "int8_calc_qparam_per_group_sym",
    "int8_quantize_per_tensor_asym",
    "int8_quantize_per_channel_asym",
    "int8_quantize_per_group_asym",
    "int8_dequantize_per_tensor_asym",
    "int8_dequantize_per_channel_asym",
    "int8_dequantize_per_group_asym",
    "int8_quantize_dynamic_per_tensor_asym",
    "int8_quantize_dynamic_per_channel_asym",
    "int8_quantize_dynamic_per_group_asym",
    "int8_calc_qparam_per_tensor_asym",
    "int8_calc_qparam_per_channel_asym",
    "int8_calc_qparam_per_group_asym",
    "int4_quantize_per_tensor_sym",
    "int4_quantize_per_channel_sym",
    "int4_quantize_per_group_sym",
    "int4_dequantize_per_tensor_sym",
    "int4_dequantize_per_channel_sym",
    "int4_dequantize_per_group_sym",
    "int4_quantize_dynamic_per_tensor_sym",
    "int4_quantize_dynamic_per_channel_sym",
    "int4_quantize_dynamic_per_group_sym",
    "int4_calc_qparam_per_tensor_sym",
    "int4_calc_qparam_per_channel_sym",
    "int4_calc_qparam_per_group_sym",
    "int4_quantize_per_tensor_asym",
    "int4_quantize_per_channel_asym",
    "int4_quantize_per_group_asym",
    "int4_dequantize_per_tensor_asym",
    "int4_dequantize_per_channel_asym",
    "int4_dequantize_per_group_asym",
    "int4_quantize_dynamic_per_tensor_asym",
    "int4_quantize_dynamic_per_channel_asym",
    "int4_quantize_dynamic_per_group_asym",
    "int4_calc_qparam_per_tensor_asym",
    "int4_calc_qparam_per_channel_asym",
    "int4_calc_qparam_per_group_asym",
    "fp8_quantize_per_tensor_sym",
    "fp8_quantize_per_channel_sym",
    "fp8_dequantize_per_tensor_sym",
    "fp8_dequantize_per_channel_sym",
    "fp8_quantize_dynamic_per_tensor_sym",
    "fp8_quantize_dynamic_per_channel_sym",
    "fp8_calc_qparam_per_tensor_sym",
    "fp8_calc_qparam_per_channel_sym",
    "mxfp8_quantize_sym",
    "mxfp8_dequantize_sym",
    "mxfp8_quantize_dynamic_sym",
    "mxfp8_calc_qparam_sym",
    "mxfp4_quantize_sym",
    "mxfp4_dequantize_sym",
    "mxfp4_quantize_dynamic_sym",
    "mxfp4_calc_qparam_sym",
]
