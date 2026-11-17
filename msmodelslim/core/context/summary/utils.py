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
"""

import os
from typing import Any, Dict, Iterable, Optional

from msmodelslim.core.quantization_schema.kernels import WEIGHT_SCALE_INV_SUFFIX
from msmodelslim.core.quantization_schema.qal import QDType
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.security import json_safe_load

from .summary_context import SUMMARY_LOGGER_NAME

logger = get_logger(SUMMARY_LOGGER_NAME)

BASELINE_BITS = 16

SAFETENSORS_INDEX = "model.safetensors.index.json"
DEQUANT_SUFFIX_BITS = {WEIGHT_SCALE_INV_SUFFIX: 8}

QD_TYPE_BITS = {
    QDType.INT8: 8,
    QDType.FP8_E4M3: 8,
    QDType.MXFP8: 8,
    QDType.INT4: 4,
    QDType.MXFP4: 4,
}


def logical_bits(dtype: Any) -> Optional[int]:
    try:
        return QD_TYPE_BITS.get(QDType(dtype))
    except ValueError:
        return None


def format_bit_set(bits: Iterable[int]) -> str:
    values = sorted({int(value) for value in bits})
    if not values:
        return "%dbit" % BASELINE_BITS
    if len(values) == 1:
        return "%dbit" % values[0]
    return "{%s}bit" % ", ".join(str(value) for value in values)


def load_source_weight_bits(model_path: Any) -> Dict[str, int]:
    """返回源 checkpoint 中被反量化过的权重位宽 ``{模块路径: bits}``。

    适配器加载时会把带 block scale 的权重反量化成 bf16，这里只读 safetensors 索引拿回源位宽。
    """
    index_path = os.path.join(str(model_path or ""), SAFETENSORS_INDEX)
    try:
        weight_map = json_safe_load(index_path).get("weight_map")
    except Exception as error:  # pylint: disable=broad-except
        logger.debug("Failed to inspect source checkpoint weight bits: %s", error)
        return {}
    if not isinstance(weight_map, dict):
        return {}
    result: Dict[str, int] = {}
    for name in weight_map:
        for suffix, bits in DEQUANT_SUFFIX_BITS.items():
            if name.endswith(suffix):
                result[name[: -len(suffix)]] = bits
    return result
