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

量化保存格式导出。
"""

from __future__ import annotations

import importlib
from typing import Any

# 侧效应：加载 registry 子模块会执行 QuantFormatFactory.install()，向 processor.save.registry 注册内置格式。
importlib.import_module("msmodelslim.core.quantization_format.registry")

# 符号由 __getattr__ 惰性加载；此处仅声明公开 API。
# pylint: disable=undefined-all-variable
__all__ = [
    "CompressedTensorsQuantFormatConfig",
    "CompressedTensorsQuantFormat",
    "AscendV1QuantFormatConfig",
    "MindIEQuantFormatConfig",
]


def __getattr__(name: str) -> Any:
    """惰性导出 compressed tensors，避免与 ``infra.export`` / ``QuantFormat`` 形成导入环。"""
    if name == "CompressedTensorsQuantFormat":
        from msmodelslim.core.quantization_format.compressed_tensors_format.compressed_tensors import (
            CompressedTensorsQuantFormat,
        )

        return CompressedTensorsQuantFormat
    if name == "CompressedTensorsQuantFormatConfig":
        from msmodelslim.core.quantization_format.compressed_tensors_format.compressed_tensors import (
            CompressedTensorsQuantFormatConfig,
        )

        return CompressedTensorsQuantFormatConfig

    if name == "AscendV1QuantFormatConfig":
        from msmodelslim.core.quantization_format.ascendV1_format.ascendV1 import AscendV1QuantFormatConfig

        return AscendV1QuantFormatConfig
    if name == "MindIEQuantFormatConfig":
        from msmodelslim.core.quantization_format.mindie_format.mindie import MindIEQuantFormatConfig

        return MindIEQuantFormatConfig

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
