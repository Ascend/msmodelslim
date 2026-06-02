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

按文件路径读取 JSON 对象（compressed-tensors ``config.json`` 等场景）。
"""

from __future__ import annotations

from typing import Any

from msmodelslim.format.compressed_tensors_format.compressed_tensors_json_reader_factory_infra import (
    CompressedTensorJsonReaderInfra,
)
from msmodelslim.utils.security import get_valid_read_path, json_safe_load


class JsonReader(CompressedTensorJsonReaderInfra):
    def __init__(self, file_path: str) -> None:
        self._file_path = file_path

    def load(self) -> dict[str, Any]:
        return json_safe_load(get_valid_read_path(self._file_path))


__all__ = ["JsonReader"]
