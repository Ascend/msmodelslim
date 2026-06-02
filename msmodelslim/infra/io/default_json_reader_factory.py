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

from __future__ import annotations

from msmodelslim.format.compressed_tensors_format.compressed_tensors_json_reader_factory_infra import (
    CompressedTensorJsonReaderFactoryInfra,
    CompressedTensorJsonReaderInfra,
)
from msmodelslim.infra.io.json_reader import JsonReader


class DefaultJsonReaderFactory(CompressedTensorJsonReaderFactoryInfra):
    def create_json_reader(self, file_path: str) -> CompressedTensorJsonReaderInfra:
        return JsonReader(file_path)


__all__ = ["DefaultJsonReaderFactory"]
