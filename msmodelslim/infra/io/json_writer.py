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

内存聚合若干 prefix→描述，最终在 ``close()`` 时写入单个 JSON（AscendV1 / MindIE 共用实现）。
"""

from __future__ import annotations

import os
from typing import Any
from msmodelslim.format.ascendV1_format.ascendV1_json_writer_factory_infra import (
    AscendV1JsonWriterInfra,
)
from msmodelslim.format.checkpoint_format.checkpoint_json_writer_factory_infra import (
    CheckpointJsonWriterInfra,
)
from msmodelslim.format.mindie_format.mindie_json_writer_factory_infra import (
    MindIEJsonWriterInfra,
)
from msmodelslim.format.compressed_tensors_format.compressed_tensors_json_writer_factory_infra import (
    CompressedTensorJsonWriterInfra,
)
from msmodelslim.utils.security import json_safe_dump


class JsonWriter(
    AscendV1JsonWriterInfra,
    MindIEJsonWriterInfra,
    CompressedTensorJsonWriterInfra,
    CheckpointJsonWriterInfra,
):
    def __init__(self, save_directory: str, file_name: str) -> None:
        self.save_directory = save_directory
        self.file_name = file_name
        self.value_map: dict[str, object] = {}

    def write(self, prefix: str, desc: object) -> None:
        self.value_map[prefix] = desc

    def close(self) -> None:
        json_safe_dump(self.value_map, os.path.join(self.save_directory, self.file_name), indent=4)

    def dump(self, data: dict[str, Any], *, indent: int = 2) -> None:
        json_safe_dump(data, os.path.join(self.save_directory, self.file_name), indent=indent)


__all__ = ["JsonWriter"]
