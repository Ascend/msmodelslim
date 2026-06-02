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

from msmodelslim.format.ascendV1_format.ascendV1_json_writer_factory_infra import (
    AscendV1JsonWriterFactoryInfra,
    AscendV1JsonWriterInfra,
)
from msmodelslim.format.compressed_tensors_format.compressed_tensors_json_writer_factory_infra import (
    CompressedTensorJsonWriterFactoryInfra,
    CompressedTensorJsonWriterInfra,
)
from msmodelslim.format.mindie_format.mindie_json_writer_factory_infra import (
    MindIEJsonWriterFactoryInfra,
    MindIEJsonWriterInfra,
)
from msmodelslim.infra.io.json_writer import JsonWriter


class DefaultJsonWriterFactory(
    AscendV1JsonWriterFactoryInfra,
    MindIEJsonWriterFactoryInfra,
    CompressedTensorJsonWriterFactoryInfra,
):
    """描述 JSON 落盘：AscendV1/MindIE 使用聚合 JsonWriter，compressed-tensors 使用 ConfigJsonWriter。"""

    def create_json_writer(
        self,
        save_directory: str,
        file_name: str,
    ) -> AscendV1JsonWriterInfra | MindIEJsonWriterInfra | CompressedTensorJsonWriterInfra:
        return JsonWriter(save_directory, file_name)


__all__ = ["DefaultJsonWriterFactory"]
