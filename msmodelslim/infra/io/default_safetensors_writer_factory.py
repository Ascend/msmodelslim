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

import os

from msmodelslim import logger
from msmodelslim.format.ascendV1_format.ascendV1_tensors_writer_factory_infra import (
    AscendV1SafetensorsWriterFactoryInfra,
    AscendV1SafetensorsWriterInfra,
)
from msmodelslim.format.compressed_tensors_format.compressed_tensors_safetensors_writer_factory_infra import (
    CompressedTensorSafetensorsWriterFactoryInfra,
    CompressedTensorSafetensorsWriterInfra,
)
from msmodelslim.format.mindie_format.mindie_tensors_writer_factory_infra import (
    MindIESafetensorsWriterFactoryInfra,
    MindIESafetensorsWriterInfra,
)


class DefaultSafetensorsWriterFactory(
    AscendV1SafetensorsWriterFactoryInfra,
    MindIESafetensorsWriterFactoryInfra,
    CompressedTensorSafetensorsWriterFactoryInfra,
):
    def create_safetensors_writer(
        self,
        part_file_size: int,
        save_directory: str,
        save_prefix: str,
    ) -> AscendV1SafetensorsWriterInfra | MindIESafetensorsWriterInfra | CompressedTensorSafetensorsWriterInfra:
        from msmodelslim.infra.io.buffered_safetensors_writer import BufferedSafetensorsWriter
        from msmodelslim.infra.io.safetensors_writer import SafetensorsWriter

        if part_file_size is not None and float(part_file_size) > 0:
            return BufferedSafetensorsWriter(
                logger=logger,
                max_gb_size=int(part_file_size),
                save_directory=save_directory,
                save_prefix=save_prefix,
            )
        return SafetensorsWriter(
            logger=logger,
            file_path=os.path.join(save_directory, f"{save_prefix}.safetensors"),
        )


__all__ = ["DefaultSafetensorsWriterFactory"]
