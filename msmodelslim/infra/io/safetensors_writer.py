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

Safetensors 单文件写入（AscendV1 / MindIE / CompressedTensors 共用实现）。
"""

from __future__ import annotations

import torch
from safetensors.torch import save_file

from msmodelslim.format.ascendV1_format.ascendV1_tensors_writer_factory_infra import (
    AscendV1SafetensorsWriterInfra,
)
from msmodelslim.format.compressed_tensors_format.compressed_tensors_safetensors_writer_factory_infra import (
    CompressedTensorSafetensorsWriterInfra,
)
from msmodelslim.format.mindie_format.mindie_tensors_writer_factory_infra import (
    MindIESafetensorsWriterInfra,
)
from msmodelslim.utils.security import SafeWriteUmask, get_valid_write_path


class SafetensorsWriter(
    AscendV1SafetensorsWriterInfra,
    CompressedTensorSafetensorsWriterInfra,
    MindIESafetensorsWriterInfra,
):
    def __init__(self, logger, file_path: str) -> None:
        self.logger = logger
        file_path = get_valid_write_path(file_path, extensions=[".safetensors"])
        self.file_path = file_path
        self.safetensors_weight: dict[str, torch.Tensor] = {}

    def write(self, key: str, value: torch.Tensor) -> None:
        self.safetensors_weight[key] = value.cpu().contiguous()

    def close(self) -> None:
        with SafeWriteUmask(umask=0o377):
            save_file(self.safetensors_weight, self.file_path)
        self.logger.info(f"Save safetensors to {self.file_path} successfully")


__all__ = ["SafetensorsWriter"]
