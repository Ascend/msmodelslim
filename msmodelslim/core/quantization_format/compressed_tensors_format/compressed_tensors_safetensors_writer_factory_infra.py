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

from abc import ABC, abstractmethod

import torch


class CompressedTensorSafetensorsWriterInfra(ABC):
    @abstractmethod
    def write(self, key: str, value: torch.Tensor) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class CompressedTensorSafetensorsWriterFactoryInfra(ABC):
    @abstractmethod
    def create_safetensors_writer(
        self, part_file_size: int, save_directory: str, save_prefix: str
    ) -> CompressedTensorSafetensorsWriterInfra:
        pass


__all__ = ["CompressedTensorSafetensorsWriterFactoryInfra", "CompressedTensorSafetensorsWriterInfra"]
