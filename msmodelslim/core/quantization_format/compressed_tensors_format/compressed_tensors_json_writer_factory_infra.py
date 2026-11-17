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
from typing import Any


class CompressedTensorJsonWriterInfra(ABC):
    @abstractmethod
    def dump(self, data: dict[str, Any], *, indent: int = 2) -> None:
        pass


class CompressedTensorJsonWriterFactoryInfra(ABC):
    @abstractmethod
    def create_json_writer(self, save_directory: str, file_name: str) -> CompressedTensorJsonWriterInfra:
        pass


__all__ = ["CompressedTensorJsonWriterFactoryInfra", "CompressedTensorJsonWriterInfra"]
