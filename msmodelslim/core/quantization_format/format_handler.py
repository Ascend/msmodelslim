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

Format recognition chain for fake-quant weight loading.

Each handler recognizes one export format and builds the corresponding
``IFormatLoader``. The default chain recognizes AscendV1 exports
(``quant_model_description.json``).
"""

import os
from abc import ABC, abstractmethod
from typing import Optional

from msmodelslim.utils.exception import SchemaValidateError

from .interface import IFormatLoader


class FormatHandler(ABC):
    """Chain-of-responsibility node that recognizes a weight format and builds a loader."""

    def __init__(self, successor: Optional["FormatHandler"] = None):
        self._successor = successor

    def set_next(self, handler: "FormatHandler") -> "FormatHandler":
        self._successor = handler
        return handler

    @abstractmethod
    def can_handle(self, model_path: str) -> bool:
        """Return True if this node recognizes ``model_path`` as its export format."""

    @abstractmethod
    def create(self, model_path: str, device: str = "cpu") -> IFormatLoader:
        """Build a loader for a path this node can handle."""

    def handle(self, model_path: str, device: str = "cpu") -> IFormatLoader:
        if self.can_handle(model_path):
            return self.create(model_path, device)
        if self._successor is not None:
            return self._successor.handle(model_path, device)
        raise SchemaValidateError(
            f"No fake-quant weight format handler recognized model_path={model_path}",
            action="Please provide a supported export directory "
            "(currently AscendV1 with quant_model_description.json).",
        )


class AscendV1FormatHandler(FormatHandler):
    """Recognizes an AscendV1 export directory and builds ``AscendV1Format``."""

    def can_handle(self, model_path: str) -> bool:
        from .ascendV1_format.format import ASCENDV1_DESC_JSON_NAME

        return os.path.isfile(os.path.join(model_path, ASCENDV1_DESC_JSON_NAME))

    def create(self, model_path: str, device: str = "cpu") -> IFormatLoader:
        from .ascendV1_format.format import AscendV1Format

        return AscendV1Format(model_path, device=device)


def build_default_format_chain() -> FormatHandler:
    """Default recognition chain. Append new handlers with ``set_next``."""
    return AscendV1FormatHandler()


__all__ = [
    "FormatHandler",
    "AscendV1FormatHandler",
    "build_default_format_chain",
]
