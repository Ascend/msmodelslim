#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You may use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------

Abstract base classes for per-IR format decoders.

Each concrete decoder handles the format-specific tensor reading and inverse mapping
for one IR class (or a small group of related IR classes sharing the same quantization
shape). The decoder produces IR-semantic DTOs (``IrStructureSpec`` for shapes, plain-tensor
``QuantParamSet`` for values); QParam / QStorage assembly is left to ``IrBuilder.bind_from_param_set``.

Three sub-bases reflect the three dispatch categories:
  - ``LinearIrDecoder``  : dispatch by label string (e.g. ``W8A8``)
  - ``Fa3IrDecoder``     : dispatch by ``QScheme``
  - ``KvCacheIrDecoder`` : dispatch by label string (e.g. ``C8``)
"""

from abc import ABC, abstractmethod
from typing import Optional, Type

import torch
from torch import nn

from msmodelslim.ir.qal import QScheme


class IrDecoder(ABC):
    """Common base for all per-IR decoders. Holds a reference to the tensor store."""

    def __init__(self, store) -> None:
        self.store = store


class LinearIrDecoder(IrDecoder):
    """Decoder for a Linear FakeQuant IR, dispatched by quant_type label string."""

    label: str

    @abstractmethod
    def ir_type(self, group_size: int = 0) -> Type[nn.Module]:
        """Map this label (and optional ``group_size``) to a FakeQuant Linear IR class."""

    @abstractmethod
    def structure(self, prefix: str, group_size: int = 0):
        """Build IR-semantic shapes/flags from export headers under ``prefix``."""

    @abstractmethod
    def params(self, prefix: str, device: torch.device, group_size: int = 0):
        """Decode export tensors under ``prefix`` into a plain-tensor ``QuantParamSet``."""


class Fa3IrDecoder(IrDecoder):
    """Decoder for a FA3 activation FakeQuant IR, dispatched by ``QScheme``.

    One concrete decoder may handle multiple schemes for the same quantization shape
    (e.g. ``Fa3PerTokenDecoder`` handles both ``int8_per_token_sym`` and
    ``fp8_e4m3_per_token_sym``). ``ir_type(scheme)`` selects the IR class; the scheme
    itself is carried straight into the built structure/param DTOs.
    """

    @abstractmethod
    def ir_type(self, scheme: QScheme) -> Type[nn.Module]:
        """Map ``scheme`` to the activation IR class for this decoder."""

    @abstractmethod
    def structure(self, prefix: str, group_size: int = 0, scheme: Optional[QScheme] = None):
        """Build IR-semantic shapes/flags from export headers under ``prefix``."""

    @abstractmethod
    def params(self, prefix: str, device: torch.device, group_size: int = 0, scheme: Optional[QScheme] = None):
        """Decode export tensors under ``prefix`` into a plain-tensor ``QuantParamSet``."""


class KvCacheIrDecoder(IrDecoder):
    """Decoder for a KV-cache FakeQuant IR, dispatched by kv_cache_type label string."""

    label: str
    ir_type: Type[nn.Module]

    @abstractmethod
    def structure(self, prefix: str, group_size: int = 0):
        """Build IR-semantic shapes/flags from export headers under ``prefix``."""

    @abstractmethod
    def params(self, prefix: str, device: torch.device, group_size: int = 0):
        """Decode export tensors under ``prefix`` into a plain-tensor ``QuantParamSet``."""


__all__ = [
    "IrDecoder",
    "LinearIrDecoder",
    "Fa3IrDecoder",
    "KvCacheIrDecoder",
]
