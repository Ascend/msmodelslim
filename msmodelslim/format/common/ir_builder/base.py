#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Type

from torch import nn

from msmodelslim.ir.qal.qregistry import QABCRegistry


class ApplyMode(Enum):
    REPLACE = "replace"
    INSERT = "insert"
    SKIP = "skip"


@dataclass
class TransformResult:
    mode: ApplyMode
    module: Optional[nn.Module] = None


class IrStructureSpec:
    """IR-semantic structure description (shapes / flags). Defined per helper; no Format keys."""


class IrParamBundle:
    """IR-semantic parameter payload (QParam / QStorage / tensors). Defined per helper."""


def set_module(root: nn.Module, submodule_key: str, module: nn.Module) -> None:
    """Replace a submodule of ``root`` identified by dotted ``submodule_key``."""
    tokens = submodule_key.split(".")
    parent = root
    for token in tokens[:-1]:
        parent = getattr(parent, token)
    setattr(parent, tokens[-1], module)


@QABCRegistry.register_abc(dispatch_key=Type[nn.Module])
class IrBuilder(ABC):
    """Bound to one IR type via QABCRegistry (dispatch_key = IR class). No Format types in signature."""

    @classmethod
    def create(cls, ir_type: Type[nn.Module], *args, **kwargs) -> "IrBuilder":
        """Resolve IrBuilder implementation registered for ``ir_type``."""
        return QABCRegistry.create(IrBuilder, ir_type, *args, **kwargs)

    @classmethod
    def supports_fake_quant_reload(cls) -> bool:
        return True

    @abstractmethod
    def transform_shell(
        self,
        model: nn.Module,
        module_name: str,
        source_module: nn.Module,
        *,
        structure: Any,
    ) -> TransformResult:
        """Turn a transformers float module into this IR's quantized structure."""
        pass

    @abstractmethod
    def bind_from_param_set(self, ir_module: nn.Module, param_set: Any) -> None:
        """Fill an already-transformed IR module from a plain-tensor ``QuantParamSet``.

        The helper assembles QParam / QStorage from the plain tensors (scheme selection
        is IR knowledge), constructs a filled temporary module, and ``copy_`` params
        into ``ir_module``. The decoder only produces plain tensors; no QParam/QStorage
        knowledge leaks to the Format side.
        """
        pass
