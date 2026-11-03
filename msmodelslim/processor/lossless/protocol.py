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
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

from dataclasses import dataclass
from typing import Any, Tuple

from typing_extensions import Protocol, runtime_checkable

import torch

from msmodelslim.core.base.protocol import BatchProcessRequest


@dataclass(frozen=True)
class LosslessCheckTarget:
    """One structure whose runtime input and output must stay semantically equivalent."""

    name: str
    path: str
    decoder: str = ""
    structure_type: str = "Linear"
    check_input: bool = True
    check_output: bool = True

    def relative_path(self, request_name: str) -> str:
        """Return the target path relative to the current request module."""

        if not request_name:
            return self.path
        if self.path == request_name:
            return ""
        prefix = f"{request_name}."
        if self.path.startswith(prefix):
            return self.path[len(prefix) :]
        return self.path


@dataclass(frozen=True)
class StructureIORecord:
    """Captured input/output for one invocation of a target structure."""

    request_name: str
    structure_name: str
    invocation: int
    args: Tuple[Any, ...]
    kwargs: dict
    output: Any


@dataclass(frozen=True)
class LosslessCheckPlan:
    """Algorithm operations that a complete run is expected to cover."""

    expected_targets: Tuple[str, ...]


@dataclass(frozen=True)
class LosslessCheckContext:
    """Behavior-free data needed to check one scheduled request."""

    targets: Tuple[LosslessCheckTarget, ...]
    reference_datas: Any
    covered_targets: Tuple[str, ...]
    unresolved_targets: Tuple[str, ...]


@runtime_checkable
class LosslessCheckableProcessor(Protocol):
    """Internal contract implemented by processors accepted by the container."""

    def prepare_lossless_check(self) -> LosslessCheckPlan:
        """Freeze and return the theoretical plan before changing the model."""

    def create_lossless_check_context(
        self,
        request: BatchProcessRequest,
        reference_datas: Any,
    ) -> LosslessCheckContext:
        """Return request-local targets and restored reference input data."""

    def align_lossless_check_io(
        self,
        context: LosslessCheckContext,
        target: LosslessCheckTarget,
        reference: StructureIORecord,
        processed: StructureIORecord,
    ) -> Tuple[StructureIORecord, StructureIORecord]:
        """Align one declared structure invocation to a common semantic space."""


def clone_tree(value: Any) -> Any:
    """Clone tensor leaves while preserving the surrounding Python structure."""

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(clone_tree(item) for item in value)
    if isinstance(value, list):
        return [clone_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    return value
