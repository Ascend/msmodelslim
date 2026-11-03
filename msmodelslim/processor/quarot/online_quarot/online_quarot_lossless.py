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

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.ir.quarot import OnlineRotationInfo, RotationType
from msmodelslim.processor.lossless.protocol import (
    LosslessCheckContext,
    LosslessCheckTarget,
    StructureIORecord,
)


@dataclass(frozen=True)
class OnlineQuaRotRotationPlan:
    """A frozen, request-local view of one theoretical online rotation."""

    name: str
    rotation_type: RotationType
    rotation_matrix: torch.Tensor
    rotation_side: Optional[str]
    already_applied: bool


@dataclass(frozen=True)
class OnlineQuaRotLosslessCheckContext(LosslessCheckContext):
    """Reference capture data and immutable rotation expectations for one request."""

    rotation_plans: Tuple[OnlineQuaRotRotationPlan, ...]


def create_lossless_check_context(
    processor,
    request: BatchProcessRequest,
    reference_datas: Any,
) -> OnlineQuaRotLosslessCheckContext:
    rotation_infos = processor._lossless_rotation_infos
    if rotation_infos is None:
        raise RuntimeError("prepare_lossless_check must run before creating an online QuaRot context")

    scope_modules = _get_scope_modules(request)
    target_modules = {}
    covered_targets = []
    unresolved_targets = []
    rotation_plans = []

    for name, rotation_info in rotation_infos.items():
        if not _is_in_request_scope(name, request.name):
            continue
        rotation_matrix = rotation_info.rotation_matrix.detach().clone()
        if name not in scope_modules:
            unresolved_targets.append(name)
            rotation_plans.append(
                OnlineQuaRotRotationPlan(
                    name=name,
                    rotation_type=rotation_info.rotation_type,
                    rotation_matrix=rotation_matrix,
                    rotation_side=rotation_info.rotation_side,
                    already_applied=False,
                )
            )
            continue

        module = scope_modules[name]
        target_modules[name] = module
        covered_targets.append(name)
        rotation_plans.append(
            OnlineQuaRotRotationPlan(
                name=name,
                rotation_type=rotation_info.rotation_type,
                rotation_matrix=rotation_matrix,
                rotation_side=rotation_info.rotation_side,
                already_applied=rotation_is_applied(
                    module,
                    name,
                    rotation_info,
                ),
            )
        )

    # Check the real linear operations in the requested module scope as well as
    # explicit rotation sites. Keep the rotation target when both select the
    # same module so its frozen theoretical plan is still applied by align().
    for name, module in scope_modules.items():
        if isinstance(module, nn.Linear):
            target_modules.setdefault(name, module)

    targets = [
        LosslessCheckTarget(
            name=name,
            path=name,
            decoder=request.name,
            structure_type=module.__class__.__name__,
        )
        for name, module in sorted(target_modules.items())
    ]

    return OnlineQuaRotLosslessCheckContext(
        targets=tuple(targets),
        reference_datas=reference_datas,
        covered_targets=tuple(sorted(covered_targets)),
        unresolved_targets=tuple(sorted(unresolved_targets)),
        rotation_plans=tuple(rotation_plans),
    )


def align_lossless_check_io(
    context: OnlineQuaRotLosslessCheckContext,
    target: LosslessCheckTarget,
    reference: StructureIORecord,
    processed: StructureIORecord,
) -> Tuple[StructureIORecord, StructureIORecord]:
    rotation_plan = next(
        (item for item in context.rotation_plans if item.name == target.name),
        None,
    )
    if rotation_plan is None:
        return reference, processed
    if rotation_plan.already_applied:
        return reference, processed

    matrix = rotation_plan.rotation_matrix
    rotation_type = rotation_plan.rotation_type
    processed_args = processed.args
    processed_output = processed.output

    if rotation_type == RotationType.INPUT:
        processed_args = _transform_first_input(processed_args, matrix)
    elif rotation_type == RotationType.OUTPUT:
        processed_output = _transform_output(processed_output, matrix)
    elif rotation_type == RotationType.REPLACE:
        processed_output = _transform_output(processed_output, matrix)
    elif rotation_type == RotationType.OFFLINE:
        if (rotation_plan.rotation_side or "right") == "right":
            # Right-rotated weights consume activations in the rotated input basis.
            processed_args = _transform_first_input(processed_args, matrix, transpose=True)
        else:
            # A left rotation is expressed in the output basis.
            processed_output = _transform_output(processed_output, matrix, transpose=True)
    else:
        raise ValueError(f"Unsupported online QuaRot rotation type: {rotation_type}")

    return (
        reference,
        replace(
            processed,
            args=processed_args,
            output=processed_output,
        ),
    )


def rotation_is_applied(module: nn.Module, name: str, expected_info: OnlineRotationInfo) -> bool:
    """Return whether this module already carries the requested online rotation."""

    if _has_rotation_identity(module, name, expected_info):
        return True
    for hook_group in ("_forward_pre_hooks", "_forward_hooks"):
        for hook in getattr(module, hook_group, {}).values():
            if _has_rotation_identity(hook, name, expected_info):
                return True
    return False


def _has_rotation_identity(value: Any, name: str, expected_info: OnlineRotationInfo) -> bool:
    existing_info = getattr(value, "rotation_info", None)
    if (
        getattr(value, "layer_name", None) != name
        or not isinstance(existing_info, OnlineRotationInfo)
        or existing_info.rotation_type != expected_info.rotation_type
    ):
        return False

    existing_matrix = existing_info.rotation_matrix
    expected_matrix = expected_info.rotation_matrix
    if existing_matrix.dtype != expected_matrix.dtype or not torch.equal(existing_matrix, expected_matrix):
        raise ValueError(f"Existing online rotation for target '{name}' uses a different rotation matrix")
    if (
        expected_info.rotation_type == RotationType.OFFLINE
        and existing_info.rotation_side != expected_info.rotation_side
    ):
        raise ValueError(f"Existing offline rotation for target '{name}' uses a different rotation side")
    return True


def _get_scope_modules(request: BatchProcessRequest) -> Dict[str, nn.Module]:
    modules = {}
    for local_name, module in request.module.named_modules():
        if request.name and local_name:
            name = f"{request.name}.{local_name}"
        else:
            name = request.name or local_name
        modules[name] = module
    return modules


def _is_in_request_scope(name: str, request_name: str) -> bool:
    return not request_name or name == request_name or name.startswith(f"{request_name}.")


def _transform_first_input(
    args: Tuple[Any, ...],
    matrix: torch.Tensor,
    transpose: bool = False,
) -> Tuple[Any, ...]:
    if not args:
        raise ValueError("Online QuaRot input alignment requires a positional tensor input")
    return (_transform_tensor(args[0], matrix, transpose=transpose),) + args[1:]


def _transform_output(
    output: Any,
    matrix: torch.Tensor,
    transpose: bool = False,
) -> Any:
    if isinstance(output, tuple):
        if not output:
            raise ValueError("Online QuaRot output alignment received an empty tuple")
        return (_transform_tensor(output[0], matrix, transpose=transpose),) + output[1:]
    return _transform_tensor(output, matrix, transpose=transpose)


def _transform_tensor(
    tensor: Any,
    matrix: torch.Tensor,
    transpose: bool = False,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0:
        raise TypeError("Online QuaRot alignment requires a non-scalar tensor")
    if tensor.shape[-1] != matrix.shape[0]:
        raise ValueError(
            f"Online QuaRot alignment dimension {tensor.shape[-1]} does not match rotation dimension {matrix.shape[0]}"
        )
    aligned_matrix = matrix.to(device=tensor.device, dtype=tensor.dtype)
    if transpose:
        aligned_matrix = aligned_matrix.transpose(-1, -2)
    return torch.matmul(tensor, aligned_matrix)
