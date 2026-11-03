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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.lossless.protocol import (
    LosslessCheckContext,
    LosslessCheckTarget,
    StructureIORecord,
)
from .lossless_alignment import align_baked_mean_output
from .quarot_interface import RotSide, RotateCommand


@dataclass(frozen=True)
class QuaRotLosslessCheckContext(LosslessCheckContext):
    """Request-local QuaRot data; it deliberately contains no behavior or model reference."""

    theoretical_commands: Tuple[RotateCommand, ...]
    theoretical_bake_names: Tuple[str, ...]
    norm_affines: Dict[str, List[Tuple[torch.Tensor, Optional[torch.Tensor]]]]


def create_quarot_lossless_check_context(
    processor,
    request: BatchProcessRequest,
    reference_datas: Any,
) -> QuaRotLosslessCheckContext:
    scope_modules = _get_scope_modules(request)
    scope_names = tuple(scope_modules)
    targets = tuple(
        LosslessCheckTarget(
            name=name,
            path=name,
            decoder=request.name,
            structure_type=module.__class__.__name__,
        )
        for name, module in scope_modules.items()
        if isinstance(module, (nn.Linear, nn.Embedding))
    )
    theoretical_pre_run_commands = _select_commands(processor._lossless_pre_run_commands, scope_names)
    theoretical_request_commands = _select_commands(processor._lossless_rotate_commands, scope_names)
    theoretical_commands = tuple(theoretical_pre_run_commands + theoretical_request_commands)
    theoretical_pre_run_fused_map = _select_fused_map(processor._lossless_pre_run_fused_map, scope_names)
    theoretical_request_fused_map = _select_fused_map(processor._lossless_fused_map, scope_names)
    theoretical_fused_map = _merge_fused_maps(
        theoretical_pre_run_fused_map,
        theoretical_request_fused_map,
    )
    theoretical_pre_run_bake_names = _select_names(processor._lossless_pre_run_bake_names, scope_names)
    theoretical_request_bake_names = _select_names(processor._lossless_bake_names, scope_names)
    theoretical_bake_names = tuple(theoretical_pre_run_bake_names + theoretical_request_bake_names)
    prefix = f"{request.name}." if request.name else ""
    execution_fused_map = _select_fused_map_by_prefix(processor.fused_map, prefix)
    execution_bake_names = [name for name in processor.bake_names if _request_target_matches(name, prefix)]
    execution_commands = [
        command for command in processor.rotate_commands if _request_target_matches(command.target, prefix)
    ]
    covered_targets = _operation_targets(
        theoretical_pre_run_fused_map,
        theoretical_pre_run_bake_names,
        theoretical_pre_run_commands,
    )
    covered_targets.update(
        _operation_targets(
            execution_fused_map,
            execution_bake_names,
            execution_commands,
        )
    )
    unresolved_targets = {target for target in covered_targets if not _target_exists(processor.model, target)}
    covered_targets.difference_update(unresolved_targets)
    if processor._lossless_global_rotation is not None:
        reference_datas = _map_hidden_state(
            reference_datas,
            lambda tensor: _inverse_rotation_if_compatible(
                tensor,
                processor._lossless_global_rotation,
            ),
        )
    return QuaRotLosslessCheckContext(
        targets=targets,
        reference_datas=reference_datas,
        covered_targets=tuple(sorted(covered_targets)),
        unresolved_targets=tuple(sorted(unresolved_targets)),
        theoretical_commands=theoretical_commands,
        theoretical_bake_names=theoretical_bake_names,
        norm_affines=_capture_norm_affines(processor.model, theoretical_fused_map),
    )


def align_quarot_lossless_check_io(
    context: QuaRotLosslessCheckContext,
    target: LosslessCheckTarget,
    reference: StructureIORecord,
    processed: StructureIORecord,
) -> Tuple[StructureIORecord, StructureIORecord]:
    commands = [command for command in context.theoretical_commands if command.target == target.name]
    right_commands = [command for command in commands if command.side == RotSide.RIGHT]
    left_commands = [command for command in commands if command.side == RotSide.LEFT]

    processed_args = processed.args
    processed_kwargs = processed.kwargs
    processed_output = processed.output
    if target.structure_type == "Embedding":
        for command in reversed(right_commands):
            processed_output = _map_last_dim(
                processed_output,
                lambda tensor, rot=command.rot: _inverse_rotation(tensor, rot),
            )
    else:
        for command in reversed(right_commands):
            processed_args = _map_last_dim(
                processed_args,
                lambda tensor, rot=command.rot: _inverse_rotation(tensor, rot),
            )
            processed_kwargs = _map_last_dim(
                processed_kwargs,
                lambda tensor, rot=command.rot: _inverse_rotation(tensor, rot),
            )

    for gamma, beta in context.norm_affines.get(target.name, []):
        processed_args = _map_last_dim(
            processed_args,
            lambda tensor, weight=gamma, bias=beta: _restore_norm_affine(tensor, weight, bias),
            expected_dim=gamma.numel(),
        )
        processed_kwargs = _map_last_dim(
            processed_kwargs,
            lambda tensor, weight=gamma, bias=beta: _restore_norm_affine(tensor, weight, bias),
            expected_dim=gamma.numel(),
        )

    for command in reversed(left_commands):
        processed_output = _map_last_dim(
            processed_output,
            lambda tensor, rot=command.rot: _inverse_rotation(tensor, rot),
        )

    reference_output = reference.output
    if target.name in context.theoretical_bake_names:
        reference_output, processed_output = _align_baked_tree(reference_output, processed_output)

    return (
        replace(reference, output=reference_output),
        replace(
            processed,
            args=processed_args,
            kwargs=processed_kwargs,
            output=processed_output,
        ),
    )


def _get_scope_modules(request: BatchProcessRequest) -> Dict[str, nn.Module]:
    modules = {}
    for local_name, module in request.module.named_modules():
        if request.name and local_name:
            name = f"{request.name}.{local_name}"
        else:
            name = request.name or local_name
        modules[name] = module
    return modules


def _select_commands(commands: Iterable[RotateCommand], scope_names: Sequence[str]) -> List[RotateCommand]:
    scope = set(scope_names)
    return [command for command in commands if command.target in scope]


def _select_fused_map(fused_map: Mapping[Any, Any], scope_names: Sequence[str]) -> Dict[Any, Any]:
    scope = set(scope_names)
    selected = {}
    for source, targets in fused_map.items():
        target_names = targets if isinstance(targets, (list, tuple)) else [targets]
        if any(target in scope for target in target_names):
            selected[source] = targets
    return selected


def _select_fused_map_by_prefix(fused_map: Mapping[Any, Any], prefix: str) -> Dict[Any, Any]:
    selected = {}
    for source, targets in fused_map.items():
        target_names = targets if isinstance(targets, (list, tuple)) else [targets]
        if any(_request_target_matches(target, prefix) for target in target_names):
            selected[source] = targets
    return selected


def _request_target_matches(target: str, prefix: str) -> bool:
    if not prefix:
        return True
    return target == prefix.rstrip(".") or target.startswith(prefix)


def _target_exists(model: nn.Module, target: str) -> bool:
    try:
        model.get_submodule(target)
        return True
    except AttributeError:
        if "." not in target:
            return hasattr(model, target)
        parent_name, attribute_name = target.rsplit(".", 1)
        try:
            parent = model.get_submodule(parent_name)
        except AttributeError:
            return False
        return hasattr(parent, attribute_name)


def _select_names(names: Iterable[str], scope_names: Sequence[str]) -> List[str]:
    scope = set(scope_names)
    return [name for name in names if name in scope]


def _merge_fused_maps(*mappings: Mapping[Any, Any]) -> Dict[Any, Any]:
    merged = {}
    for mapping in mappings:
        merged.update(mapping)
    return merged


def _operation_targets(
    fused_map: Mapping[Any, Any],
    bake_names: Iterable[str],
    commands: Iterable[RotateCommand],
):
    targets = set(bake_names)
    targets.update(command.target for command in commands)
    for value in fused_map.values():
        if isinstance(value, (list, tuple)):
            targets.update(value)
        else:
            targets.add(value)
    return targets


def _capture_norm_affines(model: nn.Module, fused_map: Mapping[Any, Any]):
    affines: Dict[str, List[Tuple[torch.Tensor, Optional[torch.Tensor]]]] = {}
    for source, targets in fused_map.items():
        source_names = list(source) if isinstance(source, (list, tuple)) else [source]
        norms = [model.get_submodule(name) for name in source_names]
        gamma = torch.cat([norm.weight.detach().clone() for norm in norms], dim=0)
        biases = [getattr(norm, "bias", None) for norm in norms]
        beta = None
        if biases and all(bias is not None for bias in biases):
            beta = torch.cat([bias.detach().clone() for bias in biases], dim=0)
        target_names = list(targets) if isinstance(targets, (list, tuple)) else [targets]
        for target in target_names:
            affines.setdefault(target, []).append((gamma, beta))
    return affines


def _restore_norm_affine(
    tensor: torch.Tensor,
    gamma: torch.Tensor,
    beta: Optional[torch.Tensor],
):
    gamma = gamma.to(device=tensor.device, dtype=tensor.dtype)
    restored = tensor * gamma
    if beta is not None:
        restored = restored + beta.to(device=tensor.device, dtype=tensor.dtype)
    return restored


def _inverse_rotation(tensor: torch.Tensor, rotation: Any) -> torch.Tensor:
    matrix = _rotation_tensor(rotation, tensor)
    processed_dim = matrix.shape[1]
    if tensor.shape[-1] != processed_dim:
        if matrix.shape[0] != matrix.shape[1] or tensor.shape[-1] % processed_dim != 0:
            raise ValueError(f"hidden dim {tensor.shape[-1]} does not match processed rotation dim {processed_dim}")
        matrix = torch.block_diag(*([matrix] * (tensor.shape[-1] // processed_dim)))
    return tensor.matmul(matrix.transpose(-1, -2))


def _rotation_tensor(rotation: Any, tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(rotation, (list, tuple)):
        matrices = [_as_rotation_tensor(item, tensor) for item in rotation]
        return torch.block_diag(*matrices)
    return _as_rotation_tensor(rotation, tensor)


def _as_rotation_tensor(rotation: Any, tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(rotation, torch.Tensor):
        rotation = torch.as_tensor(rotation)
    return rotation.to(device=tensor.device, dtype=tensor.dtype)


def _map_hidden_state(value: Any, transform):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and value.ndim > 0:
            return transform(value)
        return value
    if isinstance(value, list):
        return [_map_hidden_state(item, transform) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_hidden_state(item, transform) for item in value)
    if isinstance(value, dict):
        return {key: _map_hidden_state(item, transform) for key, item in value.items()}
    return value


def _inverse_rotation_if_compatible(tensor: torch.Tensor, rotation: Any) -> torch.Tensor:
    matrix = _rotation_tensor(rotation, tensor)
    processed_dim = matrix.shape[1]
    compatible = tensor.shape[-1] == processed_dim or (
        matrix.shape[0] == matrix.shape[1] and tensor.shape[-1] % processed_dim == 0
    )
    if not compatible:
        return tensor
    return _inverse_rotation(tensor, rotation)


def _map_last_dim(
    value: Any,
    transform,
    expected_dim: Optional[int] = None,
):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and (expected_dim is None or value.shape[-1] == expected_dim):
            return transform(value)
        return value
    if isinstance(value, tuple):
        return tuple(_map_last_dim(item, transform, expected_dim) for item in value)
    if isinstance(value, list):
        return [_map_last_dim(item, transform, expected_dim) for item in value]
    if isinstance(value, dict):
        return {key: _map_last_dim(item, transform, expected_dim) for key, item in value.items()}
    return value


def _align_baked_tree(reference: Any, processed: Any):
    if isinstance(reference, torch.Tensor) and isinstance(processed, torch.Tensor):
        return align_baked_mean_output(reference, processed)
    if isinstance(reference, tuple) and isinstance(processed, tuple):
        if len(reference) != len(processed):
            raise ValueError("Baked output tuple structure does not match")
        pairs = [_align_baked_tree(left, right) for left, right in zip(reference, processed)]
        return tuple(pair[0] for pair in pairs), tuple(pair[1] for pair in pairs)
    if isinstance(reference, list) and isinstance(processed, list):
        if len(reference) != len(processed):
            raise ValueError("Baked output list structure does not match")
        pairs = [_align_baked_tree(left, right) for left, right in zip(reference, processed)]
        return [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    if isinstance(reference, dict) and isinstance(processed, dict):
        if reference.keys() != processed.keys():
            raise ValueError("Baked output mapping structure does not match")
        pairs = {key: _align_baked_tree(reference[key], processed[key]) for key in reference}
        return {key: pair[0] for key, pair in pairs.items()}, {key: pair[1] for key, pair in pairs.items()}
    if isinstance(reference, (torch.Tensor, tuple, list, dict)) or isinstance(
        processed,
        (torch.Tensor, tuple, list, dict),
    ):
        raise ValueError("Baked output structure types do not match")
    return reference, processed
