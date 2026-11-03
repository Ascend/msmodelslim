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

import random
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest

from .protocol import LosslessCheckTarget, StructureIORecord, clone_tree


class TargetResolutionError(LookupError):
    def __init__(self, target: LosslessCheckTarget):
        super().__init__(target.path)
        self.target = target


class StructureCapture:
    """Capture one invocation record per target without retaining live tensors."""

    def __init__(
        self,
        request: BatchProcessRequest,
        targets: Sequence[LosslessCheckTarget],
    ):
        self.request = request
        self.request_name = request.name
        self.targets = targets
        self.inputs: Dict[str, List[Tuple[Tuple[Any, ...], dict]]] = defaultdict(list)
        self.outputs: Dict[str, List[Any]] = defaultdict(list)
        self.invocation_order: List[Tuple[str, int]] = []
        self.handles = []

    def __enter__(self):
        for target in self.targets:
            try:
                relative_path = target.relative_path(self.request.name)
                module = self.request.module if not relative_path else self.request.module.get_submodule(relative_path)
            except (AttributeError, KeyError) as error:
                raise TargetResolutionError(target) from error
            self.handles.append(
                module.register_forward_pre_hook(
                    self._make_input_hook(target.name),
                    with_kwargs=True,
                )
            )
            self.handles.append(
                module.register_forward_hook(
                    self._make_output_hook(target.name),
                    with_kwargs=True,
                )
            )
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def records(self) -> Dict[str, List[StructureIORecord]]:
        records: Dict[str, List[StructureIORecord]] = {}
        for target in self.targets:
            inputs = self.inputs.get(target.name, [])
            outputs = self.outputs.get(target.name, [])
            count = min(len(inputs), len(outputs))
            records[target.name] = [
                StructureIORecord(
                    request_name=self.request_name,
                    structure_name=target.name,
                    invocation=index,
                    args=inputs[index][0],
                    kwargs=inputs[index][1],
                    output=outputs[index],
                )
                for index in range(count)
            ]
        return records

    def _make_input_hook(self, name):
        def hook(_module, args, kwargs):
            self.invocation_order.append((name, len(self.inputs[name])))
            self.inputs[name].append((clone_tree(args), clone_tree(kwargs)))

        return hook

    def _make_output_hook(self, name):
        def hook(_module, _args, _kwargs, output):
            self.outputs[name].append(clone_tree(output))

        return hook


class ForwardExecutionState:
    """Keep the extra reference forward deterministic and observationally neutral."""

    def __init__(self, request_module: nn.Module):
        self.modules = self._collect_modules(request_module)
        self.training_states = []
        self.python_rng_state = None
        self.cpu_rng_state = None
        self.accelerator_rng_states = {}
        self.keep_advanced_rng = False

    def __enter__(self):
        self.training_states = [(module, module.training) for module in self.modules]
        self.python_rng_state = random.getstate()
        self.cpu_rng_state = torch.get_rng_state().clone()
        self.accelerator_rng_states = self._snapshot_initialized_accelerators()
        for module in self.modules:
            module.training = False
        return self

    def __exit__(self, exc_type, _exc_value, _traceback):
        if exc_type is not None or not self.keep_advanced_rng:
            self.rewind_rng()
        for module, training in self.training_states:
            module.training = training
        return False

    def rewind_rng(self) -> None:
        random.setstate(self.python_rng_state)
        torch.set_rng_state(self.cpu_rng_state)
        for backend, states in self.accelerator_rng_states.values():
            backend.set_rng_state_all(states)

    def commit(self) -> None:
        self.keep_advanced_rng = True

    @staticmethod
    def _collect_modules(request_module: nn.Module) -> List[nn.Module]:
        modules = []
        seen = set()
        for module in request_module.modules():
            if id(module) not in seen:
                modules.append(module)
                seen.add(id(module))
        return modules

    @staticmethod
    def _snapshot_initialized_accelerators():
        states = {}
        for name in ("cuda", "npu"):
            backend = getattr(torch, name, None)
            is_initialized = getattr(backend, "is_initialized", None)
            get_rng_state_all = getattr(backend, "get_rng_state_all", None)
            set_rng_state_all = getattr(backend, "set_rng_state_all", None)
            if (
                backend is not None
                and callable(is_initialized)
                and is_initialized()
                and callable(get_rng_state_all)
                and callable(set_rng_state_all)
            ):
                states[name] = (
                    backend,
                    [state.clone() for state in get_rng_state_all()],
                )
        return states


def flatten_tree(value: Any, path: str = "root"):
    """Yield deterministic paths for nested tensor-bearing Python structures."""

    if isinstance(value, tuple):
        for index, item in enumerate(value):
            yield from flatten_tree(item, f"{path}[{index}]")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from flatten_tree(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            yield from flatten_tree(value[key], f"{path}.{key}")
    else:
        yield path, value
