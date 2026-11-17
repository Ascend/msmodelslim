#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.-------------------------------------------------------------------


from collections import defaultdict
from typing import Optional, Tuple

import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest


def _is_moe_module(name: str) -> bool:
    """Return True if the module path indicates MoE (Mixture of Experts) structure."""
    moe_patterns = (".experts.", ".experts", "experts.")
    return any(pattern in name for pattern in moe_patterns)


def is_layernorm(module: nn.Module) -> bool:
    """Returns whether the module is a layernorm layer."""
    module_name = type(module).__name__
    return any(norm in module_name for norm in ["LayerNorm", "RMSNorm"])


def collect_shared_input_modules(request: BatchProcessRequest) -> Tuple[dict, Optional[dict]]:
    """Collect modules that share the same input using forward hooks.

    This is a common helper for both LLM and diffusion model fusion.

    Args:
        model: The model to analyze.
        dummy_forward_fn: A callable that runs a dummy forward pass on the model.
            Should be a function that takes no arguments.

    Returns:
        A tuple of (input_to_linear, output_to_layernorm).
        input_to_linear: Dict mapping input tensor to list of modules sharing that input.
        output_to_layernorm: Dict mapping layernorm output to the layernorm module (or None).
    """
    input_to_linear: dict = defaultdict(list)
    output_to_layernorm: dict = defaultdict(nn.Module)

    def _input_hook(module, inputs, output):
        """Update dictionary with list of all modules that share the same input."""
        if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
            # Note: DBRX MoE case is not handled yet
            input_to_linear[inputs[0]].append(module)

    def _output_hook(module, inputs, output):  # pylint: disable=unused-argument
        """Update dictionary with mapping of layernorms and their outputs."""
        if output_to_layernorm is not None and isinstance(output, torch.Tensor):
            output_to_layernorm[output] = module

    handles = []
    model = request.module
    args, kwargs = request.datas[0]

    # Register hooks on all quantized linear modules (and optionally layernorms)
    for name, module in model.named_modules():
        module.name = name
        if is_layernorm(module):
            handle = module.register_forward_hook(_output_hook)
            handles.append(handle)
        elif isinstance(module, nn.Linear):
            handle = module.register_forward_hook(_input_hook)
            handles.append(handle)

    if not handles:
        return input_to_linear, output_to_layernorm

    # Run dummy forward pass to collect modules sharing same input
    try:
        with torch.no_grad():
            if args or kwargs:
                model(*args, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
    return input_to_linear, output_to_layernorm
