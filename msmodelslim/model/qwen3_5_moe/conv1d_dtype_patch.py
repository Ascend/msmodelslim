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

import torch
from torch import nn


_ORIGINAL_CONV1D_FORWARD = None
_ORIGINAL_LAYERNORM_FORWARD = None


def _align_input_to_weight_dtype(module, input):  # pylint: disable=redefined-builtin
    weight = getattr(module, "weight", None)
    if isinstance(input, torch.Tensor) and isinstance(weight, torch.Tensor) and input.dtype != weight.dtype:
        return input.to(weight.dtype)
    return input


def _patched_conv1d_forward(self, input, *args, **kwargs):  # pylint: disable=redefined-builtin
    """Align Conv1d input dtype to weight after FakeQuant Linear upcasts activations to fp32."""
    orig = _ORIGINAL_CONV1D_FORWARD
    if orig is None:
        raise RuntimeError("Conv1d forward patch is not initialized")
    return orig(self, _align_input_to_weight_dtype(self, input), *args, **kwargs)


def _patched_layernorm_forward(self, input):  # pylint: disable=redefined-builtin
    """Align LayerNorm input dtype to weight after FakeQuant Linear upcasts activations to fp32."""
    orig = _ORIGINAL_LAYERNORM_FORWARD
    if orig is None:
        raise RuntimeError("LayerNorm forward patch is not initialized")
    return orig(self, _align_input_to_weight_dtype(self, input))


def ensure_conv1d_forward_patched() -> None:
    """Patch Conv1d / LayerNorm forward once so NPU does not mix fp32 input and bf16 weight.

    Idempotent. Must be called in each process: ``mp.spawn`` starts a fresh interpreter
    that does not inherit the parent process patch, and pickled adapters skip ``__init__``.
    """
    global _ORIGINAL_CONV1D_FORWARD, _ORIGINAL_LAYERNORM_FORWARD
    if _ORIGINAL_CONV1D_FORWARD is None:
        _ORIGINAL_CONV1D_FORWARD = nn.Conv1d.forward
        nn.Conv1d.forward = _patched_conv1d_forward
    if _ORIGINAL_LAYERNORM_FORWARD is None:
        _ORIGINAL_LAYERNORM_FORWARD = nn.LayerNorm.forward
        nn.LayerNorm.forward = _patched_layernorm_forward
