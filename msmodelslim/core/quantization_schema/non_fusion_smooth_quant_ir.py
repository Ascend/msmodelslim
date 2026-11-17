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
# pylint: disable=useless-import-alias

import copy  # noqa: F401
import torch
from torch import nn as nn

from .wrapper import WrapperIR, HookIR


class NonFusionSmoothQuantWrapper(WrapperIR):
    """
    WrapperIR for non-fusion SmoothQuant: scales input then forwards to the wrapped linear.

    Stores 1/scales as a parameter; forward does input * (1/scales) then passes to
    wrapped_module, equivalent to dividing by scales before the linear (numerically
    equivalent to the original linear when weight was pre-scaled by scales).
    """

    def __init__(self, scales: torch.Tensor, module: nn.Module):
        super().__init__(module)
        self.scales = torch.nn.Parameter(1 / scales, requires_grad=False)

    @staticmethod
    def is_atomic() -> bool:
        """Treat this IR as a single unit during save/export (do not recurse into wrapped_module)."""
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # self.scales holds 1/scales; (x * (1/scales)) then linear preserves numerical equivalence
        if isinstance(x, tuple):
            x = x[0]
        scale = self.scales.to(device=x.device, dtype=x.dtype)
        return self.wrapped_module(x * scale)


class NonFusionSmoothQuantHookIR(HookIR):
    """
    HookIR for non-fusion SmoothQuant: carries scale and builds the replacement module at save time.

    - Holds smooth scales (1/scales is applied as input scaling). Does not perform scaling in
      the hook; __call__ is a no-op and only exists to satisfy the forward_pre_hook interface.
    - When the saver calls wrapper_module(linear_module) (e.g. in _convert_hookir_to_wrapper),
      this hook returns a NonFusionSmoothQuantWrapper(scales, module) that replaces the
      original Linear in the model.

    Usage (e.g. in iter_smooth):
        hook_ir = NonFusionSmoothQuantHookIR(scales)
        hook_handle = linear_module.register_forward_pre_hook(hook_ir)
        hook_ir.set_hook_handle(hook_handle)
    """

    def __init__(self, scales: torch.Tensor):
        """
        Args:
            scales: SmoothQuant scale tensor (inverse is applied to input in the wrapper).
        """
        super().__init__()
        self.scales = scales

    def __call__(self, module: nn.Module, args: tuple) -> tuple:
        """
        Forward pre-hook callable. No-op: this HookIR only carries state for save-time
        conversion; scaling is done inside NonFusionSmoothQuantWrapper after replacement.
        """
        x = args
        if isinstance(x, tuple):
            x = x[0]

        inv_scale = (1.0 / self.scales).to(device=x.device, dtype=x.dtype)
        return x * inv_scale

    def wrapper_module(self, module: nn.Module) -> WrapperIR:
        """
        Build the replacement module for the given linear (called by saver / convert_hookir_to_wrapper).

        Args:
            module: The original nn.Linear to wrap (weight already scaled by SmoothQuant).

        Returns:
            NonFusionSmoothQuantWrapper wrapping the linear, used to replace the original module.
        """
        self.remove_hook()
        return NonFusionSmoothQuantWrapper(self.scales, module)
