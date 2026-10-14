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

Unit tests for iter_smooth NonFusionSubgraph API (iter_smooth_impl_non_fusion_linear).
Covers code added in commit f2075e1 (non-fusion iter_smooth).
"""
# pylint: disable=consider-using-from-import

import unittest
from unittest.mock import patch

from unittest.mock import MagicMock  # noqa: F401
import torch
import torch.nn as nn

from msmodelslim.processor.anti_outlier.common.subgraph_type import NonFusionSubgraph
from msmodelslim.processor.anti_outlier.iter_smooth.api import iter_smooth
from msmodelslim.processor.anti_outlier.common import (
    IterSmoothConfig,
    IterSmoothContext,
    SubgraphFusionFactory,
)
from msmodelslim.ir.non_fusion_smooth_quant_ir import NonFusionSmoothQuantHookIR


def _get_hook_objs(module):
    """从 module 的 _forward_pre_hooks 中取出 hook 实例（兼容不同 PyTorch 的存储格式）。"""
    if not hasattr(module, "_forward_pre_hooks"):
        return []
    out = []
    for h in module._forward_pre_hooks.values():
        if isinstance(h, tuple):
            for x in h:
                if callable(x) and not isinstance(x, (bool, int, float)):
                    out.append(x)
                    break
        elif hasattr(h, "hook") and callable(getattr(h, "hook")):
            out.append(h.hook)
        elif callable(h):
            out.append(h)
    return out


class TestIterSmoothImplNonFusionLinear(unittest.TestCase):
    """Tests for iter_smooth_impl_non_fusion_linear (NonFusionSubgraph)."""

    def setUp(self):
        self.config = IterSmoothConfig(alpha=0.5, scale_min=1e-5)
        self.a_scale = torch.tensor([[1.0, 2.0]])
        self.shift = torch.tensor([0.0])
        self.context = IterSmoothContext(
            version=1,
            a_smooth_scale=self.a_scale,
            shift=self.shift,
        )

    def test_empty_linears_raises_value_error(self):
        """Empty linears list raises ValueError."""
        subgraph = NonFusionSubgraph(linears=[])
        with self.assertRaises(ValueError) as ctx:
            iter_smooth(subgraph, self.config, self.context)
        self.assertIn("at least one linear layer", str(ctx.exception))

    def test_single_linear_applies_fusion_and_registers_hook(self):
        """Single linear: compute scales, apply fusion; caller registers pre_hook from returned scales."""
        linear = nn.Linear(2, 3)
        torch.nn.init.ones_(linear.weight)
        torch.nn.init.zeros_(linear.bias)
        subgraph = NonFusionSubgraph(linears=[linear])

        with patch.object(SubgraphFusionFactory, "apply_fusion_to_subgraph") as mock_fusion:
            scales = iter_smooth(subgraph, self.config, self.context)
            mock_fusion.assert_called_once()
            call_args = mock_fusion.call_args
            self.assertIs(call_args[0][0], subgraph)
            self.assertIn("scales", call_args[1])
            self.assertIn("scales", call_args[1]["scales"])

        self.assertIsNotNone(scales, "iter_smooth should return scales for NonFusionSubgraph")
        # Register hooks as processor does (API returns scales; processor registers hooks)
        for linear_module in subgraph.linears:
            hook_ir = NonFusionSmoothQuantHookIR(scales)
            hook_handle = linear_module.register_forward_pre_hook(hook_ir)
            hook_ir.set_hook_handle(hook_handle)

        # Check pre_hooks registered (one NonFusionSmoothQuantHookIR per linear)
        hook_objs = _get_hook_objs(linear)
        ir_hooks = [h for h in hook_objs if isinstance(h, NonFusionSmoothQuantHookIR)]
        self.assertEqual(len(ir_hooks), 1)
        self.assertIsNotNone(ir_hooks[0].hook_handle)

    def test_multiple_linears_applies_fusion_and_registers_hooks(self):
        """Multiple linears: w_scale from concat of per-linear stats, fusion; caller registers hooks from returned scales."""
        linear1 = nn.Linear(2, 3)
        linear2 = nn.Linear(2, 3)
        torch.nn.init.ones_(linear1.weight)
        torch.nn.init.ones_(linear2.weight)
        subgraph = NonFusionSubgraph(linears=[linear1, linear2])

        with patch.object(SubgraphFusionFactory, "apply_fusion_to_subgraph") as mock_fusion:
            scales = iter_smooth(subgraph, self.config, self.context)
            mock_fusion.assert_called_once()

        self.assertIsNotNone(scales)
        for linear_module in subgraph.linears:
            hook_ir = NonFusionSmoothQuantHookIR(scales)
            hook_handle = linear_module.register_forward_pre_hook(hook_ir)
            hook_ir.set_hook_handle(hook_handle)

        for linear in subgraph.linears:
            ir_hooks = [h for h in _get_hook_objs(linear) if isinstance(h, NonFusionSmoothQuantHookIR)]
            self.assertEqual(len(ir_hooks), 1)

    def test_single_linear_forward_after_smooth(self):
        """After iter_smooth and registering hooks from returned scales, single linear forward still runs."""
        linear = nn.Linear(2, 3)
        torch.nn.init.ones_(linear.weight)
        torch.nn.init.zeros_(linear.bias)
        subgraph = NonFusionSubgraph(linears=[linear])
        scales = iter_smooth(subgraph, self.config, self.context)
        self.assertIsNotNone(scales)
        for linear_module in subgraph.linears:
            hook_ir = NonFusionSmoothQuantHookIR(scales)
            hook_handle = linear_module.register_forward_pre_hook(hook_ir)
            hook_ir.set_hook_handle(hook_handle)
        x = torch.randn(1, 2)
        out = linear(x)
        self.assertEqual(out.shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
