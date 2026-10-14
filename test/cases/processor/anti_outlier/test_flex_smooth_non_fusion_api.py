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

Unit tests for flex_smooth_quant NonFusionSubgraph API (flex_smooth_impl_non_fusion_linear).
Covers non-fusion subgraph interface for flex_smooth_quant, mirroring iter_smooth non-fusion behavior.
"""
# pylint: disable=consider-using-from-import

import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from msmodelslim.ir.non_fusion_smooth_quant_ir import NonFusionSmoothQuantHookIR
from msmodelslim.processor.anti_outlier.common.subgraph_type import NonFusionSubgraph
from msmodelslim.processor.anti_outlier.common import (
    FlexSmoothQuantConfig,
    FlexSmoothQuantContext,
    SubgraphFusionFactory,
)
from msmodelslim.processor.anti_outlier.flex_smooth.api import (
    flex_smooth_quant,
    flex_smooth_impl_non_fusion_linear,
)


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


class TestFlexSmoothImplNonFusionLinear(unittest.TestCase):
    """Tests for flex_smooth_impl_non_fusion_linear (NonFusionSubgraph)."""

    def setUp(self):
        self.a_scale = torch.tensor([1.0, 2.0], dtype=torch.float32)
        self.act = torch.randn(4, 2)
        self.context = FlexSmoothQuantContext(
            version=1,
            a_smooth_scale=self.a_scale,
            tensors=[self.act],
        )

    def test_empty_linears_raises_value_error(self):
        """Empty linears list raises ValueError."""
        subgraph = NonFusionSubgraph(linears=[])
        config = FlexSmoothQuantConfig(version=1)
        with self.assertRaises(ValueError) as ctx:
            flex_smooth_impl_non_fusion_linear(subgraph, config, self.context)
        self.assertIn("at least one linear layer", str(ctx.exception))

    def test_empty_linears_via_entry_point_raises_value_error(self):
        """Dispatch via flex_smooth_quant() with empty linears also raises ValueError."""
        subgraph = NonFusionSubgraph(linears=[])
        config = FlexSmoothQuantConfig(version=1)
        with self.assertRaises(ValueError) as ctx:
            flex_smooth_quant(subgraph, config, self.context)
        self.assertIn("at least one linear layer", str(ctx.exception))

    def test_single_linear_applies_fusion_and_registers_hook(self):
        """Single linear: compute scales, apply fusion; caller registers one NonFusionSmoothQuantHookIR from returned scales."""
        linear = nn.Linear(2, 3)
        torch.nn.init.ones_(linear.weight)
        torch.nn.init.zeros_(linear.bias)
        subgraph = NonFusionSubgraph(linears=[linear])
        config = FlexSmoothQuantConfig(version=1, alpha=0.5, beta=0.5)

        with patch.object(SubgraphFusionFactory, "apply_fusion_to_subgraph") as mock_fusion:
            scales = flex_smooth_impl_non_fusion_linear(subgraph, config, self.context)
            mock_fusion.assert_called_once()
            call_args = mock_fusion.call_args
            self.assertIs(call_args[0][0], subgraph)
            self.assertIn("scales", call_args[1])
            self.assertIn("scales", call_args[1]["scales"])
            scales_tensor = call_args[1]["scales"]["scales"]
            self.assertTrue(torch.is_tensor(scales_tensor))
            self.assertEqual(scales_tensor.shape, (2,))

        self.assertIsNotNone(scales, "flex_smooth should return scales for NonFusionSubgraph")
        for linear_module in subgraph.linears:
            hook_ir = NonFusionSmoothQuantHookIR(scales)
            hook_handle = linear_module.register_forward_pre_hook(hook_ir)
            hook_ir.set_hook_handle(hook_handle)

        ir_hooks = [h for h in _get_hook_objs(linear) if isinstance(h, NonFusionSmoothQuantHookIR)]
        self.assertEqual(len(ir_hooks), 1)
        self.assertIsNotNone(ir_hooks[0].hook_handle)

    def test_single_linear_via_entry_point(self):
        """Single linear via flex_smooth_quant() dispatches to non-fusion impl; caller registers hooks from returned scales."""
        linear = nn.Linear(2, 3)
        torch.nn.init.ones_(linear.weight)
        subgraph = NonFusionSubgraph(linears=[linear])
        config = FlexSmoothQuantConfig(version=1, alpha=0.5, beta=0.5)

        with patch.object(SubgraphFusionFactory, "apply_fusion_to_subgraph") as mock_fusion:
            scales = flex_smooth_quant(subgraph, config, self.context)
            mock_fusion.assert_called_once()

        self.assertIsNotNone(scales)
        for linear_module in subgraph.linears:
            hook_ir = NonFusionSmoothQuantHookIR(scales)
            hook_handle = linear_module.register_forward_pre_hook(hook_ir)
            hook_ir.set_hook_handle(hook_handle)

        ir_hooks = [h for h in _get_hook_objs(linear) if isinstance(h, NonFusionSmoothQuantHookIR)]
        self.assertEqual(len(ir_hooks), 1)

    def test_multiple_linears_applies_fusion_and_registers_hooks(self):
        """Multiple linears: one fusion call; caller registers one hook per linear from returned scales."""
        linear1 = nn.Linear(2, 3)
        linear2 = nn.Linear(2, 4)
        torch.nn.init.ones_(linear1.weight)
        torch.nn.init.ones_(linear2.weight)
        subgraph = NonFusionSubgraph(linears=[linear1, linear2])
        config = FlexSmoothQuantConfig(version=1, alpha=0.5, beta=0.5)

        with patch.object(SubgraphFusionFactory, "apply_fusion_to_subgraph") as mock_fusion:
            scales = flex_smooth_impl_non_fusion_linear(subgraph, config, self.context)
            mock_fusion.assert_called_once()
            call_args = mock_fusion.call_args
            self.assertIs(call_args[0][0], subgraph)
            scales_tensor = call_args[1]["scales"]["scales"]
            self.assertEqual(scales_tensor.shape, (2,))

        self.assertIsNotNone(scales)
        for linear_module in subgraph.linears:
            hook_ir = NonFusionSmoothQuantHookIR(scales)
            hook_handle = linear_module.register_forward_pre_hook(hook_ir)
            hook_ir.set_hook_handle(hook_handle)

        for linear in subgraph.linears:
            ir_hooks = [h for h in _get_hook_objs(linear) if isinstance(h, NonFusionSmoothQuantHookIR)]
            self.assertEqual(len(ir_hooks), 1)

    def test_single_linear_forward_after_smooth(self):
        """After flex_smooth and registering hooks from returned scales, single linear forward runs."""
        linear = nn.Linear(2, 3)
        torch.nn.init.ones_(linear.weight)
        torch.nn.init.zeros_(linear.bias)
        subgraph = NonFusionSubgraph(linears=[linear])
        config = FlexSmoothQuantConfig(version=1, alpha=0.5, beta=0.5)
        scales = flex_smooth_impl_non_fusion_linear(subgraph, config, self.context)
        self.assertIsNotNone(scales)
        for linear_module in subgraph.linears:
            hook_ir = NonFusionSmoothQuantHookIR(scales)
            hook_handle = linear_module.register_forward_pre_hook(hook_ir)
            hook_ir.set_hook_handle(hook_handle)

        x = torch.randn(1, 2)
        out = linear(x)
        self.assertEqual(out.shape, (1, 3))

    def test_with_alpha_beta_search(self):
        """When alpha/beta are None, search is used; fusion applied; caller registers hooks from returned scales."""
        linear = nn.Linear(2, 3)
        torch.nn.init.ones_(linear.weight)
        subgraph = NonFusionSubgraph(linears=[linear])
        config = FlexSmoothQuantConfig(version=1, alpha=None, beta=None)

        with patch.object(SubgraphFusionFactory, "apply_fusion_to_subgraph") as mock_fusion:
            scales = flex_smooth_impl_non_fusion_linear(subgraph, config, self.context)
            mock_fusion.assert_called_once()

        self.assertIsNotNone(scales)
        for linear_module in subgraph.linears:
            hook_ir = NonFusionSmoothQuantHookIR(scales)
            hook_handle = linear_module.register_forward_pre_hook(hook_ir)
            hook_ir.set_hook_handle(hook_handle)

        ir_hooks = [h for h in _get_hook_objs(linear) if isinstance(h, NonFusionSmoothQuantHookIR)]
        self.assertEqual(len(ir_hooks), 1)

    def test_fusion_receives_scales_key(self):
        """apply_fusion_to_subgraph is called with scales={'scales': tensor}."""
        linear = nn.Linear(2, 3)
        torch.nn.init.ones_(linear.weight)
        subgraph = NonFusionSubgraph(linears=[linear])
        config = FlexSmoothQuantConfig(version=1, alpha=0.5, beta=0.5)

        with patch.object(SubgraphFusionFactory, "apply_fusion_to_subgraph") as mock_fusion:
            flex_smooth_impl_non_fusion_linear(subgraph, config, self.context)
            kwargs = mock_fusion.call_args[1]
            self.assertIn("scales", kwargs)
            self.assertIn("scales", kwargs["scales"])
            s = kwargs["scales"]["scales"]
            self.assertEqual(s.dim(), 1)
            self.assertEqual(s.size(0), 2)

    def test_weight_modified_by_fusion(self):
        """Without mocking fusion, weights are scaled in-place by NonFusionSubgraphFusion."""
        linear = nn.Linear(2, 3)
        torch.nn.init.ones_(linear.weight)
        w_before = linear.weight.data.clone()
        subgraph = NonFusionSubgraph(linears=[linear])
        config = FlexSmoothQuantConfig(version=1, alpha=0.5, beta=0.5)

        flex_smooth_impl_non_fusion_linear(subgraph, config, self.context)

        # Fusion applies scales to weight; so weight should have changed
        self.assertFalse(torch.equal(linear.weight.data, w_before))


if __name__ == "__main__":
    unittest.main()
