#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd

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

import inspect
import unittest

import torch
from torch import nn

from msmodelslim.processor.anti_outlier.common.subgraph_fusion import (
    LinearLinearSubgraphFusion,
    NonFusionSubgraphFusion,
    NormLinearSubgraphFusion,
    OVSubgraphFusion,
    SubgraphFusionFactory,
    SubgraphFusionStrategy,
    UpDownSubgraphFusion,
)
from msmodelslim.processor.anti_outlier.common.subgraph_type import (
    LinearLinearSubgraph,
    NonFusionSubgraph,
    NormLinearSubgraph,
    OVSubgraph,
    Subgraph,
    UpDownSubgraph,
)


def _filled_linear(in_features, out_features, weight_value=1.0, bias_value=None):
    layer = nn.Linear(in_features, out_features, bias=bias_value is not None)
    with torch.no_grad():
        layer.weight.fill_(weight_value)
        if layer.bias is not None:
            layer.bias.fill_(bias_value)
    return layer


class TestSubgraphFusionStrategy(unittest.TestCase):
    def test_subgraph_fusion_strategy_is_abstract_for_custom_fusers(self):
        self.assertTrue(inspect.isabstract(SubgraphFusionStrategy))


class TestNonFusionSubgraphFusion(unittest.TestCase):
    def test_apply_fusion_when_scales_match_input_dim_then_scales_each_linear_weight(self):
        linear1 = _filled_linear(4, 3)
        linear2 = _filled_linear(4, 3)
        subgraph = NonFusionSubgraph(linears=[linear1, linear2])
        strategy = NonFusionSubgraphFusion()

        strategy.apply_fusion(subgraph, {"scales": torch.tensor([[2.0, 2.0, 2.0, 2.0]])}, None)

        torch.testing.assert_close(linear1.weight, torch.full_like(linear1.weight, 2.0))
        torch.testing.assert_close(linear2.weight, torch.full_like(linear2.weight, 2.0))

    def test_apply_fusion_when_shift_provided_then_does_not_apply_shift(self):
        linear = _filled_linear(2, 2)
        subgraph = NonFusionSubgraph(linears=[linear])
        strategy = NonFusionSubgraphFusion()

        strategy.apply_fusion(subgraph, {"scales": torch.ones(1, 2)}, {"linear_shift": torch.tensor([3.0, 4.0])})

        torch.testing.assert_close(linear.weight, torch.ones_like(linear.weight))
        self.assertIsNone(linear.bias)


class TestOVSubgraphFusion(unittest.TestCase):
    def test_apply_fusion_when_o_and_v_scales_differ_then_updates_each_projection_independently(self):
        o_proj = _filled_linear(2, 2, bias_value=0.0)
        v_proj = _filled_linear(2, 2, bias_value=1.0)
        subgraph = OVSubgraph(o_proj=o_proj, v_proj=v_proj, num_attention_heads=1, key_value_heads=1)
        strategy = OVSubgraphFusion()

        strategy.apply_fusion(
            subgraph,
            {"o_scales": torch.tensor([2.0, 4.0]), "v_scales": torch.tensor([2.0, 4.0])},
            {"o_shift": torch.tensor([1.0, 2.0]), "v_shift": torch.tensor([3.0, 4.0])},
        )

        torch.testing.assert_close(o_proj.weight, torch.tensor([[2.0, 4.0], [2.0, 4.0]]))
        torch.testing.assert_close(o_proj.bias, torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(v_proj.weight, torch.tensor([[0.5, 0.5], [0.25, 0.25]]))
        torch.testing.assert_close(v_proj.bias, torch.tensor([3.5, 4.25]))


class TestUpDownSubgraphFusion(unittest.TestCase):
    def test_apply_fusion_when_up_and_down_scales_are_inverse_then_updates_up_and_down_projections(self):
        up_proj = _filled_linear(2, 2, bias_value=1.0)
        down_proj = _filled_linear(2, 2, bias_value=0.0)
        gate_proj = _filled_linear(2, 2)
        subgraph = UpDownSubgraph(up_proj=up_proj, down_proj=down_proj, gate_proj=gate_proj)
        strategy = UpDownSubgraphFusion()

        strategy.apply_fusion(
            subgraph,
            {"scales": torch.tensor([2.0, 4.0])},
            {"down_shift": torch.tensor([1.0, 2.0]), "up_shift": torch.tensor([3.0, 4.0])},
        )

        torch.testing.assert_close(down_proj.weight, torch.tensor([[2.0, 4.0], [2.0, 4.0]]))
        torch.testing.assert_close(down_proj.bias, torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(up_proj.weight, torch.tensor([[0.5, 0.5], [0.25, 0.25]]))
        torch.testing.assert_close(up_proj.bias, torch.tensor([3.5, 4.25]))


class TestLinearLinearSubgraphFusion(unittest.TestCase):
    def test_apply_fusion_when_linear_scales_are_inverse_then_updates_linear1_and_linear2(self):
        linear1 = _filled_linear(2, 2, bias_value=1.0)
        linear2 = _filled_linear(2, 2, bias_value=0.0)
        subgraph = LinearLinearSubgraph(linear1=linear1, linear2=linear2)
        strategy = LinearLinearSubgraphFusion()

        strategy.apply_fusion(
            subgraph,
            {"scales": torch.tensor([2.0, 4.0])},
            {"linear1_shift": torch.tensor([3.0, 4.0]), "linear2_shift": torch.tensor([1.0, 2.0])},
        )

        torch.testing.assert_close(linear2.weight, torch.tensor([[2.0, 4.0], [2.0, 4.0]]))
        torch.testing.assert_close(linear2.bias, torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(linear1.weight, torch.tensor([[0.5, 0.5], [0.25, 0.25]]))
        torch.testing.assert_close(linear1.bias, torch.tensor([3.5, 4.25]))


class TestNormLinearSubgraphFusion(unittest.TestCase):
    def test_apply_fusion_when_multiple_linears_and_norm_shift_provided_then_updates_each_module(self):
        norm = nn.LayerNorm(2)
        with torch.no_grad():
            norm.weight.fill_(1.0)
            norm.bias.fill_(1.0)
        linear1 = _filled_linear(2, 2)
        linear2 = _filled_linear(2, 2)
        subgraph = NormLinearSubgraph(norm=norm, linears=[linear1, linear2])
        strategy = NormLinearSubgraphFusion()

        strategy.apply_fusion(
            subgraph,
            {"scales": torch.tensor([2.0, 4.0])},
            {
                "linear_shifts": [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
                "norm_shift": torch.tensor([5.0, 6.0]),
            },
        )

        torch.testing.assert_close(linear1.weight, torch.tensor([[2.0, 4.0], [2.0, 4.0]]))
        torch.testing.assert_close(linear1.bias, torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(linear2.weight, torch.tensor([[2.0, 4.0], [2.0, 4.0]]))
        torch.testing.assert_close(linear2.bias, torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(norm.weight, torch.tensor([0.5, 0.25]))
        torch.testing.assert_close(norm.bias, torch.tensor([5.5, 6.25]))


class TestSubgraphFusionFactory(unittest.TestCase):
    def test_get_fuser_when_subgraph_type_supported_then_returns_registered_strategy(self):
        linears = [_filled_linear(2, 2), _filled_linear(2, 2)]
        cases = (
            (
                OVSubgraph(o_proj=linears[0], v_proj=linears[1], num_attention_heads=1, key_value_heads=1),
                OVSubgraphFusion,
            ),
            (UpDownSubgraph(up_proj=linears[0], down_proj=linears[1], gate_proj=linears[0]), UpDownSubgraphFusion),
            (LinearLinearSubgraph(linear1=linears[0], linear2=linears[1]), LinearLinearSubgraphFusion),
            (NormLinearSubgraph(norm=nn.LayerNorm(2), linears=linears), NormLinearSubgraphFusion),
            (NonFusionSubgraph(linears=linears), NonFusionSubgraphFusion),
        )

        for subgraph, expected_type in cases:
            with self.subTest(subgraph=type(subgraph).__name__):
                self.assertIsInstance(SubgraphFusionFactory.get_fuser(subgraph), expected_type)

    def test_get_fuser_when_subgraph_type_unknown_then_returns_none(self):
        self.assertIsNone(SubgraphFusionFactory.get_fuser(Subgraph()))

    def test_apply_fusion_to_subgraph_when_subgraph_type_unknown_then_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Unsupported subgraph type: Subgraph"):
            SubgraphFusionFactory.apply_fusion_to_subgraph(Subgraph(), {}, None)


if __name__ == "__main__":
    unittest.main()
