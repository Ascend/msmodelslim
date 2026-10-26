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

Unit tests for anti_outlier.common.subgraph_type dataclasses.
"""

import unittest

from torch import nn

from msmodelslim.processor.anti_outlier.common.subgraph_type import (
    LinearLinearSubgraph,
    NonFusionSubgraph,
    NormLinearSubgraph,
    OVSubgraph,
    Subgraph,
    UpDownSubgraph,
)


class TestSubgraphDataclasses(unittest.TestCase):
    """Each subgraph dataclass stores its modules and optional name metadata."""

    def test_all_subgraphs_derive_from_subgraph_base(self):
        for cls in (NonFusionSubgraph, NormLinearSubgraph, LinearLinearSubgraph, OVSubgraph, UpDownSubgraph):
            self.assertTrue(issubclass(cls, Subgraph))

    def test_non_fusion_subgraph_defaults(self):
        linears = [nn.Linear(2, 2)]
        subgraph = NonFusionSubgraph(linears=linears)
        self.assertIs(subgraph.linears, linears)
        self.assertIsNone(subgraph.linear_names)

    def test_norm_linear_subgraph_stores_norm_and_names(self):
        norm = nn.Identity()
        linears = [nn.Linear(3, 4), nn.Linear(3, 5)]
        subgraph = NormLinearSubgraph(norm=norm, linears=linears, linear_names=["a", "b"])
        self.assertIs(subgraph.norm, norm)
        self.assertEqual(subgraph.linear_names, ["a", "b"])

    def test_linear_linear_subgraph_defaults_names_to_none(self):
        linear1, linear2 = nn.Linear(4, 4), nn.Linear(4, 8)
        subgraph = LinearLinearSubgraph(linear1=linear1, linear2=linear2)
        self.assertIs(subgraph.linear1, linear1)
        self.assertIs(subgraph.linear2, linear2)
        self.assertIsNone(subgraph.linear1_name)
        self.assertIsNone(subgraph.linear2_name)

    def test_ov_subgraph_stores_head_counts(self):
        subgraph = OVSubgraph(
            o_proj=nn.Linear(16, 8),
            v_proj=nn.Linear(8, 16),
            num_attention_heads=4,
            key_value_heads=2,
        )
        self.assertEqual(subgraph.num_attention_heads, 4)
        self.assertEqual(subgraph.key_value_heads, 2)
        self.assertIsNone(subgraph.extra_config)

    def test_ov_subgraph_keeps_extra_config_and_names(self):
        extra = {"head_dim": 64}
        subgraph = OVSubgraph(
            o_proj=nn.Linear(16, 8),
            v_proj=nn.Linear(8, 16),
            num_attention_heads=4,
            key_value_heads=4,
            extra_config=extra,
            o_proj_name="o",
            v_proj_name="v",
        )
        self.assertIs(subgraph.extra_config, extra)
        self.assertEqual(subgraph.o_proj_name, "o")
        self.assertEqual(subgraph.v_proj_name, "v")

    def test_up_down_subgraph_stores_three_linears(self):
        subgraph = UpDownSubgraph(
            up_proj=nn.Linear(4, 8),
            down_proj=nn.Linear(8, 4),
            gate_proj=nn.Linear(4, 8),
        )
        self.assertEqual(subgraph.up_proj.out_features, 8)
        self.assertEqual(subgraph.down_proj.out_features, 4)
        self.assertIsNone(subgraph.gate_proj_name)


if __name__ == '__main__':
    unittest.main()
