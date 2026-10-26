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

Unit tests for anti_outlier.common.smooth_components (HookManager / SubgraphRegistry / StatKey).
"""

# pylint: disable=abstract-class-instantiated

import unittest

import torch
from torch import nn

from msmodelslim.processor.anti_outlier.common.smooth_components import (
    HookManager,
    StatKey,
    StatsCollector,
    SubgraphRegistry,
)
from msmodelslim.processor.anti_outlier.common.subgraph_type import (
    LinearLinearSubgraph,
    NonFusionSubgraph,
    NormLinearSubgraph,
    OVSubgraph,
    UpDownSubgraph,
)


class TestHookManager(unittest.TestCase):
    """HookManager installs and removes forward/pre hooks and reports the count."""

    def setUp(self):
        self.model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))

    def test_install_hook_fires_on_forward_and_removes(self):
        manager = HookManager(self.model)
        seen = []
        self.assertTrue(manager.install_hook("0", lambda module, inputs, output: seen.append(output.shape)))
        self.model(torch.randn(2, 4))
        self.assertEqual(seen, [torch.Size([2, 4])])
        removed = manager.remove_all_hooks()
        self.assertEqual(removed, 1)
        self.assertEqual(len(manager.hook_handles), 0)
        self.model(torch.randn(2, 4))
        self.assertEqual(len(seen), 1)

    def test_install_pre_hook_fires_before_forward(self):
        manager = HookManager(self.model)
        calls = []
        self.assertTrue(manager.install_pre_hook("2", lambda module, inputs: calls.append(len(inputs))))
        self.model(torch.randn(2, 4))
        self.assertEqual(calls, [1])
        self.assertEqual(manager.remove_all_hooks(), 1)

    def test_remove_all_hooks_with_no_hooks_returns_zero(self):
        manager = HookManager(self.model)
        self.assertEqual(manager.remove_all_hooks(), 0)


class TestSubgraphRegistry(unittest.TestCase):
    """SubgraphRegistry maps subgraph classes to names and handler names."""

    def test_get_name_maps_known_types(self):
        self.assertEqual(SubgraphRegistry.get_name(NormLinearSubgraph), "norm-linear")
        self.assertEqual(SubgraphRegistry.get_name(LinearLinearSubgraph), "linear-linear")
        self.assertEqual(SubgraphRegistry.get_name(OVSubgraph), "ov")
        self.assertEqual(SubgraphRegistry.get_name(UpDownSubgraph), "up-down")
        self.assertEqual(SubgraphRegistry.get_name(NonFusionSubgraph), "non-fusion")

    def test_get_name_unknown_type_returns_unknown(self):
        self.assertEqual(SubgraphRegistry.get_name(int), "unknown")

    def test_get_handler_name(self):
        self.assertEqual(SubgraphRegistry.get_handler_name("norm-linear"), "_apply_norm_linear_smooth")
        self.assertIsNone(SubgraphRegistry.get_handler_name("non-fusion"))

    def test_is_supported(self):
        self.assertTrue(SubgraphRegistry.is_supported("ov"))
        self.assertFalse(SubgraphRegistry.is_supported("non-fusion"))

    def test_get_all_supported_types_returns_copy(self):
        first = SubgraphRegistry.get_all_supported_types()
        first.append("mutated")
        self.assertEqual(
            SubgraphRegistry.get_all_supported_types(),
            ["norm-linear", "linear-linear", "ov", "up-down"],
        )


class TestStatKeyAndStatsCollector(unittest.TestCase):
    """StatKey enum values and StatsCollector abstract contract."""

    def test_stat_key_values(self):
        self.assertEqual(StatKey.STAT_KEY_MAX.value, "max")
        self.assertEqual(StatKey.STAT_KEY_MIN.value, "min")
        self.assertEqual(StatKey.STAT_KEY_SMOOTH_SCALE.value, "smooth_scale")
        self.assertEqual(StatKey.STAT_KEY_VARIANCE.value, "std")
        self.assertEqual(StatKey.TENSOR.value, "tensor")

    def test_stats_collector_is_abstract(self):
        with self.assertRaises(TypeError):
            StatsCollector()

    def test_stats_collector_subclass_clear_stats(self):
        class _Collector(StatsCollector):
            def create_hook(self, name, subgraph_type=None):
                return lambda *a, **k: None

        collector = _Collector()
        collector.act_stats["layer"] = {"max": 1.0}
        collector.clear_stats()
        self.assertEqual(collector.act_stats, {})


if __name__ == '__main__':
    unittest.main()
