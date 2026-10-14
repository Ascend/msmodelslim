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

OASQStatsCollector 单元测试：对称/非对称统计、空输入与清理。
"""

import unittest
from unittest.mock import MagicMock

import torch
from torch import nn

from msmodelslim.processor.anti_outlier.common.smooth_components import StatKey
from msmodelslim.processor.anti_outlier.oasq.processor import OASQStatsCollector


class TestOASQStatsCollectorHookSymmetric(unittest.TestCase):
    """场景：symmetric=True 时收集 min/max/shift/channel_max。"""

    def setUp(self):
        self.collector = OASQStatsCollector(symmetric=True)
        self.linear = nn.Linear(4, 2)
        self.hook = self.collector.create_hook("layer.fc", subgraph_type="norm-linear")

    def test_stats_hook_when_valid_input_then_fills_min_max_shift_and_smooth_scale(self):
        """给定合法输入 tuple，期望写入 MAX/MIN/SHIFT/SMOOTH_SCALE。"""
        x = torch.tensor([[1.0, -2.0, 3.0, 0.5], [0.0, 1.0, -1.0, 2.0]])
        self.hook(self.linear, (x,), None)
        stats = self.collector.act_stats["layer.fc"]
        self.assertIn(StatKey.STAT_KEY_MAX, stats)
        self.assertIn(StatKey.STAT_KEY_MIN, stats)
        self.assertIn(StatKey.STAT_KEY_SHIFT, stats)
        self.assertIn(StatKey.STAT_KEY_SMOOTH_SCALE, stats)
        self.assertEqual(tuple(stats[StatKey.STAT_KEY_SMOOTH_SCALE].shape), (4,))

    def test_stats_hook_when_empty_input_tuple_then_skips_without_recording_stats(self):
        """给定空输入 tuple，期望跳过且不写入 act_stats。"""
        self.hook(self.linear, tuple(), None)
        self.assertNotIn("layer.fc", self.collector.act_stats)

    def test_stats_hook_when_input_not_tuple_then_skips_or_rejects_non_tuple(self):
        """给定非 tuple 输入（list），期望跳过且不写入 act_stats。"""
        self.hook(self.linear, [torch.randn(2, 4)], None)
        self.assertNotIn("layer.fc", self.collector.act_stats)

    def test_stats_hook_when_called_twice_then_observers_accumulate(self):
        """给定连续两次前向，期望仍保留 SMOOTH_SCALE 且 observer 可复用。"""
        x1 = torch.ones(2, 4)
        x2 = torch.ones(2, 4) * 2
        self.hook(self.linear, (x1,), None)
        first = self.collector.act_stats["layer.fc"][StatKey.STAT_KEY_SMOOTH_SCALE].clone()
        self.hook(self.linear, (x2,), None)
        second = self.collector.act_stats["layer.fc"][StatKey.STAT_KEY_SMOOTH_SCALE]
        self.assertEqual(tuple(second.shape), tuple(first.shape))
        # 第二次输入更大，channel_max 不应变小
        self.assertTrue(torch.all(second >= first - 1e-6))


class TestOASQStatsCollectorHookAsymmetric(unittest.TestCase):
    """场景：asymmetric + norm-linear 使用 shift 后绝对最大值。"""

    def test_stats_hook_when_asymmetric_norm_linear_then_uses_shifted_abs_for_channel_max(self):
        """给定 asymmetric 且 subgraph=norm-linear，期望仍写出 SMOOTH_SCALE。"""
        collector = OASQStatsCollector(symmetric=False)
        linear = nn.Linear(4, 2)
        hook = collector.create_hook("blk.q", subgraph_type="norm-linear")
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]], dtype=torch.float32)
        hook(linear, (x,), None)
        stats = collector.act_stats["blk.q"]
        self.assertIn(StatKey.STAT_KEY_SMOOTH_SCALE, stats)
        self.assertTrue(torch.all(stats[StatKey.STAT_KEY_SMOOTH_SCALE] >= 0))

    def test_stats_hook_when_asymmetric_but_non_asym_subgraph_then_falls_back_to_abs_max(self):
        """给定 asymmetric 但子图非 norm-linear，期望仍走 abs 路径且写出 SMOOTH_SCALE。"""
        collector = OASQStatsCollector(symmetric=False)
        linear = nn.Linear(4, 2)
        hook = collector.create_hook("blk.o", subgraph_type="ov")
        hook(linear, (torch.randn(3, 4),), None)
        self.assertIn(StatKey.STAT_KEY_SMOOTH_SCALE, collector.act_stats["blk.o"])


class TestOASQStatsCollectorDistAndClear(unittest.TestCase):
    """场景：DistHelper 同步开关与 clear_stats。"""

    def test_stats_hook_when_dist_helper_present_then_queries_is_shared(self):
        """给定 DistHelper，期望 update 前调用 is_shared(name)。"""
        collector = OASQStatsCollector(symmetric=True)
        dist_helper = MagicMock()
        dist_helper.is_shared.return_value = False
        collector.set_dist_helper(dist_helper)
        hook = collector.create_hook("shared.fc", subgraph_type="linear-linear")
        hook(nn.Linear(4, 2), (torch.randn(2, 4),), None)
        dist_helper.is_shared.assert_called_with("shared.fc")

    def test_clear_stats_when_observers_exist_then_clears_act_stats_and_observer_maps(self):
        """给定已收集统计，调用 clear_stats 后期望 act_stats 与 observer 字典为空。"""
        collector = OASQStatsCollector(symmetric=True)
        hook = collector.create_hook("fc", subgraph_type="norm-linear")
        hook(nn.Linear(4, 2), (torch.randn(2, 4),), None)
        self.assertTrue(collector.act_stats)
        self.assertTrue(collector.minmax_observers)
        self.assertTrue(collector.channel_max_observers)
        collector.clear_stats()
        self.assertEqual(collector.act_stats, {})
        self.assertEqual(collector.minmax_observers, {})
        self.assertEqual(collector.channel_max_observers, {})


if __name__ == "__main__":
    unittest.main()
