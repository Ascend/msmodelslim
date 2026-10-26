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
# pylint: disable=consider-using-from-import

import pytest
import torch
from torch import nn

from msmodelslim.core.context import ContextFactory, ContextManager
from msmodelslim.processor.anti_outlier.awq.awq_stats_collector import AWQStatsCollector
from msmodelslim.processor.anti_outlier.common import HookManager


class SimpleBlock(nn.Module):
    """A minimal block with a linear layer, used as both target and parent."""

    def __init__(self, in_features=16, out_features=8):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.fc(x)


class TwoLayerModel(nn.Module):
    """Model with named sub-blocks so observe_activation / observe_kwargs can find them."""

    def __init__(self, in_features=16, out_features=8):
        super().__init__()
        self.block = SimpleBlock(in_features, out_features)

    def forward(self, x):
        return self.block(x)


def _make_collector(in_features=16, out_features=8):
    model = TwoLayerModel(in_features, out_features)
    collector = AWQStatsCollector(model)
    return collector, model


@pytest.fixture(autouse=True)
def _context():
    """Ensure a global context is active for every test."""
    ctx = ContextFactory().create(is_distributed=False)
    with ContextManager(ctx):
        yield


class TestAWQStatsCollectorInit:
    def test_init_creates_empty_stats(self):
        collector, _ = _make_collector()
        assert len(collector.global_stats) == 0

    def test_init_creates_hook_manager(self):
        collector, _ = _make_collector()
        assert isinstance(collector.hook_manager, HookManager)
        assert len(collector.hook_manager.hook_handles) == 0


class TestObserveActivation:
    def test_single_forward_collects_mean(self):
        collector, model = _make_collector(in_features=16)
        collector.observe_activation("block.fc")
        x = torch.randn(2, 4, 16)
        model(x)
        mean = collector.get_activation_mean("block.fc")
        assert mean.shape == (16,)

    def test_multiple_forwards_accumulate_mean(self):
        collector, model = _make_collector(in_features=16)
        collector.observe_activation("block.fc")

        # First forward: all ones → abs mean = 1.0
        x1 = torch.ones(1, 1, 16)
        model(x1)
        mean_after_first = collector.get_activation_mean("block.fc")
        assert mean_after_first.shape == (16,)
        mean_after_first = mean_after_first.clone()

        # Second forward: all twos → abs mean = 2.0
        x2 = torch.full((1, 1, 16), 2.0)
        model(x2)
        mean_after_second = collector.get_activation_mean("block.fc")
        assert mean_after_second.shape == (16,)

        # Running mean should differ from first-batch-only mean
        assert not torch.equal(mean_after_first, mean_after_second)
        # With equal counts (1 sample each), running mean should be (1+2)/2 = 1.5
        torch.testing.assert_close(mean_after_second, torch.full((16,), 1.5))

    def test_mean_shape_matches_input_features(self):
        collector, model = _make_collector(in_features=32)
        collector.observe_activation("block.fc")
        model(torch.randn(1, 4, 32))
        mean = collector.get_activation_mean("block.fc")
        assert mean.dtype == torch.float32
        assert mean.shape == (32,)

    def test_mean_is_on_cpu(self):
        collector, model = _make_collector()
        collector.observe_activation("block.fc")
        model(torch.randn(1, 4, 16))
        mean = collector.get_activation_mean("block.fc")
        assert mean.dtype == torch.float32
        assert mean.device == torch.device("cpu")


class TestObserveKwargs:
    def test_single_forward_collects_kwargs(self):
        collector, model = _make_collector()
        collector.observe_kwargs("block")
        model(torch.randn(1, 4, 16))
        kwargs = collector.get_block_kwargs("block")
        assert isinstance(kwargs, list)
        assert set(kwargs[0]) == {"x"}
        assert kwargs[0]["x"].shape == (1, 4, 16)
        assert len(kwargs) == 1

    def test_multiple_forwards_append_kwargs(self):
        collector, model = _make_collector()
        collector.observe_kwargs("block")
        model(torch.randn(1, 4, 16))
        model(torch.randn(1, 4, 16))
        kwargs = collector.get_block_kwargs("block")
        assert isinstance(kwargs, list)
        assert set(kwargs[0]) == {"x"}
        assert kwargs[0]["x"].shape == (1, 4, 16)
        assert len(kwargs) == 2

    def test_duplicate_observe_kwargs_is_idempotent(self):
        collector, model = _make_collector()
        collector.observe_kwargs("block")
        collector.observe_kwargs("block")
        # Only one hook should be installed
        hook_count = len(collector.hook_manager.hook_handles)
        assert hook_count == 1


class TestGetActivationMean:
    def test_returns_none_for_unobserved_module(self):
        collector, _ = _make_collector()
        assert collector.get_activation_mean("block.fc") is None

    def test_returns_none_before_forward(self):
        collector, _ = _make_collector()
        collector.observe_activation("block.fc")
        assert collector.get_activation_mean("block.fc") is None


class TestGetBlockKwargs:
    def test_returns_none_for_unobserved_module(self):
        collector, _ = _make_collector()
        assert collector.get_block_kwargs("block") is None

    def test_returns_none_before_forward(self):
        collector, _ = _make_collector()
        collector.observe_kwargs("block")
        assert collector.get_block_kwargs("block") is None


class TestStopObserving:
    def test_hooks_removed_after_stop(self):
        collector, model = _make_collector()
        collector.observe_activation("block.fc")
        model(torch.randn(1, 4, 16))
        mean_before = collector.get_activation_mean("block.fc")
        assert mean_before.shape == (16,)
        mean_before = mean_before.clone()

        collector.stop_observing()
        assert len(collector.hook_manager.hook_handles) == 0

        # New forward should NOT update stats
        model(torch.full((1, 1, 16), 999.0))
        mean_after = collector.get_activation_mean("block.fc")
        torch.testing.assert_close(mean_before, mean_after)

    def test_stats_preserved_after_stop(self):
        collector, model = _make_collector()
        collector.observe_activation("block.fc")
        model(torch.randn(1, 4, 16))

        collector.stop_observing()
        assert collector.get_activation_mean("block.fc").shape == (16,)


class TestClearStats:
    def test_clear_empties_all_stats(self):
        collector, model = _make_collector()
        collector.observe_activation("block.fc")
        collector.observe_kwargs("block")
        model(torch.randn(1, 4, 16))

        collector.clear_stats()
        assert collector.get_activation_mean("block.fc") is None
        assert collector.get_block_kwargs("block") is None

    def test_hooks_still_active_after_clear(self):
        collector, model = _make_collector()
        collector.observe_activation("block.fc")
        model(torch.randn(1, 4, 16))

        collector.clear_stats()
        assert collector.get_activation_mean("block.fc") is None

        # Hooks still active — new forward repopulates stats
        model(torch.randn(1, 4, 16))
        assert collector.get_activation_mean("block.fc").shape == (16,)
