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

import pytest
import torch

from msmodelslim.core.observer.minmax import (
    MinMaxBlockObserverConfig,
    MinMaxObserverConfig,
    MsMinMaxBlockObserver,
    MsMinMaxObserver,
)
from msmodelslim.utils.exception import SpecError


class TestMsMinMaxObserver:
    """Tests for MsMinMaxObserver."""

    def test_get_min_max_returns_values_when_updated(self):
        """场景：update 后查询。预期：min/max 与输入一致。"""
        observer = MsMinMaxObserver(MinMaxObserverConfig(dim=[], keepdim=False, aggregation_type="max"))
        x = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
        observer.update(x)
        min_val, max_val = observer.get_min_max()
        assert min_val.item() == pytest.approx(1.0)
        assert max_val.item() == pytest.approx(5.0)

    def test_get_min_max_raises_spec_error_when_not_updated(self):
        """场景：未 update 直接查询。预期：SpecError。"""
        observer = MsMinMaxObserver(MinMaxObserverConfig())
        with pytest.raises(SpecError, match="no any update_stats"):
            observer.get_min_max()

    def test_reset_clears_state_when_called_after_update(self):
        """场景：update 后 reset。预期：再次查询抛 SpecError。"""
        observer = MsMinMaxObserver(MinMaxObserverConfig())
        observer.update(torch.tensor([1.0, 2.0]))
        observer.reset()
        with pytest.raises(SpecError):
            observer.get_min_max()

    def test_mean_aggregation_tracks_running_mean_when_multiple_updates(self):
        """场景：aggregation_type=mean 多次 update。预期：get_min_max 可调用。"""
        observer = MsMinMaxObserver(MinMaxObserverConfig(aggregation_type="mean"))
        observer.update(torch.tensor([1.0, 3.0]))
        observer.update(torch.tensor([2.0, 4.0]))
        min_val, max_val = observer.get_min_max()
        assert min_val is not None
        assert max_val is not None

    def test_max_aggregation_merges_min_max_when_second_update(self):
        """场景：aggregation_type=max 两次 update。预期：min/max 取全局极值。"""
        observer = MsMinMaxObserver(MinMaxObserverConfig(aggregation_type="max"))
        observer.update(torch.tensor([1.0, 5.0]))
        observer.update(torch.tensor([0.0, 3.0]))
        min_val, max_val = observer.get_min_max()
        assert min_val.item() == pytest.approx(0.0)
        assert max_val.item() == pytest.approx(5.0)


class TestMsMinMaxBlockObserver:
    """Tests for MsMinMaxBlockObserver."""

    def test_get_min_max_returns_values_when_max_method_updated(self):
        """场景：method=max 单次 update。预期：min/max 非 None。"""
        observer = MsMinMaxBlockObserver(MinMaxBlockObserverConfig(method="max"))
        observer.update(torch.tensor([[-1.0, 2.0], [3.0, -4.0]]), sync=False)
        min_val, max_val = observer.get_min_max()
        assert max_val.item() == pytest.approx(4.0)

    def test_get_min_max_raises_spec_error_when_not_updated(self):
        """场景：未 update。预期：SpecError。"""
        observer = MsMinMaxBlockObserver(MinMaxBlockObserverConfig())
        with pytest.raises(SpecError):
            observer.get_min_max()

    def test_none_method_stores_abs_values_when_updated(self):
        """场景：method=none。预期：逐元素绝对值。"""
        observer = MsMinMaxBlockObserver(MinMaxBlockObserverConfig(method="none"))
        x = torch.tensor([-2.0, 3.0])
        observer.update(x, sync=False)
        min_val, max_val = observer.get_min_max()
        assert torch.allclose(min_val, torch.tensor([2.0, 3.0]))
