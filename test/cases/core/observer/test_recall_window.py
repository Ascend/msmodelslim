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

from msmodelslim.core.observer.recall_window import (
    RecallWindowObserver,
    RecallWindowObserverConfig,
    recall_window,
)
from msmodelslim.utils.exception import SpecError


class TestRecallWindowObserver:
    """Tests for RecallWindowObserver."""

    def test_get_min_returns_value_when_updated(self):
        """场景：update 后 get_min。预期：返回 tensor。"""
        observer = RecallWindowObserver(RecallWindowObserverConfig(ratio=1.0))
        observer.update(torch.tensor([[1.0, 5.0, 3.0]]))
        assert observer.get_min().numel() >= 1

    def test_get_min_raises_spec_error_when_not_updated(self):
        """场景：未 update。预期：SpecError。"""
        observer = RecallWindowObserver(RecallWindowObserverConfig())
        with pytest.raises(SpecError):
            observer.get_min()

    def test_reset_clears_state_when_called(self):
        """场景：update 后 reset。预期：get_max 抛错。"""
        observer = RecallWindowObserver(RecallWindowObserverConfig())
        observer.update(torch.tensor([[1.0, 2.0, 3.0]]))
        observer.reset()
        with pytest.raises(SpecError):
            observer.get_max()


class TestRecallWindowFunction:
    """Tests for recall_window module function."""

    def test_recall_window_returns_endpoints_when_ratio_one(self):
        """场景：ratio=1 覆盖全窗口。预期：left <= right。"""
        x = torch.tensor([[3.0, 1.0, 4.0, 2.0]])
        left, right = recall_window(x, ratio=1.0, dim=-1)
        assert left.numel() == 1
        assert right.numel() == 1
        assert left.item() <= right.item()

    def test_recall_window_respects_keepdim_when_true(self):
        """场景：keepdim=True。预期：输出维度与输入一致。"""
        x = torch.tensor([[1.0, 2.0, 3.0]])
        left, right = recall_window(x, ratio=0.5, dim=-1, keepdim=True)
        assert left.dim() == x.dim()
        assert right.dim() == x.dim()
