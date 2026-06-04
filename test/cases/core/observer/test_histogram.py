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

from msmodelslim import ir as _ir  # noqa: F401 — break circular import via quantizer package
from msmodelslim.core.observer.histogram import (
    HistogramObserver,
    HistogramObserverConfig,
    SearchMethod,
)
from msmodelslim.utils.exception import SpecError, SchemaValidateError


class TestSearchMethod:
    """Tests for SearchMethod enum."""

    def test_enum_contains_l2_norm_and_kl_divergence_when_accessed(self):
        """场景：访问 SearchMethod 成员。预期：包含 l2_norm 与 kl_divergence。"""
        assert SearchMethod.L2_NORM.value == "l2_norm"
        assert SearchMethod.KL_DIVERGENCE.value == "kl_divergence"
        assert SearchMethod("l2_norm") is SearchMethod.L2_NORM


class TestHistogramObserverConfig:
    """Tests for HistogramObserverConfig."""

    def test_default_config_when_constructed(self):
        """场景：默认构造。预期：symmetric=False，search_method=L2_NORM。"""
        config = HistogramObserverConfig()
        assert config.symmetric is False
        assert config.search_method == SearchMethod.L2_NORM

    def test_raises_schema_validate_error_when_invalid_dtype(self):
        """场景：无效 dtype。预期：SchemaValidateError。"""
        with pytest.raises(SchemaValidateError):
            HistogramObserverConfig(dtype="int16")


class TestHistogramObserver:
    """Tests for HistogramObserver."""

    def test_update_then_get_clip_bounds_returns_tensors_when_valid_tensor(self):
        """场景：update 合法张量后查询 clip bounds。预期：返回非 None 的 min/max。"""
        observer = HistogramObserver(HistogramObserverConfig())
        observer.update(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]))
        clip_min, clip_max = observer.get_clip_bounds()
        assert clip_min is not None
        assert clip_max is not None
        assert clip_min <= clip_max

    def test_get_clip_bounds_raises_spec_error_when_not_updated(self):
        """场景：未 update 直接查询。预期：SpecError。"""
        observer = HistogramObserver(HistogramObserverConfig())
        with pytest.raises(SpecError, match="Clip min or clip max is not set"):
            observer.get_clip_bounds()

    def test_update_raises_spec_error_when_empty_tensor(self):
        """场景：update 空张量。预期：SpecError 提示 empty。"""
        observer = HistogramObserver(HistogramObserverConfig())
        with pytest.raises(SpecError, match="empty"):
            observer.update(torch.tensor([]))

    def test_update_sets_clip_bounds_when_all_same_value(self):
        """场景：张量元素全相同。预期：update 成功且 clip bounds 等于该值。"""
        observer = HistogramObserver(HistogramObserverConfig())
        observer.update(torch.tensor([2.0, 2.0, 2.0]))
        clip_min, clip_max = observer.get_clip_bounds()
        assert clip_min.item() == pytest.approx(2.0)
        assert clip_max.item() == pytest.approx(2.0)
