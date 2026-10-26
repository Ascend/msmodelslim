#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
msmodelslim.processor.kv_smooth.processor 模块的单元测试
"""

import pytest
import torch
from torch import nn

from msmodelslim.processor.kv_smooth.fused_interface import KVSmoothFusedInterface
from msmodelslim.processor.kv_smooth.processor import (
    KVSmoothProcessor,
    KVSmoothProcessorConfig,
    KeyStatesMaximumCollector,
)
from msmodelslim.utils.exception import SchemaValidateError, ToDoError, UnsupportedError


class _Adapter(KVSmoothFusedInterface):
    def get_kvcache_smooth_fused_subgraph(self):
        return []

    def get_head_dim(self):
        return 8

    def get_num_key_value_groups(self):
        return 4

    def get_num_key_value_heads(self):
        return 4


class TestKeyStatesMaximumCollector:
    def test_collect_then_get_returns_max_absolute_value(self):
        collector = KeyStatesMaximumCollector()
        collector.collect(0, torch.tensor([[[1.0, -2.0]]]), torch.tensor([[0.0]]))

        result = collector.get(0)

        assert result.numel() == 1
        assert float(result.item()) == 2.0

    def test_get_before_collect_raises_todo_error(self):
        with pytest.raises(ToDoError):
            KeyStatesMaximumCollector().get(3)

    def test_reset_clears_collected_observers(self):
        collector = KeyStatesMaximumCollector()
        collector.collect(0, torch.tensor([[[1.0, 2.0]]]), torch.tensor([[0.0]]))

        collector.reset()

        with pytest.raises(ToDoError):
            collector.get(0)


class TestKVSmoothProcessorConfig:
    def test_defaults(self):
        config = KVSmoothProcessorConfig()
        assert config.type == "kv_smooth"
        assert config.smooth_factor == 1.0
        assert config.include == ["*"]
        assert config.exclude == []

    @pytest.mark.parametrize("factor", [0.0, -1.0])
    def test_smooth_factor_must_be_positive(self, factor):
        with pytest.raises(SchemaValidateError):
            KVSmoothProcessorConfig(smooth_factor=factor)


class TestKVSmoothProcessor:
    def test_rejects_adapter_not_implementing_interface(self):
        with pytest.raises(UnsupportedError):
            KVSmoothProcessor(nn.Module(), KVSmoothProcessorConfig(), adapter=object())

    def test_constructs_with_valid_adapter_and_reports_flags(self):
        processor = KVSmoothProcessor(nn.Module(), KVSmoothProcessorConfig(), adapter=_Adapter())

        assert processor.is_data_free() is False
        assert processor.support_distributed() is False
        assert processor.need_kv_cache() is True
