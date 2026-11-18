#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import types
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from msmodelslim.core.train.metrics import (
    QParamStats,
    QParamStatsCollector,
    TokenThroughputTracker,
    is_eval_log,
    qparam_summary_distributed,
)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([6.0]))


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([2.0, 4.0]))
        self.dense_weight = nn.Parameter(torch.tensor([100.0]))
        self.block = _Block()


class _NoQparamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_weight = nn.Parameter(torch.tensor([100.0]))


class _ShardParam(nn.Parameter):
    """模拟 DTensor 分片参数：to_local() 返回本地值（或抛错/返回非张量）。"""

    def __new__(cls, data, local=None, error=None):
        return super().__new__(cls, data)

    def __init__(self, data, local=None, error=None):
        self._local = local
        self._error = error

    def to_local(self):
        if self._error is not None:
            raise self._error
        return self._local


class _FakeReduceOp:
    SUM = "fake_sum"
    MIN = "fake_min"
    MAX = "fake_max"


def _stub_logger(monkeypatch) -> MagicMock:
    fake_logger = MagicMock()
    monkeypatch.setattr("msmodelslim.core.train.metrics.get_logger", lambda: fake_logger)
    return fake_logger


def _fake_dist(monkeypatch, all_reduce) -> None:
    fake = types.ModuleType("fake_torch_distributed")
    fake.is_available = lambda: True
    fake.is_initialized = lambda: True
    fake.get_world_size = lambda: 2
    fake.ReduceOp = _FakeReduceOp
    fake.all_reduce = all_reduce
    monkeypatch.setattr("msmodelslim.core.train.metrics.dist", fake)


def _stub_qparam_summary(monkeypatch, stats) -> list:
    calls = []

    def _fake_summary(model, param_names):
        calls.append((model, param_names))
        return stats

    monkeypatch.setattr("msmodelslim.core.train.metrics.qparam_summary_distributed", _fake_summary)
    return calls


def test_is_eval_log_returns_true_when_payload_contains_eval_metric():
    assert is_eval_log({"eval_loss": 0.5}) is True
    assert is_eval_log({"loss": 0.5, "eval_accuracy": 0.9}) is True


def test_is_eval_log_returns_false_when_payload_contains_train_metrics_only():
    assert is_eval_log({"loss": 0.5, "grad_norm": 1.0}) is False
    assert is_eval_log({}) is False


def test_is_eval_log_returns_false_when_metric_keys_non_string():
    assert is_eval_log({1: 0.5, (2, 3): 0.1}) is False


def test_QParamStats_stores_stats_fields_when_constructed():
    stats = QParamStats(count=3, mean=4.0, vmin=2.0, vmax=6.0)
    assert (stats.count, stats.mean, stats.vmin, stats.vmax) == (3, 4.0, 2.0, 6.0)
    assert stats == QParamStats(count=3, mean=4.0, vmin=2.0, vmax=6.0)
    with pytest.raises(FrozenInstanceError):
        stats.count = 5


def test_qparam_summary_distributed_uses_local_shard_when_param_has_to_local():
    class _ShardedModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 参数全局视图为零值，本地分片为 [3.0, 5.0]：统计须取 to_local() 结果。
            self.alpha = _ShardParam(torch.zeros(2), local=torch.tensor([3.0, 5.0]))

    stats = qparam_summary_distributed(_ShardedModel(), ["alpha"])
    assert stats.count == 2
    assert stats.mean == pytest.approx(4.0)
    assert stats.vmin == pytest.approx(3.0)
    assert stats.vmax == pytest.approx(5.0)


def test_qparam_summary_distributed_returns_none_when_to_local_raises():
    class _BrokenShardModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.alpha = _ShardParam(torch.tensor([1.0]), error=RuntimeError("shard broken"))

    assert qparam_summary_distributed(_BrokenShardModel(), ["alpha"]) is None


def test_qparam_summary_distributed_returns_none_when_to_local_not_tensor():
    class _NonTensorShardModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.alpha = _ShardParam(torch.tensor([1.0]), local=3.5)

    assert qparam_summary_distributed(_NonTensorShardModel(), ["alpha"]) is None


def test_qparam_summary_distributed_returns_none_when_qparam_empty():
    class _EmptyQparamModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.alpha = nn.Parameter(torch.tensor([]))

    assert qparam_summary_distributed(_EmptyQparamModel(), ["alpha"]) is None


def test_qparam_summary_distributed_returns_local_stats_when_dist_not_initialized():
    stats = qparam_summary_distributed(_ToyModel(), ["alpha"])
    assert stats.count == 3
    assert stats.mean == pytest.approx(4.0)
    assert stats.vmin == pytest.approx(2.0)
    assert stats.vmax == pytest.approx(6.0)


def test_qparam_summary_distributed_aggregates_when_dist_initialized(monkeypatch):
    ops_log = []

    def _peer_reduce(tensor, op=None):
        ops_log.append(op)
        if op == _FakeReduceOp.SUM:
            tensor.add_(torch.tensor([3.0, 30.0]))
        elif op == _FakeReduceOp.MIN:
            tensor.fill_(min(float(tensor.item()), 0.5))
        else:
            tensor.fill_(max(float(tensor.item()), 9.0))

    _fake_dist(monkeypatch, all_reduce=_peer_reduce)
    stats = qparam_summary_distributed(_ToyModel(), ["alpha"])
    assert ops_log == [_FakeReduceOp.SUM, _FakeReduceOp.MIN, _FakeReduceOp.MAX]
    assert stats.count == 6
    assert stats.mean == pytest.approx(7.0)
    assert stats.vmin == pytest.approx(0.5)
    assert stats.vmax == pytest.approx(9.0)


def test_qparam_summary_distributed_warns_when_all_reduce_fails(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    _fake_dist(monkeypatch, all_reduce=MagicMock(side_effect=RuntimeError("hccl down")))
    stats = qparam_summary_distributed(_ToyModel(), ["alpha"])
    assert stats.count == 3
    assert stats.mean == pytest.approx(4.0)
    assert stats.vmin == pytest.approx(2.0)
    assert stats.vmax == pytest.approx(6.0)
    fake_logger.warning.assert_called_once()
    assert fake_logger.warning.call_args.args[0] == "QAT qparam all-reduce failed; using local-shard stats."


def test_qparam_summary_distributed_returns_none_when_no_qparam_matched():
    assert qparam_summary_distributed(_NoQparamModel(), ["alpha"]) is None


def test_TokenThroughputTracker_consume_per_sec_returns_none_when_first_call():
    tracker = TokenThroughputTracker()
    tracker.add_tokens(128)
    assert tracker.consume_per_sec() is None


def test_TokenThroughputTracker_consume_per_sec_returns_none_when_elapsed_not_positive(monkeypatch):
    fake_time = MagicMock()
    fake_time.perf_counter = MagicMock(side_effect=[5.0, 5.0])
    monkeypatch.setattr("msmodelslim.core.train.metrics.time", fake_time)
    tracker = TokenThroughputTracker()
    tracker.reset()
    tracker.add_tokens(64)
    assert tracker.consume_per_sec() is None


def test_TokenThroughputTracker_consume_per_sec_returns_throughput_when_second_window(monkeypatch):
    fake_time = MagicMock()
    fake_time.perf_counter = MagicMock(side_effect=[100.0, 102.0])
    monkeypatch.setattr("msmodelslim.core.train.metrics.time", fake_time)
    tracker = TokenThroughputTracker()
    tracker.add_tokens(1500)
    assert tracker.consume_per_sec() is None
    tracker.add_tokens(1000)
    throughput = tracker.consume_per_sec()
    assert throughput == pytest.approx(500.0)
    assert throughput > 0


def test_TokenThroughputTracker_reset_clears_tokens_when_window_restarted(monkeypatch):
    fake_time = MagicMock()
    fake_time.perf_counter = MagicMock(side_effect=[1.0, 2.0])
    monkeypatch.setattr("msmodelslim.core.train.metrics.time", fake_time)
    tracker = TokenThroughputTracker()
    tracker.add_tokens(7)
    tracker.reset()
    tracker.add_tokens(3)
    assert tracker.consume_per_sec() == pytest.approx(3.0)


def test_QParamStatsCollector_summary_refreshes_cache_when_first_call(monkeypatch):
    stats = QParamStats(count=3, mean=4.0, vmin=2.0, vmax=6.0)
    calls = _stub_qparam_summary(monkeypatch, stats)
    collector = QParamStatsCollector(["alpha"])
    model = _ToyModel()
    assert collector.summary(model, step=0, is_eval=False) is stats
    assert calls == [(model, ("alpha",))]


def test_QParamStatsCollector_summary_reuses_cache_when_within_interval(monkeypatch):
    stats = QParamStats(count=3, mean=4.0, vmin=2.0, vmax=6.0)
    calls = _stub_qparam_summary(monkeypatch, stats)
    collector = QParamStatsCollector(["alpha"])
    model = _ToyModel()
    first = collector.summary(model, step=0, is_eval=False)
    second = collector.summary(model, step=5, is_eval=False)
    third = collector.summary(model, step=9, is_eval=False)
    assert first is second is third is stats
    assert len(calls) == 1


def test_QParamStatsCollector_summary_refreshes_cache_when_interval_elapsed(monkeypatch):
    made = []

    def _fake_summary(model, param_names):
        made.append(QParamStats(count=len(made) + 1, mean=1.0, vmin=1.0, vmax=1.0))
        return made[-1]

    monkeypatch.setattr("msmodelslim.core.train.metrics.qparam_summary_distributed", _fake_summary)
    collector = QParamStatsCollector(["alpha"])
    model = _ToyModel()
    first = collector.summary(model, step=0, is_eval=False)
    cached = collector.summary(model, step=9, is_eval=False)
    second = collector.summary(model, step=10, is_eval=False)
    assert first is cached
    assert second is not first
    assert len(made) == 2


def test_QParamStatsCollector_summary_clamps_interval_when_min_interval_below_floor(monkeypatch):
    stats = QParamStats(count=3, mean=4.0, vmin=2.0, vmax=6.0)
    calls = _stub_qparam_summary(monkeypatch, stats)
    collector = QParamStatsCollector(["alpha"])
    model = _ToyModel()
    collector.summary(model, step=0, is_eval=False, min_interval=2)
    collector.summary(model, step=9, is_eval=False, min_interval=2)
    assert len(calls) == 1


def test_QParamStatsCollector_summary_honors_interval_when_min_interval_above_floor(monkeypatch):
    stats = QParamStats(count=3, mean=4.0, vmin=2.0, vmax=6.0)
    calls = _stub_qparam_summary(monkeypatch, stats)
    collector = QParamStatsCollector(["alpha"])
    model = _ToyModel()
    collector.summary(model, step=0, is_eval=False, min_interval=100)
    collector.summary(model, step=99, is_eval=False, min_interval=100)
    assert len(calls) == 1
    collector.summary(model, step=100, is_eval=False, min_interval=100)
    assert len(calls) == 2


def test_QParamStatsCollector_summary_forces_refresh_when_is_eval_true(monkeypatch):
    stats = QParamStats(count=3, mean=4.0, vmin=2.0, vmax=6.0)
    calls = _stub_qparam_summary(monkeypatch, stats)
    collector = QParamStatsCollector(["alpha"])
    model = _ToyModel()
    collector.summary(model, step=0, is_eval=False)
    collector.summary(model, step=1, is_eval=True)
    assert len(calls) == 2


def test_QParamStatsCollector_summary_returns_none_when_model_has_no_qparam():
    collector = QParamStatsCollector(["alpha"])
    assert collector.summary(_NoQparamModel(), step=0, is_eval=False) is None
