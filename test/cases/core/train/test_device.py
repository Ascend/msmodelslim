#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import types
from unittest.mock import MagicMock

import pytest
import torch

from msmodelslim.core.train.device import (
    accelerator_device,
    all_reduce_sum_float,
    empty_device_cache,
    get_device_memory,
    log_device_memory,
    os_local_rank,
)


def _stub_logger(monkeypatch) -> MagicMock:
    fake_logger = MagicMock()
    monkeypatch.setattr("msmodelslim.core.train.device.get_logger", lambda: fake_logger)
    return fake_logger


def _install_fake_torch_npu(monkeypatch, npu) -> MagicMock:
    fake_module = types.ModuleType("torch_npu")
    fake_module.npu = npu
    monkeypatch.setitem(sys.modules, "torch_npu", fake_module)
    return fake_module


def _install_fake_dist(monkeypatch, **overrides) -> MagicMock:
    fake_dist = MagicMock()
    fake_dist.is_available.return_value = overrides.get("available", True)
    fake_dist.is_initialized.return_value = overrides.get("initialized", True)
    fake_dist.get_world_size.return_value = overrides.get("world_size", 1)
    if "all_reduce" in overrides:
        fake_dist.all_reduce.side_effect = overrides["all_reduce"]
    monkeypatch.setattr("msmodelslim.core.train.device.dist", fake_dist)
    return fake_dist


def test_accelerator_device_returns_cpu_when_no_accelerator():
    assert accelerator_device() == torch.device("cpu")


def test_get_device_memory_returns_none_stats_when_no_accelerator():
    assert get_device_memory() == ("cpu", None, None)


def test_os_local_rank_returns_local_rank_when_local_rank_set(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("RANK", "9")
    assert os_local_rank() == "3"


def test_os_local_rank_returns_rank_when_local_rank_missing(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("RANK", "5")
    assert os_local_rank() == "5"


def test_os_local_rank_returns_zero_when_env_missing(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    assert os_local_rank() == "0"


def test_all_reduce_sum_float_returns_value_when_dist_uninitialized():
    assert all_reduce_sum_float(2.5) == 2.5


def test_empty_device_cache_returns_none_when_no_accelerator():
    assert empty_device_cache() is None


def _patch_device_ctor(monkeypatch, created, npu=None) -> None:
    # CPU 版 torch 未注册 "npu" 设备类型，torch.device("npu", x) 会抛 RuntimeError；
    # 真实 torch_npu 环境才会注册。这里只替换 device 模块内可见的构造器并记录参数。
    def _fake_device(*args):
        created.append(args)
        return MagicMock()

    monkeypatch.setattr(
        "msmodelslim.core.train.device.torch",
        types.SimpleNamespace(device=_fake_device, npu=npu),
    )


def test_accelerator_device_returns_npu_when_current_device_available(monkeypatch):
    fake_npu = MagicMock()
    fake_npu.is_available.return_value = True
    fake_npu.current_device.return_value = 3
    _install_fake_torch_npu(monkeypatch, fake_npu)
    created = []
    _patch_device_ctor(monkeypatch, created)
    accelerator_device()
    assert created == [("npu", 3)]


def test_accelerator_device_returns_npu_zero_when_current_device_missing(monkeypatch):
    fake_npu = types.SimpleNamespace(is_available=lambda: True)
    _install_fake_torch_npu(monkeypatch, fake_npu)
    created = []
    _patch_device_ctor(monkeypatch, created)
    accelerator_device()
    assert created == [("npu", 0)]


def test_accelerator_device_uses_torch_attr_when_torch_npu_unavailable(monkeypatch):
    unavailable_npu = MagicMock()
    unavailable_npu.is_available.return_value = False
    _install_fake_torch_npu(monkeypatch, unavailable_npu)
    fake_attr = types.SimpleNamespace(is_available=lambda: True, current_device=lambda: 7)
    created = []
    _patch_device_ctor(monkeypatch, created, npu=fake_attr)
    accelerator_device()
    assert created == [("npu", 7)]


def test_accelerator_device_returns_cpu_when_accelerators_unavailable(monkeypatch):
    unavailable_npu = MagicMock()
    unavailable_npu.is_available.return_value = False
    _install_fake_torch_npu(monkeypatch, unavailable_npu)
    # CPU 版 torch 无 torch.npu 属性，_patch_device_ctor 未注入 npu 即模拟该形态。
    created = []
    _patch_device_ctor(monkeypatch, created)
    accelerator_device()
    assert created == [("cpu",)]


def test_get_device_memory_returns_npu_stats_when_torch_npu_available(monkeypatch):
    fake_npu = MagicMock()
    fake_npu.is_available.return_value = True
    fake_npu.memory_allocated.return_value = 2 * 1024**3
    fake_npu.memory_reserved.return_value = 1024**3
    _install_fake_torch_npu(monkeypatch, fake_npu)
    assert get_device_memory() == ("npu", 2.0, 1.0)


def test_empty_device_cache_calls_empty_cache_when_npu_available(monkeypatch):
    fake_npu = MagicMock()
    fake_npu.is_available.return_value = True
    _install_fake_torch_npu(monkeypatch, fake_npu)
    empty_device_cache()
    fake_npu.empty_cache.assert_called_once_with()


def test_all_reduce_sum_float_returns_sum_when_world_size_two(monkeypatch):
    def _sum_inplace(tensor, op=None):
        tensor.add_(tensor)

    fake_dist = _install_fake_dist(monkeypatch, available=True, initialized=True, world_size=2, all_reduce=_sum_inplace)
    assert all_reduce_sum_float(1.5) == pytest.approx(3.0)
    fake_dist.all_reduce.assert_called_once()


def test_all_reduce_sum_float_returns_value_when_world_size_one(monkeypatch):
    fake_dist = _install_fake_dist(monkeypatch, world_size=1)
    assert all_reduce_sum_float(2.5) == 2.5
    fake_dist.all_reduce.assert_not_called()


def test_all_reduce_sum_float_returns_value_when_dist_unavailable(monkeypatch):
    fake_dist = _install_fake_dist(monkeypatch, available=False)
    assert all_reduce_sum_float(2.5) == 2.5
    fake_dist.all_reduce.assert_not_called()


def test_log_device_memory_logs_cpu_when_no_stats(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    log_device_memory("snapshot", rank=5)
    fake_logger.info.assert_called_once_with("QAT memory[%s] rank=%s device=cpu (no accelerator stats)", "snapshot", 5)


def test_log_device_memory_logs_npu_stats_when_memory_available(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    fake_npu = MagicMock()
    fake_npu.is_available.return_value = True
    fake_npu.memory_allocated.return_value = 2 * 1024**3
    fake_npu.memory_reserved.return_value = 1024**3
    _install_fake_torch_npu(monkeypatch, fake_npu)
    log_device_memory("snap", rank=2)
    fake_logger.info.assert_called_once_with(
        "QAT memory[%s] rank=%s device=%s allocated=%.3fGiB reserved=%.3fGiB",
        "snap",
        2,
        "npu",
        2.0,
        1.0,
    )


def test_log_device_memory_uses_os_local_rank_when_rank_missing(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    monkeypatch.setattr("msmodelslim.core.train.device.get_device_memory", lambda: ("npu", 1.5, 2.5))
    monkeypatch.setenv("LOCAL_RANK", "7")
    log_device_memory("snap")
    fake_logger.info.assert_called_once_with(
        "QAT memory[%s] rank=%s device=%s allocated=%.3fGiB reserved=%.3fGiB",
        "snap",
        "7",
        "npu",
        1.5,
        2.5,
    )
