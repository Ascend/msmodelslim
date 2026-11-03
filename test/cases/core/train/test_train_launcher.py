#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import socket

import pytest

from msmodelslim.core.train.train_launcher import TrainLauncher
from msmodelslim.utils.exception import SchemaValidateError


def test_resolve_num_processes_returns_one_when_device_indices_none():
    assert TrainLauncher._resolve_num_processes(None) == 1


def test_resolve_num_processes_returns_len_when_device_indices_given():
    assert TrainLauncher._resolve_num_processes([0, 1, 3]) == 3


def test_resolve_num_processes_raises_when_device_indices_empty():
    with pytest.raises(SchemaValidateError, match="at least one"):
        TrainLauncher._resolve_num_processes([])


def test_port_is_free_returns_false_when_port_invalid():
    assert TrainLauncher._port_is_free("not-a-port") is False


def test_port_is_free_returns_true_when_port_available():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = str(sock.getsockname()[1])
    assert TrainLauncher._port_is_free(port) is True


def test_pick_master_port_uses_env_when_free(monkeypatch):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = str(sock.getsockname()[1])
    monkeypatch.setenv("MASTER_PORT", port)
    assert TrainLauncher._pick_master_port() == port


def test_build_launch_env_sets_fsdp_when_multi_process():
    env = TrainLauncher()._build_launch_env(
        num_processes=2,
        device=__import__("msmodelslim.core.const", fromlist=["DeviceType"]).DeviceType.NPU,
        device_indices=[0, 1],
    )
    assert env["ACCELERATE_USE_FSDP"] == "true"
    assert env["FSDP_VERSION"] == "2"
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "0,1"
    assert env["MSMODELSLIM_QAT_DEBUG_STORAGE"] == "0"
