#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import contextlib
import os
import socket
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from msmodelslim.core.const import DeviceType
from msmodelslim.core.train.task import CoreTrainTaskConfig
from msmodelslim.core.train.train_launcher import TrainLauncher
from msmodelslim.utils.config import msmodelslim_config
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError

_WORKER_LOG_LEVEL_ENV = "MSMODELSLIM_WORKER_LOG_LEVEL"


def _minimal_task(tmp_path) -> CoreTrainTaskConfig:
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    return CoreTrainTaskConfig.model_validate(
        {
            "algorithm": [{"type": "paretoq", "weight": {"dtype": "int4"}, "act": {"dtype": "int8"}}],
            "dataset": {"train_path": str(data)},
            "train": {"training_arguments": {}, "qat": {"model_max_length": 16}},
        }
    )


def _recording_launcher(record, name, invoke):
    def _launcher(function, args=None, num_processes=None, **kwargs):
        record.append(
            {
                "name": name,
                "function": function,
                "args": args,
                "num_processes": num_processes,
                "env": dict(os.environ),
                "kwargs": kwargs,
            }
        )
        if invoke:
            # 模拟 accelerate 在 patch_environment 上下文内触发子进程入口。
            function(*args)

    return _launcher


def _install_fake_accelerate(monkeypatch, invoke=False):
    """替换 accelerate 边界：记录 launcher 调用与环境快照，patch_environment 真实生效。"""
    record = []

    @contextlib.contextmanager
    def _fake_patch_environment(**kwargs):
        old = {key: os.environ.get(key) for key in kwargs}
        os.environ.update(kwargs)
        try:
            yield
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    module = types.ModuleType("accelerate")
    module.debug_launcher = _recording_launcher(record, "debug", invoke)
    module.notebook_launcher = _recording_launcher(record, "notebook", invoke)
    utils_module = types.ModuleType("accelerate.utils")
    utils_module.patch_environment = _fake_patch_environment
    module.utils = utils_module
    monkeypatch.setitem(sys.modules, "accelerate", module)
    monkeypatch.setitem(sys.modules, "accelerate.utils", utils_module)
    return record


def _free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def test_TrainLauncher_launch_dispatches_worker_args_when_single_process_npu(tmp_path, monkeypatch):
    fake_logger = MagicMock()
    monkeypatch.setattr("msmodelslim.core.train.train_launcher.get_logger", lambda: fake_logger)
    record = _install_fake_accelerate(monkeypatch)
    cfg = _minimal_task(tmp_path)
    adapter = MagicMock()
    hooks = [MagicMock()]
    save_path = tmp_path / "out"
    TrainLauncher().launch(cfg, adapter, save_path, hooks, device=DeviceType.NPU, device_indices=None)
    assert len(record) == 1
    entry = record[0]
    assert entry["name"] == "notebook"
    assert entry["args"] == (cfg, adapter, save_path, list(hooks))
    assert entry["num_processes"] == 1
    assert entry["kwargs"]["mixed_precision"] == "no"
    assert int(entry["kwargs"]["use_port"]) > 0
    assert fake_logger.info.called


def test_TrainLauncher_launch_resolves_num_processes_when_device_indices_given(tmp_path, monkeypatch):
    record = _install_fake_accelerate(monkeypatch)
    TrainLauncher().launch(
        _minimal_task(tmp_path),
        MagicMock(),
        tmp_path / "out",
        [],
        device=DeviceType.NPU,
        device_indices=[0, 1, 3],
    )
    assert record[0]["num_processes"] == 3


def test_TrainLauncher_launch_raises_when_device_indices_empty(tmp_path, monkeypatch):
    _install_fake_accelerate(monkeypatch)
    with pytest.raises(SchemaValidateError, match="at least one"):
        TrainLauncher().launch(
            _minimal_task(tmp_path),
            MagicMock(),
            tmp_path / "out",
            [],
            device=DeviceType.NPU,
            device_indices=[],
        )


def test_TrainLauncher_launch_uses_debug_launcher_when_cpu_multi_process(tmp_path, monkeypatch):
    record = _install_fake_accelerate(monkeypatch)
    cfg = _minimal_task(tmp_path)
    adapter = MagicMock()
    save_path = tmp_path / "out"
    hooks = []
    TrainLauncher().launch(cfg, adapter, save_path, hooks, device=DeviceType.CPU, device_indices=[0, 1])
    assert [item["name"] for item in record] == ["debug"]
    assert record[0]["args"] == (cfg, adapter, save_path, list(hooks))
    assert record[0]["num_processes"] == 2


def test_TrainLauncher_launch_sets_fsdp_env_when_multi_process_npu(tmp_path, monkeypatch):
    record = _install_fake_accelerate(monkeypatch)
    TrainLauncher().launch(
        _minimal_task(tmp_path),
        MagicMock(),
        tmp_path / "out",
        [],
        device=DeviceType.NPU,
        device_indices=[0, 1],
    )
    env = record[0]["env"]
    assert env["ACCELERATE_USE_FSDP"] == "true"
    assert env["FSDP_VERSION"] == "2"
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "0,1"
    assert env["MSMODELSLIM_QAT_DEBUG_STORAGE"] == "0"


def test_TrainLauncher_launch_omits_fsdp_env_when_single_process_npu(tmp_path, monkeypatch):
    record = _install_fake_accelerate(monkeypatch)
    TrainLauncher().launch(_minimal_task(tmp_path), MagicMock(), tmp_path / "out", [], device=DeviceType.NPU)
    env = record[0]["env"]
    assert "ACCELERATE_USE_FSDP" not in env
    assert "ASCEND_RT_VISIBLE_DEVICES" not in env
    assert _WORKER_LOG_LEVEL_ENV in env


def test_TrainLauncher_launch_sets_use_cpu_env_when_cpu_device(tmp_path, monkeypatch):
    record = _install_fake_accelerate(monkeypatch)
    TrainLauncher().launch(_minimal_task(tmp_path), MagicMock(), tmp_path / "out", [], device=DeviceType.CPU)
    env = record[0]["env"]
    assert env["ACCELERATE_USE_CPU"] == "true"
    assert "ASCEND_RT_VISIBLE_DEVICES" not in env


def test_TrainLauncher_launch_uses_requested_master_port_when_free(tmp_path, monkeypatch):
    record = _install_fake_accelerate(monkeypatch)
    port = _free_port()
    monkeypatch.setenv("MASTER_PORT", port)
    TrainLauncher().launch(_minimal_task(tmp_path), MagicMock(), tmp_path / "out", [], device=DeviceType.NPU)
    assert record[0]["kwargs"]["use_port"] == port


def test_TrainLauncher_launch_picks_free_port_when_master_port_busy(tmp_path, monkeypatch):
    fake_logger = MagicMock()
    monkeypatch.setattr("msmodelslim.core.train.train_launcher.get_logger", lambda: fake_logger)
    record = _install_fake_accelerate(monkeypatch)
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    try:
        busy_port = str(holder.getsockname()[1])
        monkeypatch.setenv("MASTER_PORT", busy_port)
        TrainLauncher().launch(_minimal_task(tmp_path), MagicMock(), tmp_path / "out", [], device=DeviceType.NPU)
    finally:
        holder.close()
    use_port = record[0]["kwargs"]["use_port"]
    assert use_port != busy_port
    assert int(use_port) > 0
    fake_logger.warning.assert_called_once()


def test_TrainLauncher_launch_picks_free_port_when_master_port_invalid(tmp_path, monkeypatch):
    record = _install_fake_accelerate(monkeypatch)
    monkeypatch.setenv("MASTER_PORT", "not-a-port")
    TrainLauncher().launch(_minimal_task(tmp_path), MagicMock(), tmp_path / "out", [], device=DeviceType.NPU)
    assert int(record[0]["kwargs"]["use_port"]) > 0


def test_TrainLauncher_launch_child_entry_falls_back_to_config_log_level_when_root_unknown(tmp_path, monkeypatch):
    set_level_mock = MagicMock()
    monkeypatch.setattr("msmodelslim.core.train.train_launcher.rank_worker_entry", MagicMock())
    monkeypatch.setattr("msmodelslim.core.train.train_launcher.set_logger_level", set_level_mock)
    monkeypatch.setattr("msmodelslim.core.train.train_launcher.get_logger", MagicMock)
    monkeypatch.setattr(
        "msmodelslim.core.train.train_launcher.get_root_logger",
        lambda: SimpleNamespace(getEffectiveLevel=lambda: 12345),
    )
    _install_fake_accelerate(monkeypatch, invoke=True)
    TrainLauncher().launch(_minimal_task(tmp_path), MagicMock(), tmp_path / "out", [], device=DeviceType.NPU)
    set_level_mock.assert_called_once_with(msmodelslim_config.env_vars.log_level)


def test_TrainLauncher_launch_raises_when_accelerate_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "accelerate", None)
    with pytest.raises(UnsupportedError, match="accelerate is required"):
        TrainLauncher().launch(_minimal_task(tmp_path), MagicMock(), tmp_path / "out", [], device=DeviceType.NPU)
