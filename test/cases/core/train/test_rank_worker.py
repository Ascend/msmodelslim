#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import types
from unittest.mock import MagicMock

import pytest

from msmodelslim.core.train.rank_worker import RankWorker
from msmodelslim.core.train.task import CoreTrainTaskConfig
from msmodelslim.utils.exception import SchemaValidateError


def _minimal_task(tmp_path) -> CoreTrainTaskConfig:
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    return CoreTrainTaskConfig.model_validate(
        {
            "algorithm": [{"type": "paretoq"}],
            "dataset": {"path": str(data)},
            "train": {"hyperparameters": {"model_max_length": 16}},
        }
    )


def _worker(tmp_path, hooks=None) -> RankWorker:
    return RankWorker(
        _minimal_task(tmp_path),
        MagicMock(),
        tmp_path / "out",
        hooks=[] if hooks is None else hooks,
    )


def _stub_logger(monkeypatch) -> MagicMock:
    fake_logger = MagicMock()
    monkeypatch.setattr("msmodelslim.core.train.rank_worker.get_logger", lambda: fake_logger)
    return fake_logger


def _install_fake_module(monkeypatch, module_name: str, **attrs):
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


def _assert_stage_logged(fake_logger: MagicMock, stage: str) -> None:
    fake_logger.exception.assert_called()
    assert fake_logger.exception.call_args.args[1] == stage


def test_RankWorker_stores_save_path_when_constructed(tmp_path):
    assert _worker(tmp_path).save_path == tmp_path / "out"


def test_RankWorker_stores_model_max_length_when_constructed(tmp_path):
    assert _worker(tmp_path).model_max_length == 16


def test_RankWorker_stores_hooks_when_constructed(tmp_path):
    hooks = [MagicMock()]
    assert _worker(tmp_path, hooks=hooks).hooks == hooks


def test_RankWorker_assert_launcher_env_raises_when_world_size_invalid(tmp_path, monkeypatch):
    worker = _worker(tmp_path)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "0")
    with pytest.raises(SchemaValidateError, match="Invalid ranks"):
        worker._assert_launcher_env()


def test_RankWorker_assert_launcher_env_raises_when_rank_not_int(tmp_path, monkeypatch):
    worker = _worker(tmp_path)
    monkeypatch.setenv("LOCAL_RANK", "x")
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(SchemaValidateError, match="Invalid distributed env"):
        worker._assert_launcher_env()


def test_RankWorker_assert_launcher_env_passes_when_single_process(tmp_path, monkeypatch):
    worker = _worker(tmp_path)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    worker._assert_launcher_env()


def test_RankWorker_load_tokenizer_logs_stage_when_adapter_fails(tmp_path, monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    worker = _worker(tmp_path)
    worker.model_adapter.load_tokenizer_for_qat.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        worker._load_tokenizer()
    _assert_stage_logged(fake_logger, "load_tokenizer")


def test_RankWorker_load_model_logs_stage_when_adapter_fails(tmp_path, monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    worker = _worker(tmp_path)
    worker.model_adapter.load_model_for_qat.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        worker._load_model()
    _assert_stage_logged(fake_logger, "load_model")


def test_RankWorker_build_datasets_logs_stage_when_builder_fails(tmp_path, monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    worker = _worker(tmp_path)
    worker.tokenizer = MagicMock()
    worker.dist_info = {"world_size": 1}
    _install_fake_module(
        monkeypatch,
        "msmodelslim.core.train.dataset_build",
        build_qat_tokenized_datasets=MagicMock(side_effect=RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError, match="boom"):
        worker._build_datasets()
    _assert_stage_logged(fake_logger, "build_datasets")


def test_RankWorker_pre_train_logs_stage_when_hook_fails(tmp_path, monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    hook = MagicMock()
    hook.pre_train.side_effect = RuntimeError("boom")
    worker = _worker(tmp_path, hooks=[hook])
    worker.model = MagicMock()
    with pytest.raises(RuntimeError, match="boom"):
        worker._invoke_pre_train()
    _assert_stage_logged(fake_logger, "pre_train")


def test_RankWorker_train_logs_stage_when_trainer_build_fails(tmp_path, monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    worker = _worker(tmp_path, hooks=[MagicMock()])
    worker.model = MagicMock()
    worker.tokenizer = MagicMock()
    worker.train_dataset = []
    worker.eval_dataset = None
    _install_fake_module(
        monkeypatch,
        "msmodelslim.core.train.callback",
        QATCallback=MagicMock(),
    )
    _install_fake_module(
        monkeypatch,
        "msmodelslim.core.train.trainer",
        DEFAULT_QPARAM_NAMES=("weight_clip_val",),
        build_qat_trainer=MagicMock(side_effect=RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError, match="boom"):
        worker._train()
    _assert_stage_logged(fake_logger, "train")


def test_RankWorker_invoke_pre_train_raises_when_hooks_empty(tmp_path):
    worker = _worker(tmp_path, hooks=[])
    worker.model = MagicMock()
    with pytest.raises(SchemaValidateError, match="empty pre_train hooks"):
        worker._invoke_pre_train()
