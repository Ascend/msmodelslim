#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from msmodelslim.app.train.qat.application import (
    QATApplication,
    _select_checkpoint,
    get_plugin,
)
from msmodelslim.app.train.qat.task import QATTask, QATTaskConfig
from msmodelslim.app.train.proxy.task import TrainTask
from msmodelslim.core.const import DeviceType
from msmodelslim.format.checkpoint_format.constants import STRUCTURE_FILE
from msmodelslim.utils.exception import SchemaValidateError


def _write_ckpt(root: Path, step: int, *, with_structure: bool = True, eval_loss=None) -> Path:
    ckpt = root / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    if with_structure:
        (ckpt / STRUCTURE_FILE).write_text("{}", encoding="utf-8")
    if eval_loss is not None:
        state = {"log_history": [{"loss": 1.0}, {"eval_loss": eval_loss}]}
        (ckpt / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")
    return ckpt


def test_select_checkpoint_returns_none_when_no_candidates(tmp_path):
    assert _select_checkpoint(tmp_path) is None


def test_select_checkpoint_picks_lowest_eval_loss_when_available(tmp_path):
    _write_ckpt(tmp_path, 100, eval_loss=0.5)
    best = _write_ckpt(tmp_path, 200, eval_loss=0.2)
    _write_ckpt(tmp_path, 300, eval_loss=0.4)
    assert _select_checkpoint(tmp_path) == best


def test_select_checkpoint_prefers_larger_step_when_eval_loss_tied(tmp_path):
    _write_ckpt(tmp_path, 10, eval_loss=0.3)
    best = _write_ckpt(tmp_path, 20, eval_loss=0.3)
    assert _select_checkpoint(tmp_path) == best


def test_select_checkpoint_falls_back_to_latest_step_when_no_eval_loss(tmp_path):
    _write_ckpt(tmp_path, 1)
    best = _write_ckpt(tmp_path, 9)
    assert _select_checkpoint(tmp_path) == best


def test_select_checkpoint_skips_dirs_without_structure_when_scanning(tmp_path):
    _write_ckpt(tmp_path, 1, with_structure=False, eval_loss=0.1)
    best = _write_ckpt(tmp_path, 2, eval_loss=0.9)
    assert _select_checkpoint(tmp_path) == best


def test_QATApplication_do_deploy_raises_when_task_missing(tmp_path):
    app = QATApplication(
        config=MagicMock(),
        model_factory=MagicMock(),
        train_launcher=MagicMock(),
    )
    with pytest.raises(SchemaValidateError, match="QATTask missing"):
        app.do_deploy(save_path=tmp_path, device=DeviceType.CPU, device_indices=None)


def test_QATApplication_do_deploy_skips_when_no_checkpoint(tmp_path):
    app = QATApplication(
        config=MagicMock(),
        model_factory=MagicMock(),
        train_launcher=MagicMock(),
    )
    app.qat_task = MagicMock()
    app.do_deploy(save_path=tmp_path, device=DeviceType.CPU, device_indices=None)
    app.qat_task.build_convert_config.assert_not_called()


def test_QATApplication_build_task_builds_qat_task_when_config_valid(tmp_path, monkeypatch):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"text":"a"}\n', encoding="utf-8")
    cfg = QATTaskConfig.model_validate(
        {
            "type": "qat",
            "algorithm": [{"type": "paretoq", "w_bits": 4}],
            "dataset": {"path": str(dataset)},
            "train": {"hyperparameters": {"model_max_length": 32}},
        }
    )
    monkeypatch.setattr(
        "msmodelslim.core.convert.router.IRRouter.default",
        MagicMock(return_value=MagicMock()),
    )
    app = QATApplication(
        config=cfg,
        model_factory=MagicMock(),
        train_launcher=MagicMock(),
    )
    task = app.build_task(TrainTask(config=cfg))
    assert isinstance(task, QATTask)
    assert len(task.hooks) == 1


def test_get_plugin_returns_config_and_application_pair():
    config_cls, app_cls = get_plugin()
    assert config_cls is QATTaskConfig
    assert app_cls is QATApplication
    assert callable(getattr(app_cls, "run", None))
