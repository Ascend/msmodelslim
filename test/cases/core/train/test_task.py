#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from unittest.mock import patch

import pytest

from msmodelslim.core.train.task import (
    CheckpointingConfig,
    CoreTrainTask,
    CoreTrainTaskConfig,
    DatasetConfig,
    TrainHyperparametersConfig,
)
from msmodelslim.utils.exception import SchemaValidateError


def test_DatasetConfig_keeps_path_when_file_readable(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"x"}\n', encoding="utf-8")
    cfg = DatasetConfig(path=str(data), ratio_eval_samples=0.1)
    assert cfg.path.endswith("train.jsonl")
    assert cfg.ratio_eval_samples == 0.1


def test_DatasetConfig_raises_when_path_unreadable(tmp_path):
    with patch(
        "msmodelslim.core.train.task.convert_to_readable_file",
        side_effect=SchemaValidateError("unreadable", action="fix"),
    ):
        with pytest.raises(SchemaValidateError, match="unreadable"):
            DatasetConfig(path=str(tmp_path / "missing.jsonl"))


def test_DatasetConfig_rejects_ratio_eval_samples_when_out_of_range(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SchemaValidateError):
        DatasetConfig(path=str(data), ratio_eval_samples=1.0)


def test_TrainHyperparametersConfig_rejects_non_positive_model_max_length():
    with pytest.raises(SchemaValidateError):
        TrainHyperparametersConfig(model_max_length=0)


def test_CheckpointingConfig_rejects_save_total_limit_when_above_two():
    with pytest.raises(SchemaValidateError):
        CheckpointingConfig(save_total_limit=3)


def test_CoreTrainTaskConfig_assembles_when_fields_valid(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    cfg = CoreTrainTaskConfig.model_validate(
        {
            "algorithm": [{"type": "paretoq", "w_bits": 4}],
            "dataset": {"path": str(data)},
            "train": {"hyperparameters": {"model_max_length": 128}},
        }
    )
    assert cfg.train.hyperparameters.model_max_length == 128
    assert len(cfg.algorithm) == 1


def test_CoreTrainTask_defaults_checkpoint_fields_when_constructed(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    cfg = CoreTrainTaskConfig.model_validate(
        {
            "algorithm": [{"type": "paretoq"}],
            "dataset": {"path": str(data)},
            "train": {"hyperparameters": {"model_max_length": 16}},
        }
    )
    task = CoreTrainTask(config=cfg)
    assert task.best_checkpoint is None
    assert task.latest_checkpoint is None
    assert task.qparam_names == []
