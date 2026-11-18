#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from unittest.mock import patch

import pytest

from msmodelslim.core.train.train_dataset_builder import TrainDatasetConfig
from msmodelslim.utils.exception import SchemaValidateError


def test_TrainDatasetConfig_keeps_train_path_when_file_readable(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"x"}\n', encoding="utf-8")
    cfg = TrainDatasetConfig(train_path=str(data), ratio_eval_samples=0.1)
    assert cfg.train_path.endswith("train.jsonl")
    assert cfg.ratio_eval_samples == 0.1
    assert cfg.eval_path is None


def test_TrainDatasetConfig_keeps_eval_path_when_file_readable(tmp_path):
    train = tmp_path / "train.jsonl"
    eval_file = tmp_path / "eval.jsonl"
    train.write_text('{"text":"x"}\n', encoding="utf-8")
    eval_file.write_text('{"text":"y"}\n', encoding="utf-8")
    cfg = TrainDatasetConfig(train_path=str(train), eval_path=str(eval_file))
    assert cfg.eval_path.endswith("eval.jsonl")


def test_TrainDatasetConfig_raises_when_train_path_unreadable(tmp_path):
    with patch(
        "msmodelslim.core.train.train_dataset_builder.convert_to_readable_file",
        side_effect=SchemaValidateError("unreadable", action="fix"),
    ):
        with pytest.raises(SchemaValidateError, match="unreadable"):
            TrainDatasetConfig(train_path=str(tmp_path / "missing.jsonl"))


def test_TrainDatasetConfig_rejects_ratio_eval_samples_when_out_of_range(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SchemaValidateError, match=r"ratio_eval_samples must be in \[0, 1\)"):
        TrainDatasetConfig(train_path=str(data), ratio_eval_samples=1.0)
    with pytest.raises(SchemaValidateError, match=r"ratio_eval_samples must be in \[0, 1\)"):
        TrainDatasetConfig(train_path=str(data), ratio_eval_samples=-0.1)
