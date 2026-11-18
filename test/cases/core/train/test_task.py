#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pytest

from msmodelslim.core.train.task import (
    CheckpointingConfig,
    CoreTrainTask,
    CoreTrainTaskConfig,
    NestedTrainConfig,
    QATConfig,
    TrainAlgorithmConfig,
    TrainingArgumentsConfig,
)
from msmodelslim.utils.exception import SchemaValidateError


def test_QATConfig_rejects_non_positive_model_max_length():
    with pytest.raises(SchemaValidateError):
        QATConfig(model_max_length=0)


def test_QATConfig_rejects_unknown_keys():
    with pytest.raises(SchemaValidateError):
        QATConfig(model_max_length=16, not_a_qat_field=1)


def test_QATConfig_rejects_alpha_min_ratio_when_above_one():
    with pytest.raises(SchemaValidateError):
        QATConfig(model_max_length=16, alpha_min_ratio=1.5)


def test_QATConfig_rejects_alpha_max_ratio_when_below_one():
    with pytest.raises(SchemaValidateError):
        QATConfig(model_max_length=16, alpha_max_ratio=0.5)


def test_TrainingArgumentsConfig_accepts_extra_fields_for_passthrough():
    cfg = TrainingArgumentsConfig(max_steps=600)
    assert cfg.max_steps == 600


def test_CheckpointingConfig_rejects_save_total_limit_when_above_two():
    with pytest.raises(SchemaValidateError):
        CheckpointingConfig(save_total_limit=3)


def test_TrainAlgorithmConfig_defaults_when_type_omitted():
    cfg = TrainAlgorithmConfig()
    assert cfg.type == "Unknown"
    assert cfg.include == ["*"]
    assert cfg.exclude == []


def test_TrainAlgorithmConfig_rejects_unknown_keys_when_type_omitted():
    with pytest.raises(SchemaValidateError):
        TrainAlgorithmConfig(not_an_algorithm_field=1)


def test_TrainAlgorithmConfig_dispatches_to_plugin_when_type_registered():
    cfg = TrainAlgorithmConfig.model_validate(
        {
            "type": "paretoq",
            "weight": {"dtype": "int4"},
            "act": {"dtype": "int8"},
        }
    )
    assert isinstance(cfg, TrainAlgorithmConfig)
    assert cfg.type == "paretoq"
    assert cfg.weight.w_bits == 4


def test_NestedTrainConfig_defaults_checkpointing_when_omitted():
    cfg = NestedTrainConfig.model_validate({"training_arguments": {"max_steps": 8}, "qat": {"model_max_length": 16}})
    assert cfg.training_arguments.max_steps == 8
    assert cfg.qat.model_max_length == 16
    assert cfg.checkpointing == CheckpointingConfig()


def test_NestedTrainConfig_keeps_custom_checkpointing_when_provided():
    cfg = NestedTrainConfig.model_validate(
        {
            "training_arguments": {},
            "qat": {"model_max_length": 16},
            "checkpointing": {"save_eval_interval": 5, "eval_on_start": True},
        }
    )
    assert cfg.checkpointing.save_eval_interval == 5
    assert cfg.checkpointing.eval_on_start is True
    assert cfg.checkpointing.save_total_limit == 2
    assert cfg.checkpointing.load_best_model_at_end is True


def test_NestedTrainConfig_raises_when_qat_missing():
    with pytest.raises(SchemaValidateError):
        NestedTrainConfig.model_validate({"training_arguments": {}})


def test_NestedTrainConfig_rejects_unknown_keys():
    with pytest.raises(SchemaValidateError):
        NestedTrainConfig.model_validate(
            {
                "training_arguments": {},
                "qat": {"model_max_length": 16},
                "not_a_train_field": 1,
            }
        )


def test_CoreTrainTaskConfig_assembles_when_fields_valid(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    cfg = CoreTrainTaskConfig.model_validate(
        {
            "algorithm": [
                {
                    "type": "paretoq",
                    "weight": {"dtype": "int4"},
                    "act": {"dtype": "int8"},
                }
            ],
            "dataset": {"train_path": str(data)},
            "train": {"training_arguments": {}, "qat": {"model_max_length": 128}},
        }
    )
    assert cfg.train.qat.model_max_length == 128
    assert len(cfg.algorithm) == 1


def test_CoreTrainTask_defaults_checkpoint_fields_when_constructed(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    cfg = CoreTrainTaskConfig.model_validate(
        {
            "algorithm": [
                {
                    "type": "paretoq",
                    "weight": {"dtype": "int4"},
                    "act": {"dtype": "int8"},
                }
            ],
            "dataset": {"train_path": str(data)},
            "train": {"training_arguments": {}, "qat": {"model_max_length": 16}},
        }
    )
    task = CoreTrainTask(config=cfg)
    assert task.best_checkpoint is None
    assert task.latest_checkpoint is None
    assert task.qparam_names == []
