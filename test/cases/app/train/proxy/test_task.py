#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pytest

from msmodelslim.app.train.proxy.task import TrainTask, TrainTaskConfig
from msmodelslim.app.train.qat.application import get_plugin as get_qat_plugin
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.plugin import register_plugin

# 测试环境显式注册；生产路径由 config.ini entry_point 加载。
register_plugin(get_qat_plugin)


def _qat_payload(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    return {
        "type": "qat",
        "algorithm": [{"type": "paretoq", "w_bits": 4}],
        "dataset": {"path": str(data)},
        "train": {"hyperparameters": {"model_max_length": 16}},
    }


def test_TrainTaskConfig_dispatches_to_qat_when_fields_complete(tmp_path):
    cfg = TrainTaskConfig.model_validate(_qat_payload(tmp_path))
    assert cfg.type == "qat"
    assert cfg.algorithm[0].type == "paretoq"


def test_TrainTask_stores_config_when_constructed(tmp_path):
    cfg = TrainTaskConfig.model_validate(_qat_payload(tmp_path))
    task = TrainTask(cfg)
    assert task.config is cfg
    assert task.config.type == "qat"


def test_TrainTaskConfig_rejects_qat_when_required_fields_missing():
    with pytest.raises(SchemaValidateError):
        TrainTaskConfig.model_validate({"type": "qat"})


def test_TrainTaskConfig_rejects_unknown_extra_field_when_forbidden(tmp_path):
    payload = _qat_payload(tmp_path)
    payload["unexpected"] = 1
    with pytest.raises(SchemaValidateError):
        TrainTaskConfig.model_validate(payload)
