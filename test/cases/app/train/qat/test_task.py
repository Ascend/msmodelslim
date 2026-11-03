#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from msmodelslim.app.train.qat.algorithm_train_hook import AlgorithmTrainHook
from msmodelslim.app.train.qat.task import QATTask, QATTaskConfig
from msmodelslim.core.convert.types import IRKind
from msmodelslim.format.checkpoint_format.constants import STRUCTURE_FILE
from msmodelslim.processor.paretoq.convert import (
    ParetoQToW4A4DynamicProcessor,
    ParetoQToW4A8DynamicProcessor,
)
from msmodelslim.utils.exception import SchemaValidateError


def _cfg(tmp_path: Path, **overrides):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"text":"a"}\n', encoding="utf-8")
    payload = {
        "type": "qat",
        "algorithm": [{"type": "paretoq", "w_bits": 4, "deploy_act": "a8"}],
        "dataset": {"path": str(dataset)},
        "train": {"hyperparameters": {"model_max_length": 64}},
    }
    payload.update(overrides)
    return QATTaskConfig.model_validate(payload)


def _stub_ir_router(monkeypatch):
    monkeypatch.setattr(
        "msmodelslim.core.convert.router.IRRouter.default",
        MagicMock(return_value=MagicMock()),
    )


def test_QATTask_build_assembles_hooks_when_paretoq(tmp_path, monkeypatch):
    _stub_ir_router(monkeypatch)
    task = QATTask.build_from(_cfg(tmp_path))
    assert len(task.hooks) == 1
    assert isinstance(task.hooks[0], AlgorithmTrainHook)
    assert task.config.algorithm[0].type == "paretoq"


def test_QATTask_register_convert_queries_algorithm_processors_when_built(tmp_path, monkeypatch):
    _stub_ir_router(monkeypatch)
    task = QATTask.build_from(_cfg(tmp_path))
    names = {type(p).__name__ for p in task.convert_processors}
    assert names == {
        ParetoQToW4A8DynamicProcessor.__name__,
        ParetoQToW4A4DynamicProcessor.__name__,
    }


def test_QATTask_register_convert_raises_when_plugin_not_qat_linear(tmp_path, monkeypatch):
    class _NotAlgo:
        pass

    monkeypatch.setattr(
        "msmodelslim.app.train.qat.task.load_plugin_component_class",
        lambda *_args, **_kwargs: _NotAlgo,
    )
    task = QATTask(_cfg(tmp_path))
    with pytest.raises(SchemaValidateError, match="QATLinearAlgorithm"):
        task.register_convert()


def test_QATTask_assemble_hooks_raises_when_multiple_algorithms(tmp_path):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    cfg = QATTaskConfig.model_validate(
        {
            "type": "qat",
            "algorithm": [
                {"type": "paretoq", "w_bits": 4},
                {"type": "paretoq", "w_bits": 3},
            ],
            "dataset": {"path": str(dataset)},
            "train": {"hyperparameters": {"model_max_length": 16}},
        }
    )
    task = QATTask(cfg)
    with pytest.raises(SchemaValidateError, match="exactly one algorithm"):
        task.assemble_hooks()


def test_QATTaskConfig_rejects_nested_spec_when_present():
    with pytest.raises(SchemaValidateError, match="spec"):
        QATTaskConfig.model_validate(
            {
                "type": "qat",
                "spec": {
                    "algorithm": [{"type": "paretoq"}],
                    "train": {"hyperparameters": {"model_max_length": 32}},
                },
            }
        )


def test_QATTask_build_convert_config_sets_dst_ir_when_structure_valid(tmp_path, monkeypatch):
    _stub_ir_router(monkeypatch)
    dataset = tmp_path / "train.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()
    (ckpt / STRUCTURE_FILE).write_text(
        '{"modules":{"model.layers.0.mlp.up_proj":{"w_bits":4,"deploy_act":"a8","weight_layerwise":false}}}',
        encoding="utf-8",
    )
    cfg = _cfg(tmp_path)
    task = QATTask.build_from(cfg)
    convert_cfg = task.build_convert_config(
        checkpoint_path=ckpt,
        deploy_save_path=tmp_path / "deploy",
        device=__import__("msmodelslim.core.const", fromlist=["DeviceType"]).DeviceType.CPU,
    )
    assert convert_cfg.dst_format == "ascendv1"
    assert len(convert_cfg.module_rules) == 1
    assert convert_cfg.module_rules[0].source_ir == IRKind.PARETOQ_FLOAT
    assert convert_cfg.convert_rules[0].target_ir == IRKind.W4A8_DYNAMIC


def test_QATTask_build_convert_config_raises_when_modules_empty(tmp_path, monkeypatch):
    _stub_ir_router(monkeypatch)
    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()
    (ckpt / STRUCTURE_FILE).write_text('{"modules":{}}', encoding="utf-8")
    task = QATTask.build_from(_cfg(tmp_path))
    with pytest.raises(SchemaValidateError, match="No ParetoQ modules"):
        task.build_convert_config(
            checkpoint_path=ckpt,
            deploy_save_path=tmp_path / "deploy",
        )
