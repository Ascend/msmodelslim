#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import torch
from safetensors.torch import load_file
from torch import nn

# pylint: disable=no-name-in-module
from msmodelslim.core.train.checkpoint import QATCheckpointSaver
from msmodelslim.core.quantization_format.checkpoint_format.constants import STRUCTURE_FILE, WEIGHT_FILE


def _stub_logger(monkeypatch) -> MagicMock:
    fake_logger = MagicMock()
    monkeypatch.setattr("msmodelslim.core.train.checkpoint.get_logger", lambda: fake_logger)
    return fake_logger


def _tiny_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))


def _saver(source_model_path=None) -> QATCheckpointSaver:
    return QATCheckpointSaver(
        qparam_names=["weight_clip_val"],
        algorithm_cfg=[{"type": "paretoq"}],
        source_model_path=source_model_path,
    )


def _args(should_save: bool) -> SimpleNamespace:
    return SimpleNamespace(should_save=should_save, process_index=0, world_size=1)


def _accelerator(model) -> MagicMock:
    accelerator = MagicMock()
    accelerator.unwrap_model = MagicMock(return_value=model)
    return accelerator


def _fsdp_accelerator(model, state, plugin) -> MagicMock:
    accelerator = MagicMock()
    accelerator.unwrap_model = MagicMock(return_value=model)
    accelerator.state.fsdp_plugin = plugin
    accelerator.get_state_dict = MagicMock(return_value=state)
    return accelerator


def _marked_state(model) -> dict:
    """构造与模型同名但带标记的收集结果：'0.weight' 置零，证明落盘取自收集值。"""
    state = dict(model.state_dict())
    state["0.weight"] = torch.zeros_like(state["0.weight"])
    return state


def test_QATCheckpointSaver_save_writes_accelerator_state_when_fsdp_plugin_missing(tmp_path):
    model = _tiny_model()
    state = _marked_state(model)
    accelerator = MagicMock()
    accelerator.state.fsdp_plugin = None
    accelerator.get_state_dict = MagicMock(return_value=state)
    accelerator.unwrap_model = MagicMock(return_value=model)
    output_dir = tmp_path / "out"
    saver = _saver()
    saver.save(
        model=model,
        accelerator=accelerator,
        args=_args(should_save=True),
        output_dir=str(output_dir),
        is_fsdp_enabled=True,
    )
    accelerator.get_state_dict.assert_called_once_with(model)
    assert (output_dir / STRUCTURE_FILE).is_file()
    tensors = load_file(str(output_dir / WEIGHT_FILE))
    assert torch.count_nonzero(tensors["0.weight"]).item() == 0


def test_QATCheckpointSaver_save_keeps_state_dict_type_when_already_full(tmp_path):
    model = _tiny_model()
    state = _marked_state(model)
    plugin = MagicMock()
    plugin.state_dict_type = "FULL_STATE_DICT"
    accelerator = _fsdp_accelerator(model, state, plugin)
    output_dir = tmp_path / "out"
    saver = _saver()
    saver.save(
        model=model,
        accelerator=accelerator,
        args=_args(should_save=True),
        output_dir=str(output_dir),
        is_fsdp_enabled=True,
    )
    plugin.set_state_dict_type.assert_not_called()
    accelerator.get_state_dict.assert_called_once_with(model)
    tensors = load_file(str(output_dir / WEIGHT_FILE))
    assert torch.count_nonzero(tensors["0.weight"]).item() == 0


def test_QATCheckpointSaver_save_logs_exception_when_type_switch_fails(tmp_path, monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    model = _tiny_model()
    state = _marked_state(model)
    plugin = MagicMock()
    plugin.state_dict_type = "SHARDED_STATE_DICT"
    plugin.set_state_dict_type = MagicMock(side_effect=RuntimeError("boom"))
    accelerator = _fsdp_accelerator(model, state, plugin)
    output_dir = tmp_path / "out"
    saver = _saver()
    saver.save(
        model=model,
        accelerator=accelerator,
        args=_args(should_save=True),
        output_dir=str(output_dir),
        is_fsdp_enabled=True,
    )
    plugin.set_state_dict_type.assert_called_once_with("FULL_STATE_DICT")
    accelerator.get_state_dict.assert_called_once_with(model)
    assert fake_logger.exception.call_count == 1
    message, current_type = fake_logger.exception.call_args.args
    assert "failed to switch FSDP state_dict_type" in message
    assert current_type == "SHARDED_STATE_DICT"
    tensors = load_file(str(output_dir / WEIGHT_FILE))
    assert torch.count_nonzero(tensors["0.weight"]).item() == 0


def test_QATCheckpointSaver_save_switches_and_restores_type_when_plugin_not_full(tmp_path):
    model = _tiny_model()
    state = _marked_state(model)
    plugin = MagicMock()
    plugin.state_dict_type = "SHARDED_STATE_DICT"
    accelerator = _fsdp_accelerator(model, state, plugin)
    output_dir = tmp_path / "out"
    saver = _saver()
    saver.save(
        model=model,
        accelerator=accelerator,
        args=_args(should_save=True),
        output_dir=str(output_dir),
        is_fsdp_enabled=True,
    )
    assert plugin.set_state_dict_type.call_args_list == [
        call("FULL_STATE_DICT"),
        call("SHARDED_STATE_DICT"),
    ]
    accelerator.get_state_dict.assert_called_once_with(model)
    tensors = load_file(str(output_dir / WEIGHT_FILE))
    assert torch.count_nonzero(tensors["0.weight"]).item() == 0


def test_QATCheckpointSaver_save_logs_exception_when_restore_fails(tmp_path, monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    model = _tiny_model()
    state = _marked_state(model)
    plugin = MagicMock()
    plugin.state_dict_type = "SHARDED_STATE_DICT"
    plugin.set_state_dict_type = MagicMock(side_effect=[None, RuntimeError("restore boom")])
    accelerator = _fsdp_accelerator(model, state, plugin)
    output_dir = tmp_path / "out"
    saver = _saver()
    saver.save(
        model=model,
        accelerator=accelerator,
        args=_args(should_save=True),
        output_dir=str(output_dir),
        is_fsdp_enabled=True,
    )
    assert plugin.set_state_dict_type.call_args_list == [
        call("FULL_STATE_DICT"),
        call("SHARDED_STATE_DICT"),
    ]
    assert fake_logger.exception.call_count == 1
    message, restored_type = fake_logger.exception.call_args.args
    assert "failed to restore FSDP state_dict_type" in message
    assert restored_type == "SHARDED_STATE_DICT"


def test_QATCheckpointSaver_stores_copied_config_when_constructed():
    qparam_names = ["weight_clip_val"]
    algorithm_cfg = [{"type": "paretoq"}]
    saver = QATCheckpointSaver(
        qparam_names=qparam_names,
        algorithm_cfg=algorithm_cfg,
        source_model_path="src/model",
    )
    assert saver.qparam_names == ["weight_clip_val"]
    qparam_names.append("extra")
    assert saver.qparam_names == ["weight_clip_val"]
    assert saver.algorithm_cfg == [{"type": "paretoq"}]
    algorithm_cfg[0]["type"] = "mutated"
    assert saver.algorithm_cfg == [{"type": "paretoq"}]
    assert saver.source_model_path == "src/model"


def test_QATCheckpointSaver_stores_empty_algorithm_when_cfg_missing():
    saver = QATCheckpointSaver(qparam_names=["weight_clip_val"])
    assert saver.algorithm_cfg == []
    assert saver.source_model_path is None


def test_QATCheckpointSaver_save_writes_checkpoint_when_not_fsdp_and_should_save(tmp_path):
    model = _tiny_model()
    output_dir = tmp_path / "out"
    saver = _saver()
    saver.save(
        model=model,
        accelerator=_accelerator(model),
        args=_args(should_save=True),
        output_dir=str(output_dir),
    )
    assert (output_dir / STRUCTURE_FILE).is_file()
    assert (output_dir / WEIGHT_FILE).is_file()

    payload = json.loads((output_dir / STRUCTURE_FILE).read_text(encoding="utf-8"))
    assert payload["qparam_names"] == ["weight_clip_val"]
    assert payload["algorithm"] == [{"type": "paretoq"}]
    assert payload["modules"] == {}
    assert payload["tensors"]["weight_file"] == WEIGHT_FILE

    tensors = load_file(str(output_dir / WEIGHT_FILE))
    expected = model.state_dict()
    assert set(tensors) == set(expected)
    for key, value in expected.items():
        assert torch.equal(tensors[key], value)


def test_QATCheckpointSaver_save_copies_hf_sidecars_when_source_model_path_set(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    model = _tiny_model()
    output_dir = tmp_path / "out"
    saver = _saver(source_model_path=str(source_dir))
    saver.save(
        model=model,
        accelerator=_accelerator(model),
        args=_args(should_save=True),
        output_dir=str(output_dir),
    )
    assert (output_dir / "tokenizer_config.json").is_file()
    assert (output_dir / STRUCTURE_FILE).is_file()
    assert (output_dir / WEIGHT_FILE).is_file()


def test_QATCheckpointSaver_save_skips_disk_write_when_should_save_false(tmp_path):
    model = _tiny_model()
    output_dir = tmp_path / "out"
    saver = _saver()
    saver.save(
        model=model,
        accelerator=_accelerator(model),
        args=_args(should_save=False),
        output_dir=str(output_dir),
    )
    assert output_dir.is_dir()
    assert not list(output_dir.iterdir())


def test_QATCheckpointSaver_save_calls_device_helpers_when_saving_checkpoint(tmp_path, monkeypatch):
    log_memory = MagicMock()
    empty_cache = MagicMock()
    monkeypatch.setattr("msmodelslim.core.train.checkpoint.log_device_memory", log_memory)
    monkeypatch.setattr("msmodelslim.core.train.checkpoint.empty_device_cache", empty_cache)
    model = _tiny_model()
    accelerator = _accelerator(model)
    saver = _saver()
    saver.save(
        model=model,
        accelerator=accelerator,
        args=_args(should_save=True),
        output_dir=str(tmp_path / "out"),
    )
    assert log_memory.call_args_list == [
        call("before_checkpoint_gather"),
        call("after_checkpoint_gather"),
        call("after_checkpoint_release"),
    ]
    assert empty_cache.call_count == 1
    accelerator.unwrap_model.assert_called_once_with(model, keep_torch_compile=False)
