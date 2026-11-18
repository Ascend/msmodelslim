#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from msmodelslim.core.train.callback import QATCallback
from msmodelslim.utils.exception import SchemaValidateError


def _stub_logger(monkeypatch) -> MagicMock:
    fake_logger = MagicMock()
    monkeypatch.setattr("msmodelslim.core.train.callback.get_logger", lambda: fake_logger)
    return fake_logger


def _fq_module(w_bits: int = 8) -> nn.Module:
    module = nn.Module()
    module.w_bits = w_bits
    module.weight_clip_val = nn.Parameter(torch.zeros(2))
    return module


def _model_with(*name_module_pairs) -> nn.Module:
    model = nn.Module()
    for name, module in name_module_pairs:
        model.add_module(name, module)
    return model


def _state(global_step=0, world_zero=True) -> SimpleNamespace:
    return SimpleNamespace(global_step=global_step, is_world_process_zero=world_zero)


def _train_begin(callback, model, global_step=0, world_zero=True):
    return callback.on_train_begin(SimpleNamespace(), _state(global_step, world_zero), SimpleNamespace(), model=model)


def _step_begin(callback, model, global_step, world_zero=True):
    return callback.on_step_begin(SimpleNamespace(), _state(global_step, world_zero), SimpleNamespace(), model=model)


def test_QATCallback_init_raises_when_fake_quant_after_n_steps_negative():
    with pytest.raises(SchemaValidateError, match="must be >= 0"):
        QATCallback(fake_quant_after_n_steps=-1)


def test_QATCallback_init_stores_none_when_fake_quant_after_n_steps_is_zero_or_none():
    assert QATCallback(fake_quant_after_n_steps=0).fake_quant_after_n_steps is None
    assert QATCallback(fake_quant_after_n_steps=None).fake_quant_after_n_steps is None
    assert QATCallback().fake_quant_after_n_steps is None


def test_QATCallback_init_stores_int_state_when_fake_quant_after_n_steps_positive():
    assert QATCallback(fake_quant_after_n_steps=5).fake_quant_after_n_steps == 5


def test_QATCallback_on_train_begin_counts_only_duck_typed_modules_when_mixed(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    callback = QATCallback()
    fq = _fq_module(w_bits=8)
    plain = nn.Linear(2, 2)
    partial = nn.Module()
    partial.w_bits = 4
    clipped = nn.Module()
    clipped.weight_clip_val = torch.zeros(2)
    outer = nn.Module()
    outer.add_module("inner", fq)
    model = _model_with(("outer", outer), ("plain", plain), ("partial", partial), ("clipped", clipped))
    _train_begin(callback, model)
    fake_logger.info.assert_called_once()
    info_args = fake_logger.info.call_args.args
    assert info_args[1] == 1
    assert info_args[2] == [8]
    fake_logger.warning.assert_not_called()


def test_QATCallback_on_train_begin_returns_control_when_model_is_none(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    callback = QATCallback(fake_quant_after_n_steps=5)
    control = SimpleNamespace()
    result = callback.on_train_begin(SimpleNamespace(), _state(global_step=0), control, model=None)
    assert result is control
    fake_logger.info.assert_not_called()
    fake_logger.warning.assert_not_called()


def test_QATCallback_on_train_begin_warns_when_no_fake_quant_modules(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    callback = QATCallback()
    _train_begin(callback, nn.Module())
    fake_logger.warning.assert_called_once_with(
        "QATCallback: no fake-quant module found on the model; was algorithm inject skipped?"
    )


def test_QATCallback_on_train_begin_logs_summary_when_fake_quant_modules_present(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    callback = QATCallback()
    fq8 = _fq_module(w_bits=8)
    fq4 = _fq_module(w_bits=4)
    _train_begin(callback, _model_with(("fq8", fq8), ("fq4", fq4)))
    fake_logger.info.assert_called_once()
    info_args = fake_logger.info.call_args.args
    assert info_args[1] == 2
    assert info_args[2] == [4, 8]
    fake_logger.warning.assert_not_called()
    assert fq8.w_bits == 8


def test_QATCallback_on_train_begin_lifts_low_w_bits_when_global_step_below_threshold(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    callback = QATCallback(fake_quant_after_n_steps=5)
    fq8 = _fq_module(w_bits=8)
    fq16 = _fq_module(w_bits=16)
    _train_begin(callback, _model_with(("fq8", fq8), ("fq16", fq16)), global_step=0)
    assert fq8.w_bits == 16
    assert fq16.w_bits == 16
    disable_args = fake_logger.info.call_args.args
    assert disable_args[0].startswith("QATCallback: fake quant disabled")
    assert disable_args[1] == 1
    assert disable_args[2] == 5


def test_QATCallback_on_train_begin_skips_disabling_when_global_step_at_threshold(monkeypatch):
    _stub_logger(monkeypatch)
    callback = QATCallback(fake_quant_after_n_steps=2)
    fq8 = _fq_module(w_bits=8)
    _train_begin(callback, _model_with(("fq8", fq8)), global_step=2)
    assert fq8.w_bits == 8


def test_QATCallback_on_train_begin_stays_enabled_when_all_modules_have_high_w_bits(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    callback = QATCallback(fake_quant_after_n_steps=5)
    fq16 = _fq_module(w_bits=16)
    _train_begin(callback, _model_with(("fq16", fq16)), global_step=0)
    assert fq16.w_bits == 16
    summary_args, disable_args = (c.args for c in fake_logger.info.call_args_list)
    assert summary_args[1] == 1
    assert summary_args[2] == [16]
    assert disable_args[0].startswith("QATCallback: fake quant disabled")
    assert disable_args[1] == 0


def test_QATCallback_on_train_begin_skips_logging_when_not_world_process_zero(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    callback = QATCallback(fake_quant_after_n_steps=5)
    fq8 = _fq_module(w_bits=8)
    _train_begin(callback, _model_with(("fq8", fq8)), global_step=0, world_zero=False)
    fake_logger.info.assert_not_called()
    fake_logger.warning.assert_not_called()
    assert fq8.w_bits == 16


def test_QATCallback_on_train_begin_skips_warning_when_not_world_process_zero(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    callback = QATCallback()
    _train_begin(callback, nn.Module(), global_step=0, world_zero=False)
    fake_logger.info.assert_not_called()
    fake_logger.warning.assert_not_called()


def test_QATCallback_on_step_begin_returns_early_when_fake_quant_not_disabled():
    callback = QATCallback(fake_quant_after_n_steps=5)
    fq8 = _fq_module(w_bits=8)
    control = SimpleNamespace()
    result = callback.on_step_begin(SimpleNamespace(), _state(global_step=99), control, model=_model_with(("fq8", fq8)))
    assert result is control
    assert fq8.w_bits == 8


def test_QATCallback_on_step_begin_stays_enabled_when_fake_quant_after_n_steps_is_none(monkeypatch):
    _stub_logger(monkeypatch)
    callback = QATCallback(fake_quant_after_n_steps=None)
    fq8 = _fq_module(w_bits=8)
    model = _model_with(("fq8", fq8))
    _train_begin(callback, model, global_step=0)
    _step_begin(callback, model, global_step=0)
    assert fq8.w_bits == 8


def test_QATCallback_on_step_begin_stays_disabled_when_global_step_below_threshold(monkeypatch):
    _stub_logger(monkeypatch)
    callback = QATCallback(fake_quant_after_n_steps=5)
    fq8 = _fq_module(w_bits=8)
    model = _model_with(("fq8", fq8))
    _train_begin(callback, model, global_step=0)
    _step_begin(callback, model, global_step=4)
    assert fq8.w_bits == 16


def test_QATCallback_on_step_begin_keeps_disabled_when_model_is_none(monkeypatch):
    _stub_logger(monkeypatch)
    callback = QATCallback(fake_quant_after_n_steps=5)
    fq8 = _fq_module(w_bits=8)
    model = _model_with(("fq8", fq8))
    _train_begin(callback, model, global_step=0)
    _step_begin(callback, None, global_step=5)
    assert fq8.w_bits == 16


def test_QATCallback_on_step_begin_restores_w_bits_when_global_step_reaches_threshold(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    callback = QATCallback(fake_quant_after_n_steps=5)
    fq8 = _fq_module(w_bits=8)
    fq16 = _fq_module(w_bits=16)
    model = _model_with(("fq8", fq8), ("fq16", fq16))
    _train_begin(callback, model, global_step=0)
    _step_begin(callback, model, global_step=5)
    assert fq8.w_bits == 8
    assert fq16.w_bits == 16
    enable_args = fake_logger.info.call_args.args
    assert enable_args[0].startswith("QATCallback: fake quant enabled")
    assert enable_args[1] == 1
    assert enable_args[2] == 5


def test_QATCallback_on_step_begin_restores_without_logging_when_not_world_process_zero(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    callback = QATCallback(fake_quant_after_n_steps=5)
    fq8 = _fq_module(w_bits=8)
    model = _model_with(("fq8", fq8))
    _train_begin(callback, model, global_step=0, world_zero=False)
    _step_begin(callback, model, global_step=5, world_zero=False)
    assert fq8.w_bits == 8
    fake_logger.info.assert_not_called()


def test_QATCallback_on_step_end_logs_device_memory_when_global_step_is_one(monkeypatch):
    memory_mock = MagicMock()
    monkeypatch.setattr("msmodelslim.core.train.callback.log_device_memory", memory_mock)
    callback = QATCallback()
    control = SimpleNamespace()
    result = callback.on_step_end(SimpleNamespace(), _state(global_step=1), control)
    memory_mock.assert_called_once_with("after_first_optimizer_step")
    assert result is control


def test_QATCallback_on_step_end_skips_memory_log_when_global_step_is_not_one(monkeypatch):
    memory_mock = MagicMock()
    monkeypatch.setattr("msmodelslim.core.train.callback.log_device_memory", memory_mock)
    callback = QATCallback()
    for state in (_state(global_step=0), _state(global_step=2), _state(global_step=None), SimpleNamespace()):
        callback.on_step_end(SimpleNamespace(), state, SimpleNamespace())
    memory_mock.assert_not_called()
