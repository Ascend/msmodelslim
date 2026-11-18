#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import io
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, call

from msmodelslim.core.train.progress import QATProgressCallback

_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"


class _FloatRaises:
    def __float__(self):
        raise TypeError("cannot convert to float")


class _LenRaises:
    def __len__(self):
        raise TypeError("dataloader has no len")


def _stub_logger(monkeypatch) -> MagicMock:
    fake_logger = MagicMock()
    monkeypatch.setattr("msmodelslim.core.train.progress.get_logger", lambda: fake_logger)
    return fake_logger


def _stub_memory(monkeypatch, device_name="npu", allocated=12.34, reserved=20.5):
    monkeypatch.setattr(
        "msmodelslim.core.train.progress.get_device_memory",
        lambda *args, **kwargs: (device_name, allocated, reserved),
    )


def _patch_stderr_tty(monkeypatch) -> io.StringIO:
    buffer = io.StringIO()

    class _TtyStream:
        def write(self, text):
            return buffer.write(text)

        def flush(self):
            return None

        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stderr", _TtyStream())
    return buffer


def _args(disable_tqdm=False):
    return SimpleNamespace(disable_tqdm=disable_tqdm)


def _state(global_step=0, max_steps=100, is_world_process_zero=True):
    return SimpleNamespace(
        global_step=global_step,
        max_steps=max_steps,
        is_world_process_zero=is_world_process_zero,
    )


def _callback_with_bar(monkeypatch, global_step=0, max_steps=100) -> QATProgressCallback:
    _patch_stderr_tty(monkeypatch)
    callback = QATProgressCallback()
    callback.on_train_begin(_args(), _state(global_step=global_step, max_steps=max_steps), SimpleNamespace())
    return callback


# ========== 构造 / on_train_begin ==========


def test_QATProgressCallback_initializes_state_when_constructed():
    callback = QATProgressCallback()
    assert callback.training_bar is None
    assert callback.prediction_bar is None


def test_QATProgressCallback_on_train_begin_creates_bar_when_conditions_met(monkeypatch):
    _patch_stderr_tty(monkeypatch)
    control = SimpleNamespace()
    callback = QATProgressCallback()
    result = callback.on_train_begin(_args(), _state(global_step=None, max_steps=100), control)
    assert result is control
    assert callback.training_bar is not None
    assert callback.training_bar.desc == "QAT"
    assert callback.training_bar.total == 100
    assert callback.training_bar.bar_format == _BAR_FORMAT


def test_QATProgressCallback_on_train_begin_uses_none_total_when_max_steps_non_positive(monkeypatch):
    _patch_stderr_tty(monkeypatch)
    callback = QATProgressCallback()
    callback.on_train_begin(_args(), _state(global_step=0, max_steps=0), SimpleNamespace())
    assert callback.training_bar.total is None


def test_QATProgressCallback_on_train_begin_skips_bar_when_disable_tqdm_true():
    control = SimpleNamespace()
    callback = QATProgressCallback()
    result = callback.on_train_begin(_args(disable_tqdm=True), _state(global_step=5, max_steps=100), control)
    assert result is control
    assert callback.training_bar is None


def test_QATProgressCallback_on_train_begin_skips_bar_when_not_world_process_zero():
    callback = QATProgressCallback()
    callback.on_train_begin(_args(), _state(global_step=0, is_world_process_zero=False), SimpleNamespace())
    assert callback.training_bar is None


def test_QATProgressCallback_on_train_begin_skips_bar_when_tqdm_missing(monkeypatch):
    monkeypatch.setattr("msmodelslim.core.train.progress.tqdm", None)
    callback = QATProgressCallback()
    callback.on_train_begin(_args(), _state(global_step=0), SimpleNamespace())
    assert callback.training_bar is None


# ========== on_step_end ==========


def test_QATProgressCallback_on_step_end_returns_early_when_no_bar():
    control = SimpleNamespace()
    callback = QATProgressCallback()
    result = callback.on_step_end(_args(), _state(global_step=3), control)
    assert result is control
    assert callback.training_bar is None


def test_QATProgressCallback_on_step_end_updates_bar_when_step_advances(monkeypatch):
    callback = _callback_with_bar(monkeypatch, global_step=0, max_steps=10)
    result = callback.on_step_end(_args(), _state(global_step=3), SimpleNamespace())
    assert callback.training_bar.n == 3
    assert result is not None


def test_QATProgressCallback_on_step_end_skips_update_when_step_not_advanced(monkeypatch):
    callback = _callback_with_bar(monkeypatch, global_step=0, max_steps=10)
    callback.on_step_end(_args(), _state(global_step=3), SimpleNamespace())
    callback.on_step_end(_args(), _state(global_step=3), SimpleNamespace())
    assert callback.training_bar.n == 3
    callback.on_step_end(_args(), _state(global_step=1), SimpleNamespace())
    assert callback.training_bar.n == 3


# ========== on_log：postfix（bar 存在） ==========


def test_QATProgressCallback_on_log_sets_postfix_when_bar_present(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    _stub_memory(monkeypatch, device_name="npu", allocated=1.5, reserved=2.5)
    callback = _callback_with_bar(monkeypatch, global_step=20, max_steps=100)
    result = callback.on_log(_args(), _state(global_step=20), SimpleNamespace(), logs={"loss": 0.5})
    assert result is not None
    assert callback.training_bar.postfix == "loss=0.5000"
    assert fake_logger.info.call_args_list == [call("%s", "QAT train[step=20]: mem=1.5/2.5GiB(npu)")]


def test_QATProgressCallback_on_log_sets_full_postfix_when_train_payload_complete(monkeypatch):
    _stub_logger(monkeypatch)
    callback = _callback_with_bar(monkeypatch, global_step=7, max_steps=100)
    callback.on_log(
        _args(),
        _state(global_step=7),
        SimpleNamespace(),
        logs={
            "loss": 1.23456,
            "lr_weight": 5e-4,
            "lr_qparam": 2.5e-5,
            "clip_val_mean": 1.25,
            "tokens_per_sec": 1234.0,
        },
    )
    assert callback.training_bar.postfix == ("loss=1.2346  lr_w=5.00e-04  lr_q=2.50e-05  alpha=1.25  tok/s=1234")


def test_QATProgressCallback_on_log_sets_partial_postfix_when_train_payload_partial(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    callback = _callback_with_bar(monkeypatch, global_step=7, max_steps=10)
    callback.on_log(_args(), _state(global_step=7), SimpleNamespace(), logs={"loss": 2.0, "lr_weight": 1e-3})
    assert callback.training_bar.postfix == "loss=2.0000  lr_w=1.00e-03"
    fake_logger.info.assert_not_called()


def test_QATProgressCallback_on_log_sets_eval_postfix_when_payload_is_eval(monkeypatch):
    _stub_logger(monkeypatch)
    _stub_memory(monkeypatch, device_name="cpu", allocated=None, reserved=None)
    callback = _callback_with_bar(monkeypatch, global_step=7, max_steps=10)
    callback.on_log(_args(), _state(global_step=7), SimpleNamespace(), logs={"eval_loss": 0.5, "clip_val_mean": 2.0})
    assert callback.training_bar.postfix == "eval_loss=0.5000  alpha=2"


def test_QATProgressCallback_on_log_skips_postfix_when_values_not_numeric(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    callback = _callback_with_bar(monkeypatch, global_step=7, max_steps=10)
    callback.on_log(
        _args(),
        _state(global_step=7),
        SimpleNamespace(),
        logs={"loss": _FloatRaises(), "tokens_per_sec": "abc"},
    )
    assert callback.training_bar.postfix is None
    fake_logger.info.assert_not_called()


def test_QATProgressCallback_on_log_skips_postfix_when_payload_empty_and_bar_present(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    callback = _callback_with_bar(monkeypatch, global_step=0, max_steps=10)
    callback.on_log(_args(), _state(global_step=0), SimpleNamespace(), logs={})
    assert callback.training_bar.postfix is None
    fake_logger.info.assert_not_called()


# ========== on_log：logger（bar 缺失 / detail 行） ==========


def test_QATProgressCallback_on_log_writes_to_logger_when_bar_absent(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    callback = QATProgressCallback()
    control = SimpleNamespace()
    result = callback.on_log(_args(), _state(global_step=7), control, logs={"loss": 0.5})
    assert result is control
    assert fake_logger.info.call_args_list == [call("QAT %s/%s %s", 7, 100, "loss=0.5000")]


def test_QATProgressCallback_on_log_writes_detail_line_when_due_and_bar_absent(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    _stub_memory(monkeypatch, device_name="npu", allocated=1.5, reserved=2.5)
    callback = QATProgressCallback()
    callback.on_log(_args(), _state(global_step=20), SimpleNamespace(), logs={"loss": 0.5})
    assert fake_logger.info.call_args_list == [
        call("QAT %s/%s %s", 20, 100, "loss=0.5000"),
        call("%s", "QAT train[step=20]: mem=1.5/2.5GiB(npu)"),
    ]


def test_QATProgressCallback_on_log_writes_train_detail_when_step_hits_interval(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    _stub_memory(monkeypatch, device_name="npu", allocated=12.34, reserved=20.5)
    callback = QATProgressCallback()
    callback.on_log(
        _args(),
        _state(global_step=20),
        SimpleNamespace(),
        logs={"grad_norm": 1.23456, "clip_val_min": 0.1, "clip_val_max": 5.0, "clip_val_n": 3.0},
    )
    assert fake_logger.info.call_args_list == [
        call("%s", "QAT train[step=20]: grad_norm=1.235 clip_min=0.1 clip_max=5 clip_n=3 mem=12.3/20.5GiB(npu)")
    ]


def test_QATProgressCallback_on_log_omits_mem_when_allocated_memory_none(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    _stub_memory(monkeypatch, device_name="cpu", allocated=None, reserved=None)
    callback = QATProgressCallback()
    callback.on_log(_args(), _state(global_step=20), SimpleNamespace(), logs={"grad_norm": 1.0})
    assert fake_logger.info.call_args_list == [call("%s", "QAT train[step=20]: grad_norm=1")]


def test_QATProgressCallback_on_log_omits_reserved_when_reserved_memory_none(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    _stub_memory(monkeypatch, device_name="npu", allocated=8.0, reserved=None)
    callback = QATProgressCallback()
    callback.on_log(_args(), _state(global_step=20), SimpleNamespace(), logs={"grad_norm": 1.0})
    assert fake_logger.info.call_args_list == [call("%s", "QAT train[step=20]: grad_norm=1 mem=8.0GiB(npu)")]


def test_QATProgressCallback_on_log_writes_eval_detail_when_payload_is_eval(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    _stub_memory(monkeypatch, device_name="cpu", allocated=None, reserved=None)
    callback = QATProgressCallback()
    callback.on_log(
        _args(),
        _state(global_step=7),
        SimpleNamespace(),
        logs={"eval_loss": 0.5, "clip_val_min": 0.1, "clip_val_max": 2.0},
    )
    assert fake_logger.info.call_args_list == [
        call("QAT %s/%s %s", 7, 100, "eval_loss=0.5000"),
        call("%s", "QAT eval[step=7]: eval_loss=0.5000 clip_min=0.1 clip_max=2"),
    ]


def test_QATProgressCallback_on_log_skips_detail_when_step_misses_interval(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    _stub_memory(monkeypatch, device_name="npu", allocated=1.5, reserved=2.5)
    callback = QATProgressCallback()
    callback.on_log(_args(), _state(global_step=7), SimpleNamespace(), logs={"grad_norm": 1.0})
    fake_logger.info.assert_not_called()


def test_QATProgressCallback_on_log_writes_prefix_only_detail_when_payload_sparse(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    _stub_memory(monkeypatch, device_name="cpu", allocated=None, reserved=None)
    callback = QATProgressCallback()
    callback.on_log(_args(), _state(global_step=20), SimpleNamespace(), logs={})
    assert fake_logger.info.call_args_list == [call("%s", "QAT train[step=20]: ")]


def test_QATProgressCallback_on_log_skips_logger_when_not_world_process_zero(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    callback = QATProgressCallback()
    callback.on_log(_args(), _state(global_step=20, is_world_process_zero=False), SimpleNamespace(), logs={"loss": 0.5})
    fake_logger.info.assert_not_called()


# ========== on_prediction_step / on_evaluate / on_predict / on_train_end ==========


def test_QATProgressCallback_on_prediction_step_returns_early_when_bar_unusable():
    control = SimpleNamespace()
    callback = QATProgressCallback()
    result = callback.on_prediction_step(_args(disable_tqdm=True), _state(), control, eval_dataloader=range(3))
    assert result is control
    assert callback.prediction_bar is None


def test_QATProgressCallback_on_prediction_step_skips_creation_when_dataloader_none():
    control = SimpleNamespace()
    callback = QATProgressCallback()
    result = callback.on_prediction_step(_args(), _state(), control, eval_dataloader=None)
    assert result is control
    assert callback.prediction_bar is None


def test_QATProgressCallback_on_prediction_step_skips_creation_when_len_raises_type_error():
    callback = QATProgressCallback()
    callback.on_prediction_step(_args(), _state(), SimpleNamespace(), eval_dataloader=_LenRaises())
    assert callback.prediction_bar is None


def test_QATProgressCallback_on_prediction_step_skips_creation_when_dataloader_empty():
    callback = QATProgressCallback()
    callback.on_prediction_step(_args(), _state(), SimpleNamespace(), eval_dataloader=[])
    assert callback.prediction_bar is None


def test_QATProgressCallback_on_prediction_step_creates_and_updates_bar_when_dataloader_valid(monkeypatch):
    _patch_stderr_tty(monkeypatch)
    control = SimpleNamespace()
    callback = QATProgressCallback()
    result = callback.on_prediction_step(_args(), _state(), control, eval_dataloader=range(3))
    assert result is control
    eval_bar = callback.prediction_bar
    assert eval_bar is not None
    assert eval_bar.desc == "QAT eval"
    assert eval_bar.total == 3
    assert eval_bar.n == 1
    callback.on_prediction_step(_args(), _state(), SimpleNamespace(), eval_dataloader=range(3))
    assert callback.prediction_bar is eval_bar
    assert eval_bar.n == 2


def test_QATProgressCallback_on_evaluate_closes_prediction_bar_when_present(monkeypatch):
    _patch_stderr_tty(monkeypatch)
    callback = QATProgressCallback()
    callback.on_prediction_step(_args(), _state(), SimpleNamespace(), eval_dataloader=range(3))
    eval_bar = callback.prediction_bar
    control = SimpleNamespace()
    result = callback.on_evaluate(_args(), _state(), control)
    assert result is control
    assert callback.prediction_bar is None
    assert eval_bar.disable is True
    callback.on_evaluate(_args(), _state(), SimpleNamespace())


def test_QATProgressCallback_on_predict_closes_prediction_bar_when_present(monkeypatch):
    _patch_stderr_tty(monkeypatch)
    callback = QATProgressCallback()
    callback.on_prediction_step(_args(), _state(), SimpleNamespace(), eval_dataloader=range(3))
    eval_bar = callback.prediction_bar
    control = SimpleNamespace()
    result = callback.on_predict(_args(), _state(), control)
    assert result is control
    assert callback.prediction_bar is None
    assert eval_bar.disable is True
    callback.on_predict(_args(), _state(), SimpleNamespace())


def test_QATProgressCallback_on_train_end_closes_bars_when_present(monkeypatch):
    _patch_stderr_tty(monkeypatch)
    callback = QATProgressCallback()
    callback.on_train_begin(_args(), _state(global_step=0, max_steps=10), SimpleNamespace())
    training_bar = callback.training_bar
    callback.on_prediction_step(_args(), _state(), SimpleNamespace(), eval_dataloader=range(3))
    prediction_bar = callback.prediction_bar
    control = SimpleNamespace()
    result = callback.on_train_end(_args(), _state(), control)
    assert result is control
    assert callback.training_bar is None
    assert callback.prediction_bar is None
    assert training_bar.disable is True
    assert prediction_bar.disable is True
    callback.on_train_end(_args(), _state(), SimpleNamespace())
    assert callback.training_bar is None
