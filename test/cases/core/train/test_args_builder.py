#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# pylint: disable=no-name-in-module

import dataclasses
from unittest.mock import MagicMock

import pytest
from transformers import TrainingArguments

from msmodelslim.core.train.args_builder import TrainingArgumentsBuilder, TrainingPlan
from msmodelslim.core.train.task import NestedTrainConfig
from msmodelslim.utils.exception import SchemaValidateError


def _train_cfg(training_arguments=None, qat=None, checkpointing=None) -> NestedTrainConfig:
    train = {
        "training_arguments": training_arguments or {},
        "qat": qat or {"model_max_length": 16},
    }
    if checkpointing is not None:
        train["checkpointing"] = checkpointing
    return NestedTrainConfig.model_validate(train)


def _build(cfg: NestedTrainConfig, tmp_path, has_eval_dataset: bool = True) -> TrainingPlan:
    return TrainingArgumentsBuilder().build(cfg, output_dir=str(tmp_path), has_eval_dataset=has_eval_dataset)


def _stub_logger(monkeypatch) -> MagicMock:
    fake_logger = MagicMock()
    monkeypatch.setattr("msmodelslim.core.train.args_builder.get_logger", lambda: fake_logger)
    return fake_logger


def _warning_messages(fake_logger: MagicMock) -> list:
    return [call.args[0] for call in fake_logger.warning.call_args_list]


def _info_messages(fake_logger: MagicMock) -> list:
    return [call.args[0] for call in fake_logger.info.call_args_list]


def _dropped_hyper_keys(fake_logger: MagicMock) -> set:
    return {
        call.args[1]
        for call in fake_logger.warning.call_args_list
        if call.args[0].startswith("train.training_arguments.")
    }


def _has_native_warmup_ratio() -> bool:
    """当前 transformers 是否原生提供 TrainingArguments.warmup_ratio 字段。"""
    return "warmup_ratio" in {f.name for f in dataclasses.fields(TrainingArguments)}


def test_TrainingArgumentsBuilder_build_returns_default_plan_when_cfg_minimal(tmp_path, monkeypatch):
    fake_logger = _stub_logger(monkeypatch)

    plan = _build(_train_cfg(), tmp_path, has_eval_dataset=True)

    assert isinstance(plan, TrainingPlan)
    training_args = plan.training_args
    assert isinstance(training_args, TrainingArguments)
    assert training_args.output_dir == str(tmp_path)
    # 通用超参默认值直通 transformers.TrainingArguments。
    assert training_args.num_train_epochs == 1.0
    assert training_args.learning_rate == 1e-5
    assert training_args.per_device_train_batch_size == 1
    assert training_args.per_device_eval_batch_size == 1
    assert training_args.gradient_accumulation_steps == 8
    assert training_args.logging_steps == 1
    assert training_args.bf16 is False
    # QAT 附加超参保留在 TrainingPlan 上。
    assert plan.qparam_lr_scale == 10.0
    assert plan.ce_chunk_size == 512
    assert plan.log_qparams is False
    assert plan.alpha_min_ratio == 0.5
    assert plan.alpha_max_ratio == 2.0
    if _has_native_warmup_ratio():
        # 老 transformers 原生支持 warmup_ratio：直通 training_args，plan 不挂起、不告警。
        assert training_args.warmup_ratio == 0.05
        assert plan.warmup_ratio is None
        assert _dropped_hyper_keys(fake_logger) == set()
    else:
        # 当前 transformers 无 warmup_ratio 字段：值挂起在 plan 上（由 trainer 延迟换算），
        # 不写入 training_args，直通时告警丢弃。
        assert plan.warmup_ratio == 0.05
        assert training_args.warmup_steps == 0
        assert _dropped_hyper_keys(fake_logger) == {"warmup_ratio"}


def test_TrainingArgumentsBuilder_build_maps_custom_overrides_when_configured(tmp_path, monkeypatch):
    _stub_logger(monkeypatch)
    cfg = _train_cfg(
        training_arguments={
            "num_train_epochs": 2.5,
            "learning_rate": 3e-4,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 2,
        },
        qat={
            "model_max_length": 32,
            "qparam_lr_scale": 100.0,
            "ce_chunk_size": 128,
            "log_qparams": True,
            "alpha_min_ratio": 0.25,
            "alpha_max_ratio": 3.0,
        },
        checkpointing={
            "save_eval_interval": 50,
            "eval_on_start": True,
            "save_total_limit": 2,
            "load_best_model_at_end": True,
        },
    )

    plan = _build(cfg, tmp_path, has_eval_dataset=True)

    training_args = plan.training_args
    assert training_args.num_train_epochs == 2.5
    assert training_args.learning_rate == 3e-4
    assert training_args.per_device_train_batch_size == 4
    assert training_args.gradient_accumulation_steps == 2
    assert plan.qparam_lr_scale == 100.0
    assert plan.ce_chunk_size == 128
    assert plan.log_qparams is True
    assert plan.alpha_min_ratio == 0.25
    assert plan.alpha_max_ratio == 3.0
    # checkpointing 段映射到 save/eval 策略。
    assert training_args.do_eval is True
    assert training_args.save_strategy.value == "steps"
    assert training_args.save_steps == 50
    assert training_args.eval_strategy.value == "steps"
    assert training_args.eval_steps == 50
    assert training_args.eval_on_start is True
    assert training_args.save_total_limit == 2
    assert training_args.load_best_model_at_end is True
    assert training_args.metric_for_best_model == "eval_loss"
    assert training_args.greater_is_better is False


def test_TrainingArgumentsBuilder_build_passes_extra_fields_when_supported_by_transformers(tmp_path):
    cfg = _train_cfg(training_arguments={"max_steps": 7, "save_only_model": True})

    plan = _build(cfg, tmp_path)

    assert plan.training_args.max_steps == 7
    assert plan.training_args.save_only_model is True


def test_TrainingArgumentsBuilder_build_drops_legacy_qat_key_when_misplaced_in_hyper(tmp_path, monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    cfg = _train_cfg(training_arguments={"qparam_lr_scale": 100.0})

    plan = _build(cfg, tmp_path)

    expected_dropped = {"qparam_lr_scale"}
    if not _has_native_warmup_ratio():
        expected_dropped.add("warmup_ratio")
    assert _dropped_hyper_keys(fake_logger) == expected_dropped
    assert any(
        "train.training_arguments" in message and "is not a TrainingArguments field" in message
        for message in _warning_messages(fake_logger)
    )
    # 旧位置 QAT 键被丢弃，回落到 train.qat 段默认值，也不进入 TrainingArguments。
    assert plan.qparam_lr_scale == 10.0
    assert "qparam_lr_scale" not in {f.name for f in dataclasses.fields(TrainingArguments)}


def test_TrainingArgumentsBuilder_build_disables_gradient_checkpointing_when_accelerate_fsdp_enabled(
    tmp_path, monkeypatch
):
    fake_logger = _stub_logger(monkeypatch)
    monkeypatch.setenv("ACCELERATE_USE_FSDP", "true")
    monkeypatch.delenv("FSDP_ACTIVATION_CHECKPOINTING", raising=False)
    cfg = _train_cfg(training_arguments={"gradient_checkpointing": True})

    plan = _build(cfg, tmp_path)

    assert plan.training_args.gradient_checkpointing is False
    assert any(
        message.startswith("Dropping TrainingArguments.gradient_checkpointing")
        for message in _warning_messages(fake_logger)
    )


def test_TrainingArgumentsBuilder_build_disables_gradient_checkpointing_when_fsdp_activation_checkpointing_set(
    tmp_path, monkeypatch
):
    _stub_logger(monkeypatch)
    monkeypatch.setenv("FSDP_ACTIVATION_CHECKPOINTING", "true")
    monkeypatch.delenv("ACCELERATE_USE_FSDP", raising=False)

    plan = _build(_train_cfg(), tmp_path)

    training_args = plan.training_args
    assert training_args.gradient_checkpointing is False
    assert training_args.gradient_checkpointing_kwargs is None


def test_TrainingArgumentsBuilder_build_enables_gradient_checkpointing_when_fsdp_env_absent(tmp_path, monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    monkeypatch.delenv("ACCELERATE_USE_FSDP", raising=False)
    monkeypatch.delenv("FSDP_ACTIVATION_CHECKPOINTING", raising=False)

    plan = _build(_train_cfg(), tmp_path)

    training_args = plan.training_args
    assert training_args.gradient_checkpointing is True
    assert training_args.gradient_checkpointing_kwargs == {"use_reentrant": False}
    assert any(
        message.startswith("Enabled TrainingArguments.gradient_checkpointing")
        for message in _info_messages(fake_logger)
    )


def test_TrainingArgumentsBuilder_build_disables_eval_and_best_model_when_eval_dataset_absent(tmp_path, monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    cfg = _train_cfg(checkpointing={"load_best_model_at_end": True})

    plan = _build(cfg, tmp_path, has_eval_dataset=False)

    training_args = plan.training_args
    assert training_args.do_eval is False
    assert training_args.eval_strategy.value == "no"
    assert training_args.load_best_model_at_end is False
    assert any(
        message.startswith("checkpointing.load_best_model_at_end=true requires eval")
        for message in _warning_messages(fake_logger)
    )


def test_TrainingArgumentsBuilder_build_enables_eval_when_eval_dataset_present(tmp_path):
    plan = _build(_train_cfg(), tmp_path, has_eval_dataset=True)

    training_args = plan.training_args
    assert training_args.do_eval is True
    assert training_args.eval_strategy.value == "steps"
    assert training_args.eval_steps == 200
    assert training_args.save_strategy.value == "steps"
    assert training_args.save_steps == 200
    assert training_args.load_best_model_at_end is True
    assert training_args.metric_for_best_model == "eval_loss"
    assert training_args.greater_is_better is False


def test_TrainingArgumentsBuilder_build_raises_SchemaValidateError_when_hyper_rejected_by_transformers(tmp_path):
    cfg = _train_cfg(training_arguments={"lr_scheduler_type": "bogus"})

    with pytest.raises(SchemaValidateError, match="train.training_arguments"):
        _build(cfg, tmp_path)


def test_TrainingPlan_is_frozen_when_attribute_assigned(tmp_path):
    plan = _build(_train_cfg(), tmp_path)

    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.qparam_lr_scale = 1.0


def test_TrainingPlan_declares_all_fields_when_inspected():
    field_names = {f.name for f in dataclasses.fields(TrainingPlan)}

    assert field_names == {
        "training_args",
        "qparam_lr_scale",
        "ce_chunk_size",
        "warmup_ratio",
        "log_qparams",
        "alpha_min_ratio",
        "alpha_max_ratio",
    }
