#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# pylint: disable=no-name-in-module

import dataclasses
import types
from unittest.mock import MagicMock, call

import pytest
import torch
from torch import nn
from tokenizers import Tokenizer
from tokenizers import models as tokenizers_models
from tokenizers import pre_tokenizers as tokenizers_pre_tokenizers
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)
from transformers.trainer_callback import PrinterCallback, ProgressCallback

from msmodelslim.core.train.callback import QATCallback
from msmodelslim.core.train.loss import LossOnlyOutput
from msmodelslim.core.train.optimizer import RelativeAlphaAdamW
from msmodelslim.core.train.progress import QATProgressCallback
from msmodelslim.core.train.task import NestedTrainConfig
from msmodelslim.core.train.trainer import QATTrainer
from msmodelslim.core.quantization_format.checkpoint_format.constants import STRUCTURE_FILE, WEIGHT_FILE
from msmodelslim.utils.exception import SchemaValidateError

_SEQ = 16
_VOCAB = 32


def _has_native_warmup_ratio() -> bool:
    """当前 transformers 是否原生提供 TrainingArguments.warmup_ratio 字段。"""
    return "warmup_ratio" in {f.name for f in dataclasses.fields(TrainingArguments)}


def _hub_revision_supported() -> bool:
    return "hub_revision" in {f.name for f in dataclasses.fields(TrainingArguments)}


def _stub_logger(monkeypatch) -> MagicMock:
    fake_logger = MagicMock()
    monkeypatch.setattr("msmodelslim.core.train.trainer.get_logger", lambda: fake_logger)
    return fake_logger


def _tiny_gpt2() -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=_VOCAB,
        n_positions=_SEQ,
        n_embd=16,
        n_layer=1,
        n_head=2,
        use_cache=False,
        tie_word_embeddings=False,
        bos_token_id=0,
        eos_token_id=1,
    )
    return GPT2LMHeadModel(config)


def _rows(n=2, seq=_SEQ, vocab=_VOCAB):
    gen = torch.Generator().manual_seed(7)
    rows = []
    for _ in range(n):
        ids = torch.randint(1, vocab, (seq,), generator=gen).tolist()
        rows.append({"input_ids": ids, "labels": list(ids)})
    return rows


def _train_cfg(*, training_arguments=None, qat=None, checkpointing=None) -> NestedTrainConfig:
    hyper = {
        "max_steps": 2,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.01,
    }
    hyper.update(training_arguments or {})
    return NestedTrainConfig.model_validate(
        {
            "training_arguments": hyper,
            "qat": {"model_max_length": _SEQ, **(qat or {})},
            "checkpointing": {"load_best_model_at_end": False, **(checkpointing or {})},
        }
    )


def _trainer(
    tmp_path,
    *,
    qat=None,
    training_arguments=None,
    checkpointing=None,
    with_clip=False,
    tokenizer=None,
    callbacks=None,
) -> QATTrainer:
    model = _tiny_gpt2()
    if with_clip:
        model.register_parameter("weight_clip_val", nn.Parameter(torch.tensor([1.0, 2.0])))
    return QATTrainer(
        model=model,
        train_cfg=_train_cfg(qat=qat, training_arguments=training_arguments, checkpointing=checkpointing),
        output_dir=str(tmp_path),
        train_dataset=_rows(),
        qparam_names=("weight_clip_val",),
        tokenizer=tokenizer,
        callbacks=callbacks,
    )


def _gpt_inputs(batch=1):
    ids = torch.randint(2, _VOCAB, (batch, _SEQ))
    return {"input_ids": ids, "labels": ids.clone()}


def _make_tokenizer():
    inner = Tokenizer(tokenizers_models.WordLevel(vocab={"<pad>": 0, "a": 1, "b": 2, "c": 3}, unk_token="<pad>"))
    inner.pre_tokenizer = tokenizers_pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=inner)
    tokenizer.pad_token = "<pad>"
    return tokenizer


class _LogRecorder(TrainerCallback):
    """经公开回调钩子记录 log() 注入后的 payload。"""

    def __init__(self):
        self.payloads = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.payloads.append(dict(logs or {}))


def _patch_super_compute_loss(monkeypatch, marker="fallback"):
    calls = []

    def fake_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        calls.append({"return_outputs": return_outputs, "num_items_in_batch": num_items_in_batch})
        return (marker, return_outputs)

    monkeypatch.setattr("msmodelslim.core.train.trainer.Trainer.compute_loss", fake_compute_loss)
    return calls


# ========== __init__：入参校验 ==========


def test_QATTrainer_init_raises_when_model_is_none(tmp_path):
    with pytest.raises(SchemaValidateError, match="requires an injected model"):
        QATTrainer(
            model=None,
            train_cfg=_train_cfg(),
            output_dir=str(tmp_path),
            train_dataset=_rows(),
            qparam_names=("weight_clip_val",),
        )


def test_QATTrainer_init_raises_when_train_dataset_is_none(tmp_path):
    with pytest.raises(SchemaValidateError, match="requires a train_dataset"):
        QATTrainer(
            model=_tiny_gpt2(),
            train_cfg=_train_cfg(),
            output_dir=str(tmp_path),
            train_dataset=None,
            qparam_names=("weight_clip_val",),
        )


def test_QATTrainer_init_raises_when_output_dir_is_blank(tmp_path):
    with pytest.raises(SchemaValidateError, match="requires a non-empty output_dir"):
        QATTrainer(
            model=_tiny_gpt2(),
            train_cfg=_train_cfg(),
            output_dir="  ",
            train_dataset=_rows(),
            qparam_names=("weight_clip_val",),
        )


def test_QATTrainer_init_raises_when_qparam_names_is_empty(tmp_path):
    with pytest.raises(SchemaValidateError, match="qparam_names is empty"):
        QATTrainer(
            model=_tiny_gpt2(),
            train_cfg=_train_cfg(),
            output_dir=str(tmp_path),
            train_dataset=_rows(),
            qparam_names=(),
        )


# ========== __init__：tokenizer / processing_class 版本适配 ==========


def test_QATTrainer_init_maps_tokenizer_to_processing_class_when_supported(tmp_path):
    tokenizer = _make_tokenizer()
    trainer = _trainer(tmp_path, tokenizer=tokenizer)
    assert trainer.processing_class is tokenizer


def test_QATTrainer_init_keeps_processing_class_none_when_tokenizer_omitted(tmp_path):
    trainer = _trainer(tmp_path)
    assert trainer.processing_class is None


def test_QATTrainer_init_passes_tokenizer_kwarg_when_trainer_lacks_processing_class_param(tmp_path, monkeypatch):
    received = {}

    def legacy_init(
        self,
        model=None,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        callbacks=None,
        tokenizer=None,
    ):
        received.update(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
            tokenizer=tokenizer,
        )
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        handler_callbacks = list(callbacks or [])
        self.callback_handler = types.SimpleNamespace(
            callbacks=handler_callbacks,
            add_callback=handler_callbacks.append,
            pop_callback=lambda cb: next(
                (handler_callbacks.pop(i) for i, c in enumerate(handler_callbacks) if isinstance(c, cb)), None
            ),
        )
        self.get_train_dataloader = lambda: [None, None]

    monkeypatch.setattr("msmodelslim.core.train.trainer.Trainer.__init__", legacy_init)
    tokenizer = _make_tokenizer()
    trainer = _trainer(tmp_path, tokenizer=tokenizer)
    assert received["tokenizer"] is tokenizer
    assert "processing_class" not in received
    assert received["model"] is trainer.model


# ========== __init__：训练前计划日志与 warmup 换算 ==========


def test_QATTrainer_init_converts_warmup_ratio_when_max_steps_set(tmp_path, monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    trainer = _trainer(tmp_path, training_arguments={"max_steps": 4, "warmup_ratio": 0.25})
    if _has_native_warmup_ratio():
        # 老 transformers 原生透传 warmup_ratio，由 HF 建调度器时按比例换算。
        assert trainer.args.warmup_ratio == 0.25
        assert trainer.args.warmup_steps == 0
        assert not [c for c in fake_logger.info.call_args_list if "Converted warmup_ratio" in str(c.args[0])]
        return
    assert trainer.args.warmup_steps == 1
    converted = [c for c in fake_logger.info.call_args_list if "Converted warmup_ratio" in str(c.args[0])]
    assert converted == [call("Converted warmup_ratio=%s to warmup_steps=%s (max_steps=%s)", 0.25, 1, 4)]


def test_QATTrainer_init_logs_epoch_based_plan_when_max_steps_unset(tmp_path, monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    trainer = _trainer(tmp_path, training_arguments={"max_steps": -1, "num_train_epochs": 1.0})
    if _has_native_warmup_ratio():
        assert trainer.args.warmup_ratio == 0.05
        assert trainer.args.warmup_steps == 0
    else:
        assert trainer.args.warmup_steps == 1
    planned = [c for c in fake_logger.info.call_args_list if "QAT planned training steps" in str(c.args[0])]
    assert len(planned) == 1
    # (num_examples, world_size, micro_bs, accum, batches, steps_per_epoch,
    #  epochs, total_steps, warmup_steps, global_batch_size)
    assert planned[0].args[1:] == (2, 1, 1, 1, 2, 2, 1.0, 2, 1, 1)


def test_QATTrainer_init_keeps_explicit_warmup_steps_when_configured(tmp_path, monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    trainer = _trainer(tmp_path, training_arguments={"max_steps": 4, "warmup_steps": 2, "warmup_ratio": 0.5})
    assert trainer.args.warmup_steps == 2
    assert not [c for c in fake_logger.info.call_args_list if "Converted" in str(c.args[0])]


def test_QATTrainer_init_warns_when_dataloader_has_no_length(tmp_path, monkeypatch):
    fake_logger = _stub_logger(monkeypatch)

    class _BadLen:
        def __len__(self):
            raise TypeError("no length")

    trainer = QATTrainer(
        model=_tiny_gpt2(),
        train_cfg=_train_cfg(training_arguments={"group_by_length": False}),
        output_dir=str(tmp_path),
        train_dataset=_BadLen(),
        qparam_names=("weight_clip_val",),
    )
    fake_logger.warning.assert_called_once_with(
        "QAT planned schedule: train dataloader has no length; skip max_steps log."
    )
    assert trainer.args.warmup_steps == 0
    assert not [c for c in fake_logger.info.call_args_list if "QAT planned training steps" in str(c.args[0])]


# ========== log()：父类签名版本适配 ==========


def test_QATTrainer_log_forwards_start_time_when_super_log_accepts_it(tmp_path, monkeypatch):
    captured = []

    def fake_log(self, logs, start_time=None):
        captured.append((dict(logs), start_time))

    monkeypatch.setattr("msmodelslim.core.train.trainer.Trainer.log", fake_log)
    trainer = _trainer(tmp_path)
    trainer.log({"loss": 1.0})
    assert captured == [({"loss": 1.0}, None)]


def test_QATTrainer_log_omits_start_time_when_super_log_lacks_param(tmp_path, monkeypatch):
    captured = []

    def fake_log(self, logs):
        captured.append(dict(logs))

    monkeypatch.setattr("msmodelslim.core.train.trainer.Trainer.log", fake_log)
    trainer = _trainer(tmp_path)
    trainer.log({"loss": 1.0})
    assert captured == [{"loss": 1.0}]


# ========== log()：QAT 指标注入（经 on_log 回调断言 payload） ==========


def test_QATTrainer_log_injects_group_lrs_when_optimizer_created(tmp_path):
    recorder = _LogRecorder()
    trainer = _trainer(tmp_path, qat={"log_qparams": True}, with_clip=True, callbacks=[recorder])
    trainer.create_optimizer()
    trainer.log({})
    payload = recorder.payloads[-1]
    assert payload["lr_weight"] == pytest.approx(0.01)
    assert payload["lr_qparam"] == pytest.approx(0.1)


def test_QATTrainer_log_keeps_existing_lr_weight_when_already_present(tmp_path):
    recorder = _LogRecorder()
    trainer = _trainer(tmp_path, qat={"log_qparams": True}, with_clip=True, callbacks=[recorder])
    trainer.create_optimizer()
    trainer.log({"lr_weight": 9.9})
    payload = recorder.payloads[-1]
    assert payload["lr_weight"] == 9.9
    assert payload["lr_qparam"] == pytest.approx(0.1)


def test_QATTrainer_log_omits_lr_keys_when_optimizer_not_created(tmp_path):
    recorder = _LogRecorder()
    trainer = _trainer(tmp_path, callbacks=[recorder])
    trainer.log({"loss": 1.0})
    payload = recorder.payloads[-1]
    assert payload["loss"] == 1.0
    assert "lr_weight" not in payload
    assert "lr_qparam" not in payload


def test_QATTrainer_log_reports_tokens_per_sec_when_training_loss_computed(tmp_path):
    recorder = _LogRecorder()
    trainer = _trainer(tmp_path, callbacks=[recorder])
    trainer.log({})  # 建立吞吐窗口基准
    trainer.compute_loss(trainer.model, _gpt_inputs())
    trainer.log({"loss": 1.0})
    payload = recorder.payloads[-1]
    assert payload["loss"] == 1.0
    assert payload["tokens_per_sec"] > 0


def test_QATTrainer_log_skips_tokens_per_sec_when_eval_log(tmp_path):
    recorder = _LogRecorder()
    trainer = _trainer(tmp_path, callbacks=[recorder])
    trainer.log({})  # 建立吞吐窗口基准
    trainer.compute_loss(trainer.model, _gpt_inputs())
    trainer.log({"eval_loss": 1.0})
    payload = recorder.payloads[-1]
    assert payload["eval_loss"] == 1.0
    assert "tokens_per_sec" not in payload


def test_QATTrainer_log_injects_clip_stats_when_qparam_present(tmp_path):
    recorder = _LogRecorder()
    trainer = _trainer(tmp_path, with_clip=True, callbacks=[recorder])
    trainer.log({})
    payload = recorder.payloads[-1]
    assert payload["clip_val_n"] == 2.0
    assert payload["clip_val_mean"] == 1.5
    assert payload["clip_val_min"] == 1.0
    assert payload["clip_val_max"] == 2.0


def test_QATTrainer_log_omits_clip_keys_when_no_qparam_matched(tmp_path):
    recorder = _LogRecorder()
    trainer = _trainer(tmp_path, callbacks=[recorder])
    trainer.log({"loss": 1.0})
    payload = recorder.payloads[-1]
    assert "clip_val_n" not in payload
    assert "clip_val_mean" not in payload


# ========== compute_loss ==========


def test_QATTrainer_compute_loss_returns_loss_and_output_when_labels_present(tmp_path):
    trainer = _trainer(tmp_path)
    loss, outputs = trainer.compute_loss(trainer.model, _gpt_inputs(), return_outputs=True)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert isinstance(outputs, LossOnlyOutput)
    assert outputs.loss is loss
    assert outputs.logits is None


def test_QATTrainer_compute_loss_returns_loss_only_when_return_outputs_false(tmp_path):
    trainer = _trainer(tmp_path)
    result = trainer.compute_loss(trainer.model, _gpt_inputs())
    assert isinstance(result, torch.Tensor)
    assert result.dim() == 0
    assert torch.isfinite(result)


def test_QATTrainer_compute_loss_skips_token_count_when_model_is_eval(tmp_path):
    recorder = _LogRecorder()
    trainer = _trainer(tmp_path, callbacks=[recorder])
    trainer.log({})  # 建立吞吐窗口基准
    trainer.model.eval()
    trainer.compute_loss(trainer.model, _gpt_inputs())
    trainer.log({"loss": 1.0})
    assert recorder.payloads[-1]["tokens_per_sec"] == 0.0


def test_QATTrainer_compute_loss_falls_back_when_labels_missing(tmp_path, monkeypatch):
    calls = _patch_super_compute_loss(monkeypatch)
    trainer = _trainer(tmp_path)
    inputs = {"input_ids": torch.randint(2, _VOCAB, (1, _SEQ))}
    assert trainer.compute_loss(trainer.model, inputs, num_items_in_batch=7) == ("fallback", False)
    assert calls[0]["num_items_in_batch"] == 7
    assert trainer.compute_loss(trainer.model, inputs, return_outputs=True) == ("fallback", True)


def test_QATTrainer_compute_loss_falls_back_when_model_has_no_lm_head(tmp_path, monkeypatch):
    loss_logger = MagicMock()
    monkeypatch.setattr("msmodelslim.core.train.loss.get_logger", lambda: loss_logger)
    _patch_super_compute_loss(monkeypatch)
    trainer = _trainer(tmp_path)
    result = trainer.compute_loss(nn.Module(), {"labels": torch.zeros(2, 3, dtype=torch.long)})
    assert result == ("fallback", False)
    assert "no lm_head" in loss_logger.warning.call_args.args[0]


def test_QATTrainer_compute_loss_omits_num_items_when_super_lacks_param(tmp_path, monkeypatch):
    calls = []

    def fake_compute_loss(self, model, inputs, return_outputs=False):
        calls.append({"return_outputs": return_outputs})
        return "legacy"

    monkeypatch.setattr("msmodelslim.core.train.trainer.Trainer.compute_loss", fake_compute_loss)
    trainer = _trainer(tmp_path)
    result = trainer.compute_loss(nn.Module(), {"labels": torch.zeros(1, 2, dtype=torch.long)}, num_items_in_batch=5)
    assert result == "legacy"
    assert calls == [{"return_outputs": False}]


def test_QATTrainer_compute_loss_retries_unwrap_when_accelerator_rejects_kwarg(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path)
    real_model = trainer.model
    seen = []

    def unwrap_model(model, keep_torch_compile=None):
        seen.append(keep_torch_compile)
        if keep_torch_compile is not None:
            raise TypeError("keep_torch_compile unsupported")
        return real_model

    monkeypatch.setattr(trainer, "accelerator", types.SimpleNamespace(unwrap_model=unwrap_model))
    loss = trainer.compute_loss(real_model, _gpt_inputs())
    assert seen == [False, None]
    assert loss.dim() == 0
    assert torch.isfinite(loss)


# ========== prediction_step ==========


def test_QATTrainer_prediction_step_returns_finite_loss_when_context_manager_present(tmp_path):
    trainer = _trainer(tmp_path)
    loss, logits, labels = trainer.prediction_step(trainer.model, _gpt_inputs(), True)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert not loss.requires_grad
    assert logits is None
    assert labels is None


def test_QATTrainer_prediction_step_computes_loss_directly_when_context_manager_missing(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path)
    monkeypatch.setattr(trainer, "compute_loss_context_manager", None)
    trainer.model.eval()  # 关闭 dropout，保证两次前向结果可比
    inputs = _gpt_inputs()
    expected = trainer.compute_loss(trainer.model, inputs).item()
    loss, logits, labels = trainer.prediction_step(trainer.model, inputs, True)
    assert loss.item() == pytest.approx(expected)
    assert (logits, labels) == (None, None)


# ========== create_optimizer ==========


def test_QATTrainer_create_optimizer_uses_legacy_getter_when_model_kwarg_rejected(tmp_path, monkeypatch):
    def legacy_getter(args, model=None):
        if model is not None:
            raise TypeError("legacy getter takes no model")
        return (torch.optim.SGD, {"momentum": 0.9})

    monkeypatch.setattr("msmodelslim.core.train.trainer.Trainer.get_optimizer_cls_and_kwargs", legacy_getter)
    trainer = _trainer(tmp_path)
    trainer.create_optimizer()
    assert isinstance(trainer.optimizer, torch.optim.SGD)
    assert trainer.optimizer.param_groups[0]["momentum"] == 0.9


# ========== save_model（真 QATCheckpointSaver 落盘） ==========


def test_QATTrainer_save_model_writes_files_when_dir_omitted(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.save_model()
    assert (tmp_path / STRUCTURE_FILE).is_file()
    assert (tmp_path / WEIGHT_FILE).is_file()


def test_QATTrainer_save_model_writes_files_when_dir_given(tmp_path):
    trainer = _trainer(tmp_path)
    custom = tmp_path / "custom"
    trainer.save_model(str(custom))
    assert (custom / STRUCTURE_FILE).is_file()
    assert (custom / WEIGHT_FILE).is_file()


def test_QATTrainer_save_model_pushes_to_hub_when_requested(tmp_path, monkeypatch):
    # 父类 __init__ 在 push_to_hub=True 时会联网 init_hf_repo；此处为被测边界打桩。
    monkeypatch.setattr("msmodelslim.core.train.trainer.Trainer.init_hf_repo", lambda self: None)
    trainer = _trainer(tmp_path, training_arguments={"push_to_hub": True, "hub_revision": "main"})
    push = MagicMock()
    monkeypatch.setattr(trainer, "push_to_hub", push)
    trainer.save_model()
    expected_revision = "main" if _hub_revision_supported() else None
    push.assert_called_once_with(commit_message="Model save", revision=expected_revision)
    assert (tmp_path / WEIGHT_FILE).is_file()


def test_QATTrainer_save_model_skips_push_when_internal_call(tmp_path, monkeypatch):
    monkeypatch.setattr("msmodelslim.core.train.trainer.Trainer.init_hf_repo", lambda self: None)
    trainer = _trainer(tmp_path, training_arguments={"push_to_hub": True, "hub_revision": "main"})
    push = MagicMock()
    monkeypatch.setattr(trainer, "push_to_hub", push)
    trainer.save_model(_internal_call=True)
    push.assert_not_called()
    assert (tmp_path / WEIGHT_FILE).is_file()


# ========== evaluate ==========


def test_QATTrainer_evaluate_restarts_throughput_window_when_evaluation_raises(tmp_path, monkeypatch):
    def boom(self, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("msmodelslim.core.train.trainer.Trainer.evaluate", boom)
    recorder = _LogRecorder()
    trainer = _trainer(tmp_path, callbacks=[recorder])
    with pytest.raises(RuntimeError, match="boom"):
        trainer.evaluate()
    trainer.log({})
    # reset 已重置窗口基准：0 token / 有基准 -> 0.0（无基准时该键不存在）
    assert recorder.payloads[-1]["tokens_per_sec"] == 0.0


# ========== 微型端到端（真实 Trainer 机制 + 真 checkpoint saver） ==========


def test_QATTrainer_train_and_evaluate_when_tiny_gpt2_runs_full_loop(tmp_path):
    model = _tiny_gpt2()
    model.register_parameter("weight_clip_val", nn.Parameter(torch.ones(2)))
    trainer = QATTrainer(
        model=model,
        train_cfg=_train_cfg(qat={"log_qparams": True}, checkpointing={"save_eval_interval": 1}),
        output_dir=str(tmp_path),
        train_dataset=_rows(),
        eval_dataset=_rows(),
        tokenizer=None,
        qparam_names=("weight_clip_val",),
        algorithm_cfg=[{"type": "paretoq"}],
        callbacks=[QATCallback()],
        source_model_path=None,
    )
    assert trainer.optimizer is None
    assert trainer.qparam_names == ("weight_clip_val",)
    assert trainer.log_qparams is True
    if _has_native_warmup_ratio():
        assert trainer.args.warmup_ratio == 0.05
    else:
        assert trainer.args.warmup_steps == 1
    assert trainer.args.eval_strategy == "steps"
    handler_callbacks = trainer.callback_handler.callbacks
    assert not any(isinstance(cb, (ProgressCallback, PrinterCallback)) for cb in handler_callbacks)
    assert sum(isinstance(cb, QATProgressCallback) for cb in handler_callbacks) == 1

    result = trainer.train()
    assert result.metrics["train_loss"] > 0
    inner_optimizer = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
    assert isinstance(inner_optimizer, RelativeAlphaAdamW)
    assert trainer.create_optimizer() is trainer.optimizer
    history = trainer.state.log_history
    assert any("lr_weight" in entry and "lr_qparam" in entry for entry in history)
    assert any("eval_loss" in entry for entry in history)
    assert any("tokens_per_sec" in entry for entry in history)
    assert any("clip_val_mean" in entry for entry in history)
    for step in (1, 2):
        assert (tmp_path / f"checkpoint-{step}" / STRUCTURE_FILE).is_file()
        assert (tmp_path / f"checkpoint-{step}" / WEIGHT_FILE).is_file()

    eval_metrics = trainer.evaluate()
    assert eval_metrics["eval_loss"] > 0

    trainer.save_model()
    assert (tmp_path / STRUCTURE_FILE).is_file()
    assert (tmp_path / WEIGHT_FILE).is_file()


def test_QATTrainer_train_uses_plain_adamw_when_tokenizer_given_without_qat_logging(tmp_path):
    trainer = QATTrainer(
        model=_tiny_gpt2(),
        train_cfg=_train_cfg(training_arguments={"disable_tqdm": True}, checkpointing={"save_eval_interval": 1}),
        output_dir=str(tmp_path),
        train_dataset=_rows(),
        eval_dataset=None,
        tokenizer=_make_tokenizer(),
        qparam_names=("weight_clip_val",),
        data_collator=default_data_collator,
    )
    assert trainer.data_collator is default_data_collator
    assert trainer.args.do_eval is False
    assert trainer.args.eval_strategy == "no"

    result = trainer.train()
    assert result.metrics["train_loss"] > 0
    inner_optimizer = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
    assert not isinstance(inner_optimizer, RelativeAlphaAdamW)
    assert "AdamW" in type(inner_optimizer).__name__
    assert (tmp_path / "checkpoint-2" / STRUCTURE_FILE).is_file()
    assert (tmp_path / "checkpoint-2" / WEIGHT_FILE).is_file()
