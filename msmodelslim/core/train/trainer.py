#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

# QATTrainer: HuggingFace Trainer 子类（薄编排层）。构造即完成入参校验、
# NestedTrainConfig -> TrainingArguments 翻译、HF 版本兼容适配与训练前日志；
# 分组 qparam LR / chunked CE / QAT checkpoint 保存分别委托给 optimizer /
# loss / checkpoint 模块的协作对象。Never train()/deploy.

from __future__ import annotations

import inspect
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset

from msmodelslim.core.train.args_builder import TrainingArgumentsBuilder, TrainingPlan
from msmodelslim.core.train.checkpoint import QATCheckpointSaver
from msmodelslim.core.train.loss import ChunkedCrossEntropyLoss, count_shifted_ce_tokens
from msmodelslim.core.train.metrics import (
    QParamStatsCollector,
    TokenThroughputTracker,
    is_eval_log,
)
from msmodelslim.core.train.optimizer import QParamGroupBuilder, RelativeAlphaAdamW
from msmodelslim.core.train.progress import QATProgressCallback
from msmodelslim.core.train.task import NestedTrainConfig
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import get_logger

try:
    from transformers import (
        PreTrainedTokenizerBase,
        Trainer,
        TrainerCallback,
    )
except ImportError as _exc:  # pragma: no cover - environment guard
    raise UnsupportedError(
        "transformers is required for msmodelslim QAT training",
        action="Please install transformers (and accelerate) before running msmodelslim qat.",
    ) from _exc


class QATTrainer(Trainer):
    """Thin ``transformers.Trainer`` subclass for fake-quant (QAT) training.

    直接实例化即可：``__init__`` 内完成校验、配置翻译与训练前日志；
    调用方（rank worker）驱动 ``train()``。

    Adds behaviors on top of the stock Trainer:

    1. **Grouped learning rate** — ``QParamGroupBuilder`` 把命中 ``qparam_names``
       的参数组成独立优化器参数组：``lr = learning_rate * qparam_lr_scale``、
       ``weight_decay = 0``。
    2. **QAT checkpoint save** — ``QATCheckpointSaver`` 收集 full state dict
       （FSDP ``FULL_STATE_DICT``）并经 ``CheckpointSaverFormat`` 写出
       （``qat_checkpoint.json`` + ``model.safetensors``）。
    3. **Chunked CE** — ``ChunkedCrossEntropyLoss`` 不物化 ``[B, S, V]``
       logits；目标函数与 HF causal-LM loss 一致。
    4. **Train logs** — ``log()`` 注入 ``lr_weight`` / ``lr_qparam``、全规约
       ``clip_val_{mean,min,max}``（按 ``qparam_names`` 泛化统计）与
       ``tokens_per_sec``。
    5. **Progress bar** — ``QATProgressCallback`` 以 step 为主进度，活跃指标
       挂 postfix（明细进 logger）。

    It must never rewrite the model graph nor run IR / deploy export.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        train_cfg: NestedTrainConfig,
        output_dir: str,
        train_dataset: Dataset,
        qparam_names: Sequence[str],
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        algorithm_cfg: Optional[Sequence[Dict[str, Any]]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        data_collator: Any = None,
        source_model_path: Optional[str] = None,
        args_builder: Optional[TrainingArgumentsBuilder] = None,
    ):
        """从 YAML ``train`` 段构造 QAT trainer（唯一构造入口）。

        :param model: 已完成算法注入的 fake-quant 模型（如 ParetoQFakeQuantLinear）。
        :param train_cfg: ``train`` 段（``NestedTrainConfig``）。
        :param output_dir: checkpoint 根目录（``save_path``）。
        :param qparam_names: ``pre_train`` 上报的可训练量化参数名；空为配置错误。
        :param algorithm_cfg: 序列化的 QAT 算法节点，写入 ``qat_checkpoint.json``。
        :param source_model_path: 源模型路径（写入 checkpoint 元数据）。
        :param args_builder: 配置翻译策略（默认 ``TrainingArgumentsBuilder``，可注入替换）。
        """
        self._validate_inputs(
            model=model,
            train_dataset=train_dataset,
            output_dir=output_dir,
            qparam_names=qparam_names,
        )
        plan = (args_builder or TrainingArgumentsBuilder()).build(
            train_cfg,
            output_dir=output_dir,
            has_eval_dataset=eval_dataset is not None,
        )

        # 协作对象：分组 / 损失 / 保存 / 指标（构造参数层可扩展替换）。
        self.qparam_names: Tuple[str, ...] = tuple(qparam_names)
        self.log_qparams: bool = bool(plan.log_qparams)
        self.alpha_min_ratio: float = float(plan.alpha_min_ratio)
        self.alpha_max_ratio: float = float(plan.alpha_max_ratio)
        self._param_group_builder = QParamGroupBuilder(
            self.qparam_names,
            plan.qparam_lr_scale,
            alpha_min_ratio=self.alpha_min_ratio,
            alpha_max_ratio=self.alpha_max_ratio,
        )
        self._loss_fn = ChunkedCrossEntropyLoss(chunk_size=plan.ce_chunk_size)
        self._checkpoint_saver = QATCheckpointSaver(
            qparam_names=self.qparam_names,
            algorithm_cfg=algorithm_cfg,
            source_model_path=source_model_path,
        )
        self._throughput = TokenThroughputTracker()
        self._qparam_stats = QParamStatsCollector(self.qparam_names)
        self._pending_warmup_ratio = plan.warmup_ratio

        trainer_kwargs = self._build_hf_init_kwargs(
            model=model,
            plan=plan,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
            data_collator=data_collator,
        )
        super().__init__(**trainer_kwargs)
        if getattr(self, "optimizer", None) is None:
            self.optimizer = None

        self._install_progress_callback()
        self._log_planned_training_steps()
        get_logger().info(
            "QATTrainer ready: output_dir=%s save_steps=%s eval_steps=%s "
            "eval_on_start=%s load_best_model_at_end=%s group_by_length=%s "
            "gradient_checkpointing=%s ce_chunk_size=%s qparam_names=%s "
            "qparam_lr_scale=%s log_qparams=%s alpha_ratio=[%s, %s] algorithm_cfg=%d",
            plan.training_args.output_dir,
            getattr(plan.training_args, "save_steps", None),
            getattr(plan.training_args, "eval_steps", None),
            getattr(plan.training_args, "eval_on_start", None),
            getattr(plan.training_args, "load_best_model_at_end", None),
            getattr(plan.training_args, "group_by_length", None),
            getattr(plan.training_args, "gradient_checkpointing", None),
            plan.ce_chunk_size,
            self.qparam_names,
            self._param_group_builder.qparam_lr_scale,
            self.log_qparams,
            self.alpha_min_ratio,
            self.alpha_max_ratio,
            len(self._checkpoint_saver.algorithm_cfg),
        )

    @staticmethod
    def _validate_inputs(
        *,
        model: Any,
        train_dataset: Any,
        output_dir: str,
        qparam_names: Sequence[str],
    ) -> None:
        """构造前置校验（fail fast，带可操作的修复提示）。"""
        if model is None:
            raise SchemaValidateError(
                "QATTrainer requires an injected model",
                action="Please run algorithm inject before building the trainer.",
            )
        if train_dataset is None:
            raise SchemaValidateError(
                "QATTrainer requires a train_dataset",
                action="Please build datasets from spec.dataset before building the trainer.",
            )
        if not str(output_dir or "").strip():
            raise SchemaValidateError(
                "QATTrainer requires a non-empty output_dir",
                action="Please pass save_path as output_dir.",
            )
        if not tuple(qparam_names):
            raise SchemaValidateError(
                "qparam_names is empty; algorithm inject did not report trainable quant params",
                action="Please run ITrainHook.pre_train and pass core_task.qparam_names to QATTrainer.",
            )

    def _build_hf_init_kwargs(
        self,
        *,
        model: nn.Module,
        plan: TrainingPlan,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        tokenizer: Optional[PreTrainedTokenizerBase],
        callbacks: Optional[List[TrainerCallback]],
        data_collator: Any,
    ) -> Dict[str, Any]:
        """组装 HF ``Trainer.__init__`` 参数（含 tokenizer/processing_class 版本兼容）。"""
        kwargs: Dict[str, Any] = {
            "model": model,
            "args": plan.training_args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "data_collator": data_collator,
            "callbacks": callbacks,
        }
        if tokenizer is not None:
            init_params = inspect.signature(Trainer.__init__).parameters
            if "processing_class" in init_params:
                kwargs["processing_class"] = tokenizer
            else:
                kwargs["tokenizer"] = tokenizer
        return kwargs

    def _install_progress_callback(self) -> None:
        """Replace HF ProgressCallback / PrinterCallback with QATProgressCallback."""
        try:
            from transformers.trainer_callback import PrinterCallback, ProgressCallback
        except ImportError:
            ProgressCallback = None
            PrinterCallback = None
        if ProgressCallback is not None:
            self.pop_callback(ProgressCallback)
        if PrinterCallback is not None:
            self.pop_callback(PrinterCallback)
        handler = getattr(self, "callback_handler", None)
        existing = getattr(handler, "callbacks", None) or []
        if not any(isinstance(cb, QATProgressCallback) for cb in existing):
            self.add_callback(QATProgressCallback())

    def create_optimizer(self, model=None):
        """Create the optimizer with a dedicated param group for qparams (α).

        分组委托 ``QParamGroupBuilder``：
          - qparam (name endswith ``qparam_names``): ``lr * qparam_lr_scale``, no decay
          - regular decayed params: base lr, ``args.weight_decay``
          - regular no-decay params (bias / norm): base lr, no decay
        """
        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None:
            return optimizer

        opt_model = self.model_wrapped if self.model_wrapped is not None else self.model
        decay_names = set(self.get_decay_parameter_names(opt_model))  # get the names of the parameters that have decay

        optimizer_cls, optimizer_kwargs = self._resolve_optimizer_cls_and_kwargs(opt_model)
        base_lr = float(optimizer_kwargs.pop("lr", self.args.learning_rate))

        result = self._param_group_builder.build(
            opt_model,
            decay_names=decay_names,
            base_lr=base_lr,
            weight_decay=self.args.weight_decay,
        )
        if self.log_qparams and result.n_qparam > 0:
            qparam_params = []
            for group in result.groups:
                if group.get("qat_group") == "qparam":
                    qparam_params.extend(group["params"])
            alpha_ref = {id(param): param.detach().to(torch.float32).clone() for param in qparam_params}
            self.optimizer = RelativeAlphaAdamW(
                result.groups,
                lr=base_lr,
                alpha_ref=alpha_ref,
                alpha_min_ratio=self.alpha_min_ratio,
                alpha_max_ratio=self.alpha_max_ratio,
                **optimizer_kwargs,
            )
            optimizer_name = "RelativeAlphaAdamW(log-space alpha)"
        else:
            self.optimizer = optimizer_cls(result.groups, lr=base_lr, **optimizer_kwargs)
            optimizer_name = optimizer_cls.__name__
        get_logger().info(
            "QATTrainer optimizer created: %s | decay=%d no_decay=%d qparam=%d | "
            "base_lr=%.3e qparam_lr=%.3e (scale=%.3g) log_qparams=%s alpha_ratio=[%s, %s]",
            optimizer_name,
            result.n_decay,
            result.n_no_decay,
            result.n_qparam,
            base_lr,
            base_lr * self._param_group_builder.qparam_lr_scale,
            self._param_group_builder.qparam_lr_scale,
            self.log_qparams,
            self.alpha_min_ratio,
            self.alpha_max_ratio,
        )
        return self.optimizer

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """HF ``loss`` / ``learning_rate`` plus QAT lr / α stats / tokens_per_sec."""
        payload = dict(logs or {})
        self._inject_qat_log_metrics(payload)
        params = inspect.signature(Trainer.log).parameters
        if "start_time" in params:
            return Trainer.log(self, payload, start_time=start_time)
        return Trainer.log(self, payload)

    def _inject_qat_log_metrics(self, logs: Dict[str, Any]) -> None:
        """Inject QAT log metrics into the logs dictionary."""
        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None:
            for group in optimizer.param_groups:
                tag = group.get("qat_group")
                lr = group.get("lr")
                if lr is None:
                    continue
                if tag == "weight" and "lr_weight" not in logs:
                    logs["lr_weight"] = float(lr)
                elif tag == "qparam":
                    logs["lr_qparam"] = float(lr)
        eval_log = is_eval_log(logs)
        if not eval_log:
            tokens_per_sec = self._throughput.consume_per_sec()
            if tokens_per_sec is not None:
                logs["tokens_per_sec"] = tokens_per_sec
        model = getattr(self, "model", None)
        if model is None:
            return
        # 日志键沿用 clip_val_*（进度条/明细格式化按此消费）；统计本身按 qparam_names 泛化。
        stats = self._qparam_stats.summary(
            model,
            step=int(getattr(getattr(self, "state", None), "global_step", 0) or 0),
            is_eval=eval_log,
            min_interval=int(getattr(self.args, "logging_steps", 1) or 1),
        )
        if stats is None:
            return
        logs["clip_val_n"] = float(stats.count)
        logs["clip_val_mean"] = float(stats.mean)
        logs["clip_val_min"] = float(stats.vmin)
        logs["clip_val_max"] = float(stats.vmax)

    def evaluate(self, *args: Any, **kwargs: Any):
        try:
            return Trainer.evaluate(self, *args, **kwargs)
        finally:
            self._throughput.reset()

    def _log_planned_training_steps(self) -> None:
        """Log planned optimizer steps using the same dataloader-length formula as HF Trainer."""
        args = self.args
        train_dataset = getattr(self, "train_dataset", None)
        num_examples = None
        if train_dataset is not None and hasattr(train_dataset, "__len__"):
            try:
                num_examples = len(train_dataset)
            except TypeError:
                num_examples = None

        try:
            num_batches = len(self.get_train_dataloader())
        except TypeError:
            get_logger().warning("QAT planned schedule: train dataloader has no length; skip max_steps log.")
            return

        accum = max(int(args.gradient_accumulation_steps), 1)
        steps_per_epoch = max(num_batches // accum, 1)
        configured_max_steps = int(getattr(args, "max_steps", -1) or -1)
        if configured_max_steps > 0:
            total_steps = configured_max_steps
            epochs_shown: Any = total_steps / steps_per_epoch
        else:
            epochs_shown = args.num_train_epochs
            total_steps = math.ceil(float(args.num_train_epochs) * steps_per_epoch)

        ratio = self._pending_warmup_ratio
        configured_warmup_steps = int(getattr(args, "warmup_steps", 0) or 0)
        if ratio is not None and configured_warmup_steps <= 0:
            args.warmup_steps = max(0, int(math.ceil(float(ratio) * float(total_steps))))
        self._pending_warmup_ratio = None

        warmup_steps = None
        if hasattr(args, "get_warmup_steps"):
            warmup_steps = args.get_warmup_steps(total_steps)

        world_size = int(getattr(args, "world_size", 1) or 1)
        micro_bs = int(args.per_device_train_batch_size)
        is_main = True
        if hasattr(self, "is_world_process_zero"):
            is_main = bool(self.is_world_process_zero())
        if not is_main:
            return

        if ratio is not None and configured_warmup_steps <= 0:
            get_logger().info(
                "Converted warmup_ratio=%s to warmup_steps=%s (max_steps=%s)",
                ratio,
                getattr(args, "warmup_steps", None),
                total_steps,
            )

        get_logger().info(
            "QAT planned training steps (before train): num_examples=%s world_size=%s "
            "per_device_train_batch_size=%s gradient_accumulation_steps=%s "
            "dataloader_batches_per_rank=%s steps_per_epoch=%s num_train_epochs=%s "
            "max_steps=%s warmup_steps=%s global_batch_size=%s",
            num_examples,
            world_size,
            micro_bs,
            accum,
            num_batches,
            steps_per_epoch,
            epochs_shown,
            total_steps,
            warmup_steps,
            micro_bs * world_size * accum,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """Compute the loss for the model (chunked CE with HF fallback)."""
        labels = inputs.get("labels")
        if labels is not None and bool(
            getattr(model, "training", False)
        ):  # if the model is training and the labels are not None, count the tokens
            self._throughput.add_tokens(count_shifted_ce_tokens(labels))
        result = None
        if labels is not None:
            result = self._loss_fn(
                model,
                inputs,
                unwrap=self._unwrap_training_model,
                num_items_in_batch=num_items_in_batch,
            )
        if result is None:  # no labels / no lm_head / hidden not captured: fall back to HF
            return self._call_super_compute_loss(model, inputs, return_outputs, num_items_in_batch, kwargs)
        loss, outputs = result
        if return_outputs:
            return loss, outputs
        return loss

    def prediction_step(
        self,
        model: Any,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """Evaluate with chunked CE; do not gather [B, S, V] logits (HF indexes outputs[1:])."""
        del prediction_loss_only, ignore_keys, kwargs
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss_ctx = getattr(self, "compute_loss_context_manager", None)
            if callable(loss_ctx):
                with loss_ctx():  # pylint: disable=not-callable
                    loss = self.compute_loss(model, inputs, return_outputs=False)
            else:
                loss = self.compute_loss(model, inputs, return_outputs=False)
            if loss is not None:
                loss = loss.mean().detach()
        return loss, None, None

    def _unwrap_training_model(self, model: Any) -> Any:
        """Unwrap the training model."""
        accelerator = getattr(self, "accelerator", None)
        if accelerator is None:
            return model
        try:
            return accelerator.unwrap_model(model, keep_torch_compile=False)
        except TypeError:
            return accelerator.unwrap_model(model)

    def _call_super_compute_loss(
        self,
        model: Any,
        inputs: Dict[str, Any],
        return_outputs: bool,
        num_items_in_batch: Optional[Any],
        extra_kwargs: Dict[str, Any],
    ):
        params = inspect.signature(Trainer.compute_loss).parameters
        kwargs = dict(extra_kwargs)
        if "num_items_in_batch" in params:
            kwargs["num_items_in_batch"] = num_items_in_batch
        return Trainer.compute_loss(self, model, inputs, return_outputs=return_outputs, **kwargs)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save QAT checkpoint via ``CheckpointSaverFormat`` (not HF ``Trainer._save``).

        All ranks participate in FSDP full-state gather; only the main process
        writes ``qat_checkpoint.json`` + ``model.safetensors`` (+ HF sidecars
        inside the format). Disk pick of best vs latest is App
        ``_select_checkpoint``; in-memory end-of-train reload follows
        ``checkpointing.load_best_model_at_end``.
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        self._checkpoint_saver.save(
            model=self.model,
            accelerator=self.accelerator,
            args=self.args,
            output_dir=output_dir,
            is_fsdp_enabled=bool(getattr(self, "is_fsdp_enabled", False)),
        )

        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save", revision=getattr(self.args, "hub_revision", None))

    def _resolve_optimizer_cls_and_kwargs(self, opt_model: Any) -> Tuple[Any, Dict[str, Any]]:
        """Version-tolerant wrapper around Trainer.get_optimizer_cls_and_kwargs."""
        getter = Trainer.get_optimizer_cls_and_kwargs
        try:
            return getter(self.args, opt_model)
        except TypeError:
            # Older transformers: get_optimizer_cls_and_kwargs(args) only.
            return getter(self.args)
