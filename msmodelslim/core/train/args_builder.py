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

# NestedTrainConfig -> transformers.TrainingArguments 翻译策略（可注入替换）：
# QAT 专属超参留在类型化 schema 上，其余字段按 transformers 版本直通或告警丢弃。

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from msmodelslim.core.train.task import NestedTrainConfig, TrainingArgumentsConfig
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import get_logger

try:
    from transformers import TrainingArguments
except ImportError as _exc:  # pragma: no cover - environment guard
    raise UnsupportedError(
        "transformers is required for msmodelslim QAT training",
        action="Please install transformers (and accelerate) before running msmodelslim qat.",
    ) from _exc


@dataclass(frozen=True)
class TrainingPlan:
    """由 ``NestedTrainConfig`` 翻译出的 HF 训练参数与 QAT 附加超参。"""

    training_args: TrainingArguments
    qparam_lr_scale: float
    ce_chunk_size: int
    # 老 transformers 无 warmup_ratio 字段时待换算（需 dataloader 长度，由 trainer 延迟换算）。
    warmup_ratio: Optional[float] = None
    log_qparams: bool = False
    alpha_min_ratio: float = 0.5
    alpha_max_ratio: float = 2.0


class TrainingArgumentsBuilder:
    """``NestedTrainConfig`` → ``TrainingArguments`` 翻译策略。

    ``train.training_arguments``（通用）直通 HF；``train.qat``（QAT 专属）由
    TrainingPlan 携带，QAT-owned 字段不再需要从直通字典中过滤。
    """

    # Fixed defaults for the QAT MVP training loop (overridable via train.training_arguments.*).
    _DEFAULT_TRAINING_ARGS: Dict[str, Any] = {
        "bf16": False,
        "logging_steps": 1,
        "report_to": [],  # about report to which logger such as tensorboard, etc.
        "remove_unused_columns": True,
        "save_safetensors": True,
        "group_by_length": True,  # group by length to save memory
        "prediction_loss_only": True,  # only need loss, not logits
    }

    def build(
        self,
        train_cfg: NestedTrainConfig,
        *,
        output_dir: str,
        has_eval_dataset: bool,
    ) -> TrainingPlan:
        """Map ``NestedTrainConfig`` onto ``TrainingArguments``."""
        logger = get_logger()
        hyper = train_cfg.training_arguments
        qat = train_cfg.qat
        ckpt = train_cfg.checkpointing

        qparam_lr_scale = float(qat.qparam_lr_scale)
        ce_chunk_size = int(qat.ce_chunk_size)
        log_qparams = bool(qat.log_qparams)
        alpha_min_ratio = float(qat.alpha_min_ratio)
        alpha_max_ratio = float(qat.alpha_max_ratio)
        logger.info(
            "train.qat.model_max_length=%s is applied by RankWorker to the tokenizer; not a TrainingArguments field.",
            qat.model_max_length,
        )

        valid_fields = {f.name for f in dataclasses.fields(TrainingArguments)}
        eval_strategy_key = "eval_strategy" if "eval_strategy" in valid_fields else "evaluation_strategy"

        args: Dict[str, Any] = {"output_dir": output_dir}
        for key, value in self._DEFAULT_TRAINING_ARGS.items():
            if key in valid_fields:
                args[key] = value

        for key, value in self._hyper_passthrough(hyper).items():
            if key in valid_fields:
                args[key] = value
            else:
                logger.warning(
                    "train.training_arguments.%s=%r is not a TrainingArguments field "
                    "in this transformers version; dropped.",
                    key,
                    value,
                )

        # warmup_ratio is a native TrainingArguments field; pass it through.
        # Only convert to warmup_steps when this transformers build has no such field.
        warmup_ratio_pending: Optional[float] = None
        if "warmup_ratio" not in valid_fields:
            warmup_ratio_pending = float(hyper.warmup_ratio)

        # FSDP2 + activation recompute frees all-gather weight storage before ParetoQ
        # backward; keep AC off on multi-card. Single-card has no FSDP, so default HF GC.
        fsdp_enabled = os.environ.get("ACCELERATE_USE_FSDP", "").lower() == "true"
        fsdp_ac = os.environ.get("FSDP_ACTIVATION_CHECKPOINTING", "").lower() == "true"
        if fsdp_ac or fsdp_enabled:
            if args.get("gradient_checkpointing"):
                logger.warning(
                    "Dropping TrainingArguments.gradient_checkpointing=true because "
                    "FSDP is enabled (FSDP_ACTIVATION_CHECKPOINTING=%s).",
                    fsdp_ac,
                )
            args["gradient_checkpointing"] = False
        elif "gradient_checkpointing" not in args:
            args["gradient_checkpointing"] = True
            logger.info("Enabled TrainingArguments.gradient_checkpointing for single-card QAT.")

        if args.get("gradient_checkpointing") and "gradient_checkpointing_kwargs" not in args:
            if "gradient_checkpointing_kwargs" in valid_fields:
                args["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

        do_eval = bool(has_eval_dataset)
        args["do_eval"] = do_eval

        interval = int(ckpt.save_eval_interval)
        args["save_strategy"] = "steps"
        args["save_steps"] = interval
        if do_eval:
            args[eval_strategy_key] = "steps"
            args["eval_steps"] = interval
        else:
            args[eval_strategy_key] = "no"

        if "eval_on_start" in valid_fields:
            args["eval_on_start"] = bool(ckpt.eval_on_start) and do_eval
        args["save_total_limit"] = int(ckpt.save_total_limit)
        load_best = bool(ckpt.load_best_model_at_end)
        if load_best and not do_eval:
            logger.warning(
                "checkpointing.load_best_model_at_end=true requires eval (dataset.ratio_eval_samples > 0); disabled."
            )
            load_best = False
        args["load_best_model_at_end"] = load_best
        if load_best:
            args["metric_for_best_model"] = "eval_loss"
            args["greater_is_better"] = False

        try:
            training_args = TrainingArguments(**args)
        except (TypeError, ValueError) as exc:
            raise SchemaValidateError(
                f"Failed to build TrainingArguments from train.training_arguments: {exc}",
                action="Please check train.training_arguments.* values in the QAT YAML.",
            ) from exc
        return TrainingPlan(
            training_args=training_args,
            qparam_lr_scale=qparam_lr_scale,
            ce_chunk_size=ce_chunk_size,
            warmup_ratio=warmup_ratio_pending,
            log_qparams=log_qparams,
            alpha_min_ratio=alpha_min_ratio,
            alpha_max_ratio=alpha_max_ratio,
        )

    def _hyper_passthrough(self, hyper: TrainingArgumentsConfig) -> Dict[str, Any]:
        """通用训练参数直通字典（声明字段 + extra 字段）。"""
        return hyper.model_dump(mode="python", by_alias=False)
