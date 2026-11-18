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

# QATProgressCallback: tqdm shows optimizer-step progress; postfix holds
# loss / lr_w / lr_q / alpha / tok/s; clip min/max, grad_norm, memory go to logger.

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - transformers already depends on tqdm
    tqdm = None

from msmodelslim.core.train.device import get_device_memory
from msmodelslim.core.train.metrics import is_eval_log
from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.utils.logging import get_logger

try:
    from transformers import TrainerCallback
except ImportError as _exc:  # pragma: no cover - environment guard
    raise UnsupportedError(
        "transformers is required for msmodelslim QAT training",
        action="Please install transformers (and accelerate) before running msmodelslim qat.",
    ) from _exc

_DETAIL_LOG_EVERY = 10  # log the detail every 10 steps
_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"  # the format of the progress bar


class QATProgressCallback(TrainerCallback):
    """tqdm bar shows step progress; live metrics go to postfix; details to the logger.

    Postfix: ``loss``, ``lr_w``, ``lr_q``, ``alpha``, ``tok/s``.
    Logger (every ``_DETAIL_LOG_EVERY`` train steps, and every eval): ``grad_norm``,
    ``clip_val`` min/max/n, device memory.
    """

    def __init__(self):
        self.training_bar = None
        self.prediction_bar = None
        self._current_step = 0

    @staticmethod
    def _use_bar(args: Any, state: Any) -> bool:
        if not getattr(state, "is_world_process_zero", False):
            return False
        if bool(getattr(args, "disable_tqdm", False)):
            return False
        return tqdm is not None

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any):
        self._current_step = int(getattr(state, "global_step", 0) or 0)
        if self._use_bar(args, state):
            total = int(getattr(state, "max_steps", 0) or 0)
            self.training_bar = tqdm(
                total=total if total > 0 else None,
                desc="QAT",
                dynamic_ncols=True,
                bar_format=_BAR_FORMAT,
            )
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any):
        if self.training_bar is None:
            return control
        step = int(getattr(state, "global_step", 0) or 0)
        delta = step - self._current_step
        if delta > 0:
            self.training_bar.update(delta)
            self._current_step = step
        return control

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        payload = dict(logs or {})
        postfix = _format_train_postfix(payload)
        if self.training_bar is not None:
            if postfix:
                self.training_bar.set_postfix_str(postfix, refresh=True)
        elif state.is_world_process_zero and postfix:
            max_steps = int(getattr(state, "max_steps", 0) or 0)
            get_logger().info(
                "QAT %s/%s %s",
                getattr(state, "global_step", 0),
                max_steps,
                postfix,
            )
        if state.is_world_process_zero and _should_log_detail(payload, state):
            get_logger().info("%s", _format_detail_line(payload, state))
        return control

    def on_prediction_step(
        self,
        args: Any,
        state: Any,
        control: Any,
        eval_dataloader: Any = None,
        **kwargs: Any,
    ):
        if not self._use_bar(args, state):
            return control
        if self.prediction_bar is None and eval_dataloader is not None:
            try:
                total = len(eval_dataloader)
            except TypeError:
                total = None
            if total:
                self.prediction_bar = tqdm(
                    total=total,
                    desc="QAT eval",
                    leave=False,
                    dynamic_ncols=True,
                    bar_format=_BAR_FORMAT,
                )
        if self.prediction_bar is not None:
            self.prediction_bar.update(1)
        return control

    def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: Any):
        self._close_prediction_bar()
        return control

    def on_predict(self, args: Any, state: Any, control: Any, **kwargs: Any):
        self._close_prediction_bar()
        return control

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any):
        self._close_prediction_bar()
        if self.training_bar is not None:
            self.training_bar.close()
            self.training_bar = None
        return control

    def _close_prediction_bar(self) -> None:
        if self.prediction_bar is not None:
            self.prediction_bar.close()
            self.prediction_bar = None


def _should_log_detail(logs: Dict[str, Any], state: Any) -> bool:
    if is_eval_log(logs):
        return True
    step = int(getattr(state, "global_step", 0) or 0)
    return step > 0 and step % _DETAIL_LOG_EVERY == 0


def _format_metric(value: Any, spec: str) -> Optional[str]:
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return None


def _format_train_postfix(logs: Dict[str, Any]) -> str:
    parts: List[str] = []
    if is_eval_log(logs):
        formatted = _format_metric(logs.get("eval_loss"), ".4f")
        if formatted is not None:
            parts.append(f"eval_loss={formatted}")
        formatted = _format_metric(logs.get("clip_val_mean"), ".3g")
        if formatted is not None:
            parts.append(f"alpha={formatted}")
        return "  ".join(parts)

    formatted = _format_metric(logs.get("loss"), ".4f")
    if formatted is not None:
        parts.append(f"loss={formatted}")
    formatted = _format_metric(logs.get("lr_weight"), ".2e")
    if formatted is not None:
        parts.append(f"lr_w={formatted}")
    formatted = _format_metric(logs.get("lr_qparam"), ".2e")
    if formatted is not None:
        parts.append(f"lr_q={formatted}")
    formatted = _format_metric(logs.get("clip_val_mean"), ".3g")
    if formatted is not None:
        parts.append(f"alpha={formatted}")
    formatted = _format_metric(logs.get("tokens_per_sec"), ".0f")
    if formatted is not None:
        parts.append(f"tok/s={formatted}")
    return "  ".join(parts)


def _format_detail_line(logs: Dict[str, Any], state: Any) -> str:
    step = getattr(state, "global_step", 0)
    kind = "eval" if is_eval_log(logs) else "train"
    fields: List[str] = []
    if kind == "eval":
        formatted = _format_metric(logs.get("eval_loss"), ".4f")
        if formatted is not None:
            fields.append(f"eval_loss={formatted}")
    else:
        formatted = _format_metric(logs.get("grad_norm"), ".4g")
        if formatted is not None:
            fields.append(f"grad_norm={formatted}")
    formatted = _format_metric(logs.get("clip_val_min"), ".4g")
    if formatted is not None:
        fields.append(f"clip_min={formatted}")
    formatted = _format_metric(logs.get("clip_val_max"), ".4g")
    if formatted is not None:
        fields.append(f"clip_max={formatted}")
    if logs.get("clip_val_n") is not None:
        fields.append(f"clip_n={int(float(logs['clip_val_n']))}")
    device_name, allocated_gb, reserved_gb = get_device_memory()
    if allocated_gb is not None:
        reserved_s = f"/{reserved_gb:.1f}" if reserved_gb is not None else ""
        fields.append(f"mem={allocated_gb:.1f}{reserved_s}GiB({device_name})")
    return f"QAT {kind}[step={step}]: " + " ".join(fields)
