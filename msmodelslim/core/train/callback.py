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

# QATCallback: training-loop hooks for fake-quant (QAT) runs.
#
# - Optional delayed fake-quant (``fake_quant_after_n_steps``, 详设 §2 / §9.4):
#   fake quant is *disabled* for the first N global steps and enabled after.
#   The toggle reuses the existing ``ParetoQFakeQuantLinear`` semantics —
#   ``w_bits >= 16`` makes its forward skip fake quantization — so no module
#   code change and no graph rewrite is needed.
# - Device-memory snapshot after the first optimizer step.
# - α(qparam) 统计由 ``QATTrainer`` 的日志注入统一负责，本回调不重复计算。

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

from torch import nn

from msmodelslim.core.train.device import log_device_memory
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import get_logger

try:
    from transformers import TrainerCallback
except ImportError as _exc:  # pragma: no cover - environment guard
    raise UnsupportedError(
        "transformers is required for msmodelslim QAT training",
        action="Please install transformers (and accelerate) before running msmodelslim qat.",
    ) from _exc

_FQ_DISABLED_W_BITS = 16  # lift w_bits to 16 so the float forward path runs (fake quant off)


def _iter_fake_quant_modules(model: nn.Module) -> Iterator[Tuple[str, nn.Module]]:
    """Yield (name, module) for the injected fake-quant shells (duck-typed).

    A module counts as a fake-quant shell when it exposes an integer
    w_bits and a weight_clip_val parameter (ParetoQFakeQuantLinear).
    """
    for name, module in model.named_modules():
        if isinstance(getattr(module, "w_bits", None), int) and hasattr(module, "weight_clip_val"):
            yield name, module


class QATCallback(TrainerCallback):
    """训练回调：延迟启用 fake quant + 训练起点/首步显存日志。

    :param fake_quant_after_n_steps: when set to N > 0, fake quant is kept OFF
        for the first N global steps (modules run the float path) and switched
        ON afterwards. ``None`` (default) means fake quant is active from step 0.
    """

    def __init__(self, fake_quant_after_n_steps: Optional[int] = None):
        if fake_quant_after_n_steps is not None and int(fake_quant_after_n_steps) < 0:
            raise SchemaValidateError(
                f"fake_quant_after_n_steps must be >= 0, got {fake_quant_after_n_steps}",
                action="Please pass a non-negative step count (or None to disable).",
            )
        self.fake_quant_after_n_steps: Optional[int] = (
            int(fake_quant_after_n_steps) if fake_quant_after_n_steps else None
        )
        self._orig_w_bits: Dict[int, int] = {}
        self._fake_quant_disabled = False

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any):
        """Log the number of fake-quant modules active and the fake_quant_after_n_steps."""
        model = kwargs.get("model")
        if model is None:
            return control

        fq_modules = list(_iter_fake_quant_modules(model))
        if state.is_world_process_zero:
            bits = sorted({module.w_bits for _, module in fq_modules})
            get_logger().info(
                "QATCallback: %d fake-quant modules active (w_bits=%s), fake_quant_after_n_steps=%s",
                len(fq_modules),
                bits,
                self.fake_quant_after_n_steps,
            )
        if not fq_modules and state.is_world_process_zero:
            get_logger().warning("QATCallback: no fake-quant module found on the model; was algorithm inject skipped?")

        if self.fake_quant_after_n_steps and state.global_step < self.fake_quant_after_n_steps:
            self._disable_fake_quant(fq_modules, state)
        return control

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any):
        """Enable fake quant if the global step is greater than or equal to the fake_quant_after_n_steps."""
        if not self._fake_quant_disabled:
            return control
        if state.global_step >= (self.fake_quant_after_n_steps or 0):
            model = kwargs.get("model")
            if model is not None:
                self._enable_fake_quant(model, state)
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any):
        if int(getattr(state, "global_step", 0) or 0) == 1:
            log_device_memory("after_first_optimizer_step")
        return control

    def _disable_fake_quant(self, fq_modules: List[Tuple[str, nn.Module]], state: Any) -> None:
        """Turn fake quant OFF by lifting ``w_bits`` to 16 (float forward path)."""
        self._orig_w_bits.clear()
        toggled = 0
        for _, module in fq_modules:
            if module.w_bits < _FQ_DISABLED_W_BITS:
                self._orig_w_bits[id(module)] = module.w_bits
                module.w_bits = _FQ_DISABLED_W_BITS
                toggled += 1
        self._fake_quant_disabled = toggled > 0
        if state.is_world_process_zero:
            get_logger().info(
                "QATCallback: fake quant disabled on %d modules until global_step >= %s.",
                toggled,
                self.fake_quant_after_n_steps,
            )

    def _enable_fake_quant(self, model: nn.Module, state: Any) -> None:
        """Restore each module's original ``w_bits`` (fake quant back ON)."""
        restored = 0
        for _, module in _iter_fake_quant_modules(model):
            orig = self._orig_w_bits.get(id(module))
            if orig is not None:
                module.w_bits = orig
                restored += 1
        self._orig_w_bits.clear()
        self._fake_quant_disabled = False
        if state.is_world_process_zero:
            get_logger().info(
                "QATCallback: fake quant enabled on %d modules at global_step=%s.",
                restored,
                state.global_step,
            )
