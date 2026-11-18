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

# 因果 LM chunked 交叉熵（算法无关，可被任意 QAT 训练器复用）：
# - 不物化 [B, S, V] logits，逐 chunk 计算并与 HF ForCausalLMLoss 数值对齐；
# - capture_backbone_hidden：final-norm 钩子捕获 hidden，同时 stub 掉 lm_head；
# - ChunkedCrossEntropyLoss：可注入的损失策略，无法处理时返回 None 由调用方回退。

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import torch
from torch.nn import functional as F

from msmodelslim.core.train.task import DEFAULT_CE_CHUNK_SIZE
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.logging import get_logger

_IGNORE_INDEX: int = -100


def count_shifted_ce_tokens(labels: Any) -> int:
    """Count causal-LM CE tokens (``labels[..., 1:]`` vs ``ignore_index=-100``)."""
    if not isinstance(labels, torch.Tensor) or labels.numel() == 0:
        return 0
    target = labels[:, 1:] if labels.dim() >= 2 and labels.size(1) > 1 else labels
    return int(target.ne(_IGNORE_INDEX).sum().item())


def _find_final_norm_module(unwrapped: Any) -> Any:
    """Locate the pre-``lm_head`` final norm (``model.norm`` / ``transformer.ln_f``)."""
    for attr in ("model", "transformer"):
        base = getattr(unwrapped, attr, None)
        if base is None:
            continue
        for norm_name in ("norm", "ln_f", "final_layernorm"):
            norm = getattr(base, norm_name, None)
            if norm is not None:
                return norm
    return None


def chunked_shifted_cross_entropy(
    lm_head: Any,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_items_in_batch: Optional[Any] = None,
    chunk_size: int = DEFAULT_CE_CHUNK_SIZE,
    ignore_index: int = _IGNORE_INDEX,
) -> torch.Tensor:
    """Causal-LM CE matching HF ``ForCausalLMLoss``, without a full ``[B, S, V]`` tensor.

    Shift is ``logits[..., :-1, :]`` vs ``labels[..., 1:]``. Each chunk is reduced
    with ``sum`` then divided by the same denominator HF uses (``num_items_in_batch``
    when the Trainer supplies it, otherwise the number of non-ignored tokens).

    Uses ``F.linear`` on ``lm_head.weight`` instead of ``lm_head(...)``. Under FSDP2,
    ``lm_head`` shares a param group with ``model.norm``; calling the wrapped module
    again after the backbone forward registers a post-backward that reshards that
    group before ``model.norm`` backward (NPU: data is not allocated yet).
    """
    if chunk_size < 1:
        raise SchemaValidateError(
            f"ce_chunk_size must be >= 1, got {chunk_size}",
            action="Please set train.qat.ce_chunk_size to a positive int.",
        )
    if hidden.dim() != 3:
        raise SchemaValidateError(
            f"chunked CE expects hidden [batch, seq, hidden], got shape {tuple(hidden.shape)}",
            action="Please keep the causal-LM backbone output as last_hidden_state.",
        )
    if labels.shape[:2] != hidden.shape[:2]:
        raise SchemaValidateError(
            f"labels shape {tuple(labels.shape)} does not match hidden {tuple(hidden.shape)}",
            action="Please pass collator labels aligned with input_ids.",
        )

    weight = lm_head.weight
    bias = lm_head.bias if getattr(lm_head, "bias", None) is not None else None
    shift_hidden = hidden[:, :-1, :].reshape(-1, hidden.size(-1))
    shift_labels = labels[:, 1:].reshape(-1).to(device=hidden.device)
    token_count = shift_hidden.size(0)
    total = hidden.new_zeros(())
    for start in range(0, token_count, chunk_size):
        end = min(start + chunk_size, token_count)
        logits = F.linear(shift_hidden[start:end], weight, bias)
        total = total + F.cross_entropy(
            logits.float(),
            shift_labels[start:end],
            ignore_index=ignore_index,
            reduction="sum",
        )

    if num_items_in_batch is not None:
        denom = num_items_in_batch
        if isinstance(denom, torch.Tensor):
            denom = denom.to(device=total.device, dtype=total.dtype)
        else:
            denom = total.new_tensor(float(denom))
        return total / denom

    n_valid = shift_labels.ne(ignore_index).sum().clamp(min=1).to(dtype=total.dtype)
    return total / n_valid


@contextmanager
def capture_backbone_hidden(unwrapped: Any, lm_head: Any) -> Iterator[Dict[str, torch.Tensor]]:
    """Capture final-norm hidden states; stub ``lm_head`` so vocab logits are not built.

    Hidden is taken from the final norm hook (not ``lm_head``'s FSDP-wrapped input).
    Re-entering the FSDP-wrapped ``lm_head`` for CE would reshard the shared
    ``norm``+``lm_head`` group before ``model.norm`` backward.
    """
    captured: Dict[str, torch.Tensor] = {}
    norm = _find_final_norm_module(unwrapped)
    hook_handle = None
    orig_forward = lm_head.forward

    def _norm_hook(_module: Any, _inputs: Any, output: Any) -> None:
        if isinstance(output, torch.Tensor):
            captured["hidden"] = output

    def _lm_head_stub(hidden: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return hidden.new_zeros(hidden.shape[:-1] + (1,))

    if norm is not None:
        hook_handle = norm.register_forward_hook(_norm_hook)
    else:

        def _forward(hidden: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            captured["hidden"] = hidden
            return hidden.new_zeros(hidden.shape[:-1] + (1,))

        lm_head.forward = _forward

    if hook_handle is not None:
        lm_head.forward = _lm_head_stub
    try:
        yield captured
    finally:
        lm_head.forward = orig_forward
        if hook_handle is not None:
            hook_handle.remove()


class LossOnlyOutput:
    """绕过logits的处理以节省内存。

    HuggingFace ``prediction_step`` does ``outputs[1:]`` (or treats a mapping as
    logits except ``loss``). Eval only needs ``eval_loss``; logits stay ``None``.
    """

    __slots__ = ("loss", "logits")

    def __init__(self, loss: torch.Tensor):
        self.loss = loss
        self.logits = None

    def __getitem__(self, index: Any):
        return (self.loss, self.logits)[index]

    def __len__(self) -> int:
        return 2

    def to_tuple(self) -> Tuple[Any, ...]:
        return (self.loss, self.logits)


class ChunkedCrossEntropyLoss:
    """因果 LM chunked 交叉熵策略（可注入）：不物化 ``[B, S, V]`` logits。

    无法处理（无 labels / 模型无 lm_head / final-norm hidden 未捕获）时返回
    ``None``，由调用方回退 HF Trainer 默认 ``compute_loss``。
    """

    def __init__(self, chunk_size: int = DEFAULT_CE_CHUNK_SIZE):
        if int(chunk_size) < 1:
            raise SchemaValidateError(
                f"ce_chunk_size must be >= 1, got {chunk_size}",
                action="Please set train.qat.ce_chunk_size to a positive int.",
            )
        self.chunk_size = int(chunk_size)

    def __call__(
        self,
        model: Any,
        inputs: Dict[str, Any],
        *,
        unwrap: Callable[[Any], Any],
        num_items_in_batch: Optional[Any] = None,
    ) -> Optional[Tuple[torch.Tensor, LossOnlyOutput]]:
        """计算 chunked CE；返回 ``(loss, LossOnlyOutput)`` 或 ``None``（回退信号）。"""
        labels = inputs.get("labels")
        if labels is None:
            return None

        unwrapped = unwrap(model)
        lm_head = getattr(unwrapped, "lm_head", None)
        if lm_head is None:
            get_logger().warning(
                "ChunkedCrossEntropyLoss: no lm_head on %s; falling back to full logits CE.",
                type(unwrapped).__name__,
            )
            return None

        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        with capture_backbone_hidden(unwrapped, lm_head) as captured:
            model(**model_inputs)

        hidden = captured.get("hidden")
        if hidden is None:
            get_logger().warning(
                "ChunkedCrossEntropyLoss: final-norm hidden was not captured; falling back to full logits CE."
            )
            return None

        loss = chunked_shifted_cross_entropy(
            lm_head,
            hidden,
            labels,
            num_items_in_batch=num_items_in_batch,
            chunk_size=self.chunk_size,
        )
        return loss, LossOnlyOutput(loss)
