"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

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

from __future__ import annotations

from typing import Callable, Dict, Literal

import torch

TLQElementwiseLossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
TLQLossType = Literal["l1", "custom_outlier"]

_L1_LOSS = torch.nn.L1Loss(reduction="none")


def custom_outlier_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    loss_function: TLQElementwiseLossFn,
) -> torch.Tensor:
    # Currently we turned on non-outlier regime
    mean_out = torch.mean(target, dim=1, keepdim=True)
    std_out = torch.std(target - mean_out, dim=1, keepdim=True)
    outlier_mask = torch.abs(target - mean_out) < (3 * std_out)

    outlier_loss = loss_function(output * outlier_mask, target * outlier_mask)
    common_loss = loss_function(output, target)

    return 0.3 * common_loss + 0.7 * outlier_loss


def l1_elementwise_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _L1_LOSS(output, target)


def masked_mean_loss(
    loss: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """对 elementwise loss 应用 ``(B, S)`` mask，并在 hidden 维上取均值。"""
    return (loss * loss_mask.unsqueeze(-1)).sum() / loss_mask.sum() / loss.shape[-1]


def loss_tensor_pairs(
    handler,
    output_raw,
    target_raw,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """将 block 输出与 teacher 输出对齐为若干 ``(quant, float)`` tensor 对。"""
    if isinstance(output_raw, (tuple, list)):
        if not isinstance(target_raw, (tuple, list)):
            raise TypeError(
                f"Quantized block output is {type(output_raw).__name__} but float output is {type(target_raw).__name__}"
            )
        if len(output_raw) != len(target_raw):
            raise ValueError(f"Quantized/float output tuple lengths differ: {len(output_raw)} vs {len(target_raw)}")
        return list(zip(output_raw, target_raw))
    if isinstance(target_raw, (tuple, list)):
        raise TypeError(
            f"Float output is {type(target_raw).__name__} but quantized block output is {type(output_raw).__name__}"
        )
    return [
        (
            handler.extract_hidden_states(output_raw),
            handler.extract_hidden_states(target_raw),
        )
    ]


_LOSS_REGISTRY: Dict[str, TLQElementwiseLossFn] = {
    "l1": l1_elementwise_loss,
    "custom_outlier": lambda output, target: custom_outlier_loss(output, target, _L1_LOSS),
}


def get_elementwise_loss_fn(loss_type: str) -> TLQElementwiseLossFn:
    try:
        return _LOSS_REGISTRY[loss_type]
    except KeyError as exc:
        supported = ", ".join(sorted(_LOSS_REGISTRY))
        raise ValueError(f"Unsupported TLQ loss_type={loss_type!r}; supported: {supported}") from exc
