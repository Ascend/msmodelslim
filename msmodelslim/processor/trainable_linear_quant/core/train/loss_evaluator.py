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

# Block forward + reconstruction loss for TLQ training.

from __future__ import annotations

from typing import Any, Iterator, List, Sequence, Tuple, Union

import torch
from torch import nn

from msmodelslim.processor.trainable_linear_quant.config.train_config import BlockTrainConfig
from .loss_functions import (
    TLQElementwiseLossFn,
    TLQLossType,
    get_elementwise_loss_fn,
    loss_tensor_pairs,
    masked_mean_loss,
)
from msmodelslim.processor.trainable_linear_quant.data import (
    BlockInput,
    BlockOutput,
    TLQBlockDataInterface,
)

TrainingAccumulateSlot = Sequence[Tuple[BlockInput, BlockOutput]]

# 反向前固定 loss 缩放；效果可用 lr 调节，不暴露为用户配置
_DEFAULT_LOSS_SCALE = 1000.0


def resolve_device_type(device: Union[str, torch.device]) -> str:
    """Return device type string (e.g. ``cpu``, ``cuda``, ``npu``)."""
    return torch.device(device).type


def to_device(inputs: Any, device: Union[str, torch.device] = torch.device("cpu")) -> Any:
    """Recursively move tensors in nested inputs to ``device``."""
    if inputs is None:
        return None
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if isinstance(inputs, torch.Tensor):
        if inputs.device.type == dev.type and inputs.device.index == dev.index:
            return inputs
        return inputs.to(dev)
    if isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    if isinstance(inputs, (list, tuple)):
        if len(inputs) > 0:
            return type(inputs)(to_device(inp, device) for inp in inputs)
    return inputs


def iter_tlq_accumulate_slots(
    all_datas: List[BlockInput],
    teacher_outputs: List[BlockOutput],
    indices: torch.Tensor,
    batch_size: int,
    gradient_accumulate_steps: int,
) -> Iterator[TrainingAccumulateSlot]:
    """按梯度累加步划分训练 slot，保证每个 sample 与 teacher 对齐。

    每个 slot 对应一次 ``loss.backward``（trainer 侧对一个 slot 内多 sample 取均值）。
    """
    gas = max(gradient_accumulate_steps, 1)
    idx_list = [int(i) for i in indices.detach().cpu().flatten().tolist()]
    if not idx_list:
        idx_list = [0]

    for step in range(gas):
        start = step * batch_size
        step_indices = idx_list[start : start + batch_size]
        if not step_indices:
            step_indices = idx_list[:1]

        if len(step_indices) == 1:
            j = step_indices[0]
            yield ((all_datas[j], teacher_outputs[j]),)
        else:
            yield tuple((all_datas[j], teacher_outputs[j]) for j in step_indices)


class BlockLossEvaluator:
    """Run quantized block forward and compute teacher-aligned reconstruction loss."""

    def __init__(
        self,
        block_data: TLQBlockDataInterface,
        loss_type: TLQLossType = "l1",
        batch_size: int = 1,
        gradient_accumulate_steps: int = 1,
        loss_scale: float = _DEFAULT_LOSS_SCALE,
    ) -> None:
        if block_data is None:
            raise ValueError("block_data is required for BlockLossEvaluator")
        self._block_data = block_data
        self._loss_scale = loss_scale
        self._loss_type = loss_type
        self._batch_size = batch_size
        self._gradient_accumulate_steps = gradient_accumulate_steps
        self._elementwise_loss_fn: TLQElementwiseLossFn = get_elementwise_loss_fn(loss_type)

    @classmethod
    def from_config(
        cls,
        block_data: TLQBlockDataInterface,
        config: BlockTrainConfig,
        batch_size: int = 1,
    ) -> BlockLossEvaluator:
        return cls(
            block_data,
            loss_type=config.loss_type,
            batch_size=batch_size,
            gradient_accumulate_steps=config.gradient_accumulate_steps,
        )

    def eval_and_backward(
        self,
        block: nn.Module,
        all_datas: List[BlockInput],
        teacher_outputs: List[BlockOutput],
        device: Union[str, torch.device],
        indices: torch.Tensor,
    ) -> float:
        """Mean loss over accumulate slots; accumulates gradients when backward."""
        gas = max(self._gradient_accumulate_steps, 1)
        total_loss = 0.0

        with torch.enable_grad():
            for slot in iter_tlq_accumulate_slots(
                all_datas,
                teacher_outputs,
                indices,
                batch_size=self._batch_size,
                gradient_accumulate_steps=self._gradient_accumulate_steps,
            ):
                microbatch_loss = self._compute_microbatch_loss(block, slot, device)
                self._scale_loss_and_backward(microbatch_loss)
                total_loss += microbatch_loss.detach().item()

        return total_loss / gas

    def _compute_microbatch_loss(
        self,
        block: nn.Module,
        slot: Sequence[tuple[BlockInput, BlockOutput]],
        device: Union[str, torch.device],
    ) -> torch.Tensor:
        """Mean sample loss over one accumulate slot (microbatch)."""
        sample_losses: List[torch.Tensor] = []
        for block_input, teacher_output in slot:
            sample_losses.append(self._compute_sample_loss(block, block_input, teacher_output, device))
        return sum(sample_losses) / len(sample_losses)

    def _compute_sample_loss(
        self,
        block: nn.Module,
        block_input: BlockInput,
        teacher_output: BlockOutput,
        device: Union[str, torch.device],
    ) -> torch.Tensor:
        """Quantized block forward + teacher-aligned reconstruction loss for one sample."""
        args, kwargs = block_input
        # Always align args/kwargs (incl. attention_mask) to train device. Matching only
        # on input_hidden would leave CPU masks when hidden is already on NPU.
        args = tuple(to_device(a, device) if isinstance(a, torch.Tensor) else a for a in args)
        kwargs = to_device(kwargs, device)

        output_q_raw = block(*args, **kwargs)
        output_q_raw = to_device(output_q_raw, device)
        teacher_on_device = to_device(teacher_output, device)

        tensor_pairs = loss_tensor_pairs(self._block_data, output_q_raw, teacher_on_device)
        # Prefer kwargs already on device; still pin mask to loss device for safety.
        loss_mask = self._block_data.get_loss_mask((args, kwargs), tensor_pairs[0][0]).to(
            device=tensor_pairs[0][0].device
        )

        loss = None
        for output_q, current_output in tensor_pairs:
            elem_loss = self._elementwise_loss_fn(
                output_q.to(torch.float32),
                current_output.to(torch.float32),
            )
            pair_loss = masked_mean_loss(elem_loss, loss_mask)
            loss = pair_loss if loss is None else loss + pair_loss
        return loss / len(tensor_pairs)

    def _scale_loss_and_backward(self, loss: torch.Tensor) -> torch.Tensor:
        scale_loss = loss * self._loss_scale
        scale_loss.backward()
        return scale_loss


__all__ = ["BlockLossEvaluator"]
