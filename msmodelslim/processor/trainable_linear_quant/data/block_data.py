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

# Block-level data contracts and default interface implementation for TLQ.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, MutableSequence, Optional, Tuple

import torch

from msmodelslim.utils.exception import UnsupportedError

BlockInput = Tuple[MutableSequence[Any], Dict[str, Any]]
BlockOutput = Any

_DEFAULT_HANDLER_ACTION = (
    "Block I/O does not match DefaultTLQBlockDataInterface; "
    "have the model adapter inherit TLQBlockDataInterface and implement "
    "extract_hidden_states / inject_hidden_states / get_loss_mask."
)


class TLQBlockDataInterface(ABC):
    """TLQ block 数据接口：主 hidden 读写与 loss mask。

    模型 Adapter 可直接继承本类并实现下列方法（与 IterSmoothInterface 用法一致）；
    未继承时 Processor 使用 ``DefaultTLQBlockDataInterface``。

    - ``extract_hidden_states``：从 block 输出取主 hidden（loss + 层间传播读端）
    - ``inject_hidden_states``：将主 hidden 写回 block 输入（层间传播写端）
    - ``get_loss_mask``：返回参与 loss 的 token mask
    """

    @abstractmethod
    def extract_hidden_states(self, block_output: BlockOutput) -> torch.Tensor:
        """从 block forward 返回值中提取主 hidden states。"""

    @abstractmethod
    def inject_hidden_states(self, block_input: BlockInput, hidden: torch.Tensor) -> None:
        """将主 hidden states 就地写回 ``block_input``。"""

    @abstractmethod
    def get_loss_mask(
        self,
        block_input: BlockInput,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """返回 shape ``(B, S)`` 的 bool mask，True 表示参与 loss 计算。"""


def propagate_outputs_to_inputs(
    block_data: TLQBlockDataInterface,
    datas: MutableSequence[BlockInput],
    propagation_outputs: List[BlockOutput],
) -> None:
    """用上一层输出替换当前 block 各 sample 的主输入 hidden states。"""
    for data, out in zip(datas, propagation_outputs):
        block_data.inject_hidden_states(data, block_data.extract_hidden_states(out))


class DefaultTLQBlockDataInterface(TLQBlockDataInterface):
    """默认 block 数据接口实现。

    假设：
    - ``args[0]`` 是主输入 tensor，或 ``args[0][0]`` 为嵌套结构中的主 tensor
    - block forward 返回单个 Tensor，或 tuple/list 且第一个元素是 hidden states
    """

    def extract_hidden_states(self, block_output: Any) -> torch.Tensor:
        if isinstance(block_output, (tuple, list)):
            if not block_output:
                raise UnsupportedError(
                    "DefaultTLQBlockDataInterface cannot extract hidden states from an empty "
                    f"{type(block_output).__name__} block output",
                    action=_DEFAULT_HANDLER_ACTION,
                )
            first = block_output[0]
            if isinstance(first, torch.Tensor):
                return first
            raise UnsupportedError(
                "DefaultTLQBlockDataInterface expects the first element of block output "
                f"to be a Tensor, got {type(first).__name__}",
                action=_DEFAULT_HANDLER_ACTION,
            )
        if isinstance(block_output, torch.Tensor):
            return block_output
        raise UnsupportedError(
            "DefaultTLQBlockDataInterface cannot extract hidden states from block output "
            f"type {type(block_output).__name__}",
            action=_DEFAULT_HANDLER_ACTION,
        )

    def inject_hidden_states(self, block_input: BlockInput, hidden: torch.Tensor) -> None:
        args, _kwargs = block_input
        if not args:
            raise UnsupportedError(
                "DefaultTLQBlockDataInterface cannot inject hidden states into empty block args",
                action=_DEFAULT_HANDLER_ACTION,
            )
        first = args[0]
        if isinstance(first, torch.Tensor):
            args[0] = hidden
            return
        if isinstance(first, list) and len(first) > 0:
            first[0] = hidden
            return
        raise UnsupportedError(
            f"DefaultTLQBlockDataInterface cannot inject hidden states into args[0] of type {type(first).__name__}",
            action=_DEFAULT_HANDLER_ACTION,
        )

    def get_loss_mask(
        self,
        block_input: BlockInput,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        kwargs = block_input[1]
        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
            mask = kwargs["attention_mask"]
            # Keep Boolean mask on the same device as reconstructed activations / loss.
            out = (mask.squeeze(1)[:, -1] == 0) if mask.dim() == 4 else mask.bool()
            return out.to(device=hidden_states.device)
        if hidden_states.dim() < 2:
            raise UnsupportedError(
                "DefaultTLQBlockDataInterface fallback loss mask requires hidden_states "
                f"with at least 2 dims (B, S, ...), got shape {tuple(hidden_states.shape)}",
                action=_DEFAULT_HANDLER_ACTION,
            )
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        return torch.ones((batch_size, seq_len), dtype=torch.bool, device=hidden_states.device)


def resolve_tlq_block_data_interface(adapter: Optional[object] = None) -> TLQBlockDataInterface:
    """解析 TLQ block 数据接口：Adapter 若继承 ``TLQBlockDataInterface`` 则直接使用，否则 Default。"""
    if isinstance(adapter, TLQBlockDataInterface):
        return adapter
    return DefaultTLQBlockDataInterface()


__all__ = [
    "BlockInput",
    "BlockOutput",
    "DefaultTLQBlockDataInterface",
    "TLQBlockDataInterface",
    "propagate_outputs_to_inputs",
    "resolve_tlq_block_data_interface",
]
