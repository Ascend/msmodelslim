#!/usr/bin/env python
# -*- coding: UTF-8 -*-

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

from typing import Any, Dict, Optional, Tuple

import torch

from msmodelslim.utils.exception import UnsupportedError

from .interface import MSEModelWiseAnalysisInterface

_DEFAULT_HANDLER_ACTION = (
    "Block I/O does not match DefaultMSEModelWiseBlockData; "
    "have the model adapter inherit MSEModelWiseAnalysisInterface and implement "
    "extract_hidden_states."
)

_HIDDEN_STATE_KEYS = (
    "hidden_states",
    "last_hidden_state",
    "pooler_output",
)


def _tensor_from_mapping_or_object(obj: Any, keys: Tuple[str, ...]) -> Optional[torch.Tensor]:
    for key in keys:
        if isinstance(obj, dict) and key in obj:
            value = obj[key]
            if isinstance(value, torch.Tensor):
                return value
        if hasattr(obj, key):
            value = getattr(obj, key)
            if isinstance(value, torch.Tensor):
                return value
    return None


def _is_forward_row(value: Any) -> bool:
    return isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict)


class DefaultMSEModelWiseBlockData(MSEModelWiseAnalysisInterface):
    """mse_model_wise 默认 block I/O：tensor / tuple、Transformers ModelOutput、常见 dict 字段。"""

    def extract_hidden_states(self, value: Any) -> torch.Tensor:
        if _is_forward_row(value):
            return self._extract_from_forward_row(value)
        return self._extract_from_block_output(value)

    def _extract_from_forward_row(self, row: Tuple[tuple, Dict[str, Any]]) -> torch.Tensor:
        args, _kwargs = row
        if not isinstance(args, tuple) or not args:
            raise UnsupportedError(
                "DefaultMSEModelWiseBlockData cannot extract hidden states from empty block args",
                action=_DEFAULT_HANDLER_ACTION,
            )
        first = args[0]
        if isinstance(first, torch.Tensor):
            return first
        if isinstance(first, tuple) and first and isinstance(first[0], torch.Tensor):
            return first[0]
        raise UnsupportedError(
            "DefaultMSEModelWiseBlockData cannot extract hidden states from block args "
            f"with args[0] type {type(first).__name__}",
            action=_DEFAULT_HANDLER_ACTION,
        )

    def _extract_from_block_output(self, block_output: Any) -> torch.Tensor:
        tensor = _tensor_from_mapping_or_object(block_output, _HIDDEN_STATE_KEYS)
        if tensor is not None:
            return tensor

        if isinstance(block_output, (tuple, list)):
            if not block_output:
                raise UnsupportedError(
                    "DefaultMSEModelWiseBlockData cannot extract hidden states from an empty "
                    f"{type(block_output).__name__} block output",
                    action=_DEFAULT_HANDLER_ACTION,
                )
            first = block_output[0]
            if isinstance(first, torch.Tensor):
                return first
            nested = _tensor_from_mapping_or_object(first, _HIDDEN_STATE_KEYS)
            if nested is not None:
                return nested
            raise UnsupportedError(
                "DefaultMSEModelWiseBlockData expects the first element of block output "
                f"to be a Tensor, got {type(first).__name__}",
                action=_DEFAULT_HANDLER_ACTION,
            )
        if isinstance(block_output, torch.Tensor):
            return block_output
        raise UnsupportedError(
            "DefaultMSEModelWiseBlockData cannot extract hidden states from block output "
            f"type {type(block_output).__name__}",
            action=_DEFAULT_HANDLER_ACTION,
        )


def resolve_mse_model_wise_block_data(adapter: Optional[object] = None) -> MSEModelWiseAnalysisInterface:
    """Adapter 若实现 ``MSEModelWiseAnalysisInterface`` 则用之，否则走默认实现。"""
    if isinstance(adapter, MSEModelWiseAnalysisInterface):
        return adapter
    return DefaultMSEModelWiseBlockData()
