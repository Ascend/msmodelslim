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

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Optional

import torch
from torch import nn

from msmodelslim.core.context.interface import IContext, IValidatedState, get_current_context
from msmodelslim.processor.anti_outlier.awq.best_scales_search import AWQSearcher

AWQ_CONTEXT_NAMESPACE = "awq"


def get_global_awq_stats() -> IValidatedState:
    global_context: Optional[IContext] = get_current_context()
    if global_context is None:
        raise RuntimeError("No global context found when trying to get AWQ context.")
    awq_namespace = global_context[AWQ_CONTEXT_NAMESPACE]  # pylint: disable=unsubscriptable-object
    if awq_namespace is None:
        raise RuntimeError("No AWQ context namespace found in the global context.")
    return awq_namespace.state


@dataclass
class AWQConfig:
    """AWQ algorithm configuration.

    Attributes:
        version: Config version for dispatch.
        awq_searcher: The AWQSearcher instance to use for scale searching.
    """

    version: int
    awq_searcher: AWQSearcher


@dataclass
class AWQContext:
    """Runtime context carrying pre-collected statistics for AWQ.

    Attributes:
        act_mean: Per-channel activation mean tensor (absolute values).
        inspect_module: The module being used for loss computation.
        inspect_module_args: Cached intermediate arguments for the inspect module.
    """

    act_mean: torch.Tensor
    inspect_module: nn.Module
    inspect_module_args: Any


def offload(data: Any, device: Optional[torch.device] = torch.device("cpu")) -> Any:
    """Recursively move all tensors in a value tree to ``device``."""
    if device is None:
        return data
    if isinstance(data, torch.Tensor):
        return data.to(device) if device else data
    if isinstance(data, list):
        return [offload(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(offload(v, device) for v in data)
    if isinstance(data, dict):
        return {k: offload(v, device) for k, v in data.items()}
    if is_dataclass(data):
        for field in fields(data):
            v = getattr(data, field.name)
            setattr(data, field.name, offload(v, device))
        return data
    return data


def onload(data: Any, device: Optional[torch.device] = None) -> Any:
    """Recursively move all WrapperValue-wrapped tensors in a value tree to their onload device."""
    if isinstance(data, torch.Tensor):
        return data.to(device) if device else data
    if isinstance(data, list):
        return [onload(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(onload(v, device) for v in data)
    if isinstance(data, dict):
        return {k: onload(v, device) for k, v in data.items()}
    if is_dataclass(data):
        for field in fields(data):
            v = getattr(data, field.name)
            setattr(data, field.name, onload(v, device))
        return data
    return data
