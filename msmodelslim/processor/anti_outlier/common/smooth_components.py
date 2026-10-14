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
# pylint: disable=consider-using-from-import,logging-not-lazy

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable, List
from enum import Enum
import torch.nn as nn

from msmodelslim.processor.anti_outlier.common.subgraph_type import (
    LinearLinearSubgraph,
    NonFusionSubgraph,
    NormLinearSubgraph,
    OVSubgraph,
)
from msmodelslim.utils.logging import get_logger
from msmodelslim.processor.anti_outlier.common.subgraph_type import UpDownSubgraph


class StatKey(str, Enum):
    STAT_KEY_MAX = "max"
    STAT_KEY_MIN = "min"
    STAT_KEY_SHIFT = "shift"
    STAT_KEY_THRESHOLD_CHANNEL = "thres_c"
    STAT_KEY_THRESHOLD_TENSOR = "thres_t"
    STAT_KEY_SMOOTH_SCALE_MASK = "smooth_scale_mask"
    STAT_KEY_SMOOTH_SCALE = "smooth_scale"
    STAT_KEY_VARIANCE = "std"
    TENSOR = 'tensor'
    STAT_KEY_MEAN = "mean"
    STAT_KEY_ARGS_AND_KWARGS = "args_and_kwargs"


class HookManager:
    def __init__(self, model: nn.Module):
        self.model = model
        self.hook_handles: List[Any] = []

    def install_hook(self, module_name: str, hook_fn: Callable, subgraph_type: str = None) -> bool:
        module = self.model.get_submodule(module_name)
        handle = module.register_forward_hook(hook_fn)
        self.hook_handles.append(handle)
        get_logger().debug(
            f"Successfully installed hook for module {module_name}"
            + (f" (subgraph_type: {subgraph_type})" if subgraph_type else "")
        )
        return True

    def install_pre_hook(
        self, module_name: str, hook_fn: Callable, subgraph_type: str = None, with_kwargs: bool = False
    ) -> bool:
        module = self.model.get_submodule(module_name)
        handle = module.register_forward_pre_hook(hook_fn, with_kwargs=with_kwargs)
        self.hook_handles.append(handle)
        get_logger().debug(
            f"Successfully installed pre-hook for module {module_name}"
            + (f" (subgraph_type: {subgraph_type})" if subgraph_type else "")
        )
        return True

    def remove_all_hooks(self) -> int:
        count = len(self.hook_handles)
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        get_logger().debug("Removed %d hooks", count)
        return count


class StatsCollector(ABC):
    def __init__(self):
        self.act_stats: Dict[str, Dict[str, Any]] = {}

    @abstractmethod
    def create_hook(self, name: str, subgraph_type: str = None) -> Callable:
        pass

    def clear_stats(self) -> None:
        self.act_stats.clear()


class SubgraphRegistry:
    SUBGRAPH_TO_NAME = {
        NormLinearSubgraph: "norm-linear",
        LinearLinearSubgraph: "linear-linear",
        OVSubgraph: "ov",
        UpDownSubgraph: "up-down",
        NonFusionSubgraph: "non-fusion",
    }

    NAME_TO_HANDLER = {
        "norm-linear": "_apply_norm_linear_smooth",
        "linear-linear": "_apply_linear_linear_smooth",
        "ov": "_apply_ov_smooth",
        "up-down": "_apply_up_down_smooth",
    }

    SUPPORTED_TYPES = ["norm-linear", "linear-linear", "ov", "up-down"]

    @classmethod
    def get_name(cls, subgraph_type: type) -> str:
        return cls.SUBGRAPH_TO_NAME.get(subgraph_type, "unknown")

    @classmethod
    def get_handler_name(cls, subgraph_name: str) -> Optional[str]:
        return cls.NAME_TO_HANDLER.get(subgraph_name)

    @classmethod
    def is_supported(cls, subgraph_name: str) -> bool:
        return subgraph_name in cls.SUPPORTED_TYPES

    @classmethod
    def get_all_supported_types(cls) -> list:
        return cls.SUPPORTED_TYPES.copy()
