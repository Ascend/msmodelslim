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

from typing import Callable, Optional
from collections.abc import MutableMapping
import inspect

import torch
from torch import nn

from msmodelslim.core.context.interface import IValidatedState
from msmodelslim.core.quantization_algorithm.anti_outlier.awq.common import get_global_awq_stats, offload

from ..common import HookManager
from msmodelslim.core.quantization_algorithm.anti_outlier.common.smooth_components import StatKey


class AWQStatsCollector:
    """Collects activation statistics (means and block kwargs) for AWQ."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.global_stats: IValidatedState = get_global_awq_stats()
        self.hook_manager = HookManager(model=model)
        self._kwargs_installed_module_names: set[str] = set()

    def observe_activation(self, module_name: str) -> None:
        hook_fn = self._make_mean_hook(module_name)
        self.hook_manager.install_hook(module_name, hook_fn)

    def observe_kwargs(self, module_name: str) -> None:
        if module_name in self._kwargs_installed_module_names:
            return
        hook_fn = self._make_kwargs_hook(module_name)
        self.hook_manager.install_pre_hook(module_name, hook_fn, with_kwargs=True)
        self._kwargs_installed_module_names.add(module_name)

    def get_activation_mean(self, module_name: str) -> Optional[torch.Tensor]:
        stat: Optional[MutableMapping] = self.global_stats.get(module_name)
        if stat is None:
            return None
        mean_tuple = stat.get(StatKey.STAT_KEY_MEAN)
        return mean_tuple[0] if mean_tuple else None

    def get_block_kwargs(self, module_name: str) -> Optional[list]:
        stat: Optional[MutableMapping] = self.global_stats.get(module_name)
        if stat is None:
            return None
        return stat.get(StatKey.STAT_KEY_ARGS_AND_KWARGS)

    def stop_observing(self) -> None:
        self.hook_manager.remove_all_hooks()

    def clear_stats(self) -> None:
        self.global_stats.clear()

    def _make_mean_hook(self, module_name: str) -> Callable:
        def activation_mean_hook(
            module: nn.Module,
            args: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ):
            activations = args[0].abs().detach()
            flat = activations.flatten(0, -2)
            current_count = flat.shape[0]
            current_mean = flat.mean(dim=0)

            if module_name not in self.global_stats:
                self.global_stats[module_name] = {}

            stat_entry: MutableMapping = self.global_stats[module_name]

            if StatKey.STAT_KEY_MEAN not in stat_entry:
                stat_entry[StatKey.STAT_KEY_MEAN] = (current_mean.cpu(), current_count)
            else:
                old_mean, old_count = stat_entry[StatKey.STAT_KEY_MEAN]
                old_mean = old_mean.to(current_mean.device)
                new_count = old_count + current_count
                new_mean = (old_mean * old_count + current_mean * current_count) / new_count
                stat_entry[StatKey.STAT_KEY_MEAN] = (new_mean.cpu(), new_count)

        return activation_mean_hook

    def _make_kwargs_hook(self, module_name: str) -> Callable:
        def parent_kwargs_hook(module: nn.Module, args: tuple[torch.Tensor, ...], kwargs):
            values = inspect.signature(module.forward).bind(*args, **kwargs)
            if module_name not in self.global_stats:
                self.global_stats[module_name] = {}
            stat_entry: MutableMapping = self.global_stats[module_name]
            if StatKey.STAT_KEY_ARGS_AND_KWARGS not in stat_entry:
                stat_entry[StatKey.STAT_KEY_ARGS_AND_KWARGS] = []
            stat_entry[StatKey.STAT_KEY_ARGS_AND_KWARGS].append(offload(values.arguments))

        return parent_kwargs_hook
