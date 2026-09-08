#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.
Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

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

from typing import TYPE_CHECKING, Any, Callable, List, Optional

from torch import nn

from msmodelslim.core.graph import AdapterConfig
from msmodelslim.model.base import BaseModelAdapter
from msmodelslim.processor.quant.fa3.interface import FA3QuantAdapterInterface
from msmodelslim.processor.quarot import OnlineQuaRotInterface
from ..interface_hub import (
    IterSmoothInterface,
    OASQInterface,
)

if TYPE_CHECKING:
    from .base_model_adapter import Wan2_2BaseModelAdapter


class Wan2_2ExpertSubAdapter(
    BaseModelAdapter,
    OnlineQuaRotInterface,
    FA3QuantAdapterInterface,
    IterSmoothInterface,
    OASQInterface,
):
    """
    Wan2.2 内部 expert 子适配器（不注册 model_type）。

    仅用于 LayerWiseRunner 的 per-expert 调度；场景级逻辑由 Wan2_2BaseModelAdapter 承担，
    本类通过 parent 委托 generate_model_forward / generate_model_visit 等默认实现。
    """

    # pylint: disable=super-init-not-called
    def __init__(self, parent: "Wan2_2BaseModelAdapter", expert_name: str):
        self._parent = parent
        self.expert_name = expert_name
        self._module: Optional[nn.Module] = None

    def bind_module(self, module: nn.Module) -> None:
        self._module = module

    def __getattr__(self, item: str):
        return getattr(self._parent, item)

    def quantization_context(self):
        return self._parent._quantization_context_with_no_sync(self._module)

    def generate_model_forward(self, model: nn.Module, inputs: Any):
        return self._parent.generate_model_forward(model, inputs)

    def generate_model_visit(self, model: nn.Module):
        return self._parent.generate_model_visit(model)

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        self._parent.enable_kv_cache(model, need_kv_cache)

    def get_online_rotation_configs(self, model: Optional[nn.Module] = None):
        """LayerWiseRunner 按 expert 量化时须满足 OnlineQuaRotInterface。"""
        target = model if model is not None else self._module
        return self._parent.get_online_rotation_configs(target)

    def inject_fa3_placeholders(
        self,
        root_name: str,
        root_module: nn.Module,
        should_inject: Callable[[str], bool],
    ) -> None:
        self._parent.inject_fa3_placeholders(root_name, root_module, should_inject)

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        return self._parent.get_adapter_config_for_subgraph(self._module.num_layers)


class Wan2_2LowNoiseSubAdapter(Wan2_2ExpertSubAdapter):  # pylint: disable=too-many-ancestors
    """low_noise_model 默认子适配器（可按需重写 forward/visit/context）。"""


class Wan2_2HighNoiseSubAdapter(Wan2_2ExpertSubAdapter):  # pylint: disable=too-many-ancestors
    """high_noise_model 默认子适配器（可按需重写 forward/visit/context）。"""
