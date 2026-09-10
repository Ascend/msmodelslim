# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You may use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

from torch import nn

from msmodelslim.format.interface import IFormatLoader
from msmodelslim.utils.buffer import RuntimeBufferStore, collect_nonpersistent_buffers
from msmodelslim.utils.logging import get_logger

from ..interface import FakeQuantInferenceInterface


class Session:
    """Build the fake-quant inference shell once and leave it ready for layer-wise hydrate.

    Session owns the one-time setup: empty-weights shell, FakeQuant IR over the full tree,
    shared-module hydrate onto CPU, and a snapshot of non-persistent buffers (RoPE
    ``inv_freq`` etc.) into ``RuntimeBufferStore``. It does NOT install hooks or run any
    forward; live per-layer weight management is delegated to ``WeightManager``.
    """

    def __init__(
        self,
        adapter: FakeQuantInferenceInterface,
        format_loader: IFormatLoader,
        buffer_store: RuntimeBufferStore,
    ) -> None:
        self.adapter = adapter
        self.format_loader = format_loader
        self.buffer_store = buffer_store

    def build(self) -> nn.Module:
        """Build the shell, apply IR on the full tree, hydrate shared modules, snapshot buffers."""
        get_logger().info("Building fake-quant inference shell and hydrating shared weights")
        model = self.adapter.build_meta_model()
        model.eval()

        # Snapshot non-persistent buffers before any decoder .to(meta) can drop them.
        self.buffer_store.store(collect_nonpersistent_buffers(model))

        # IR is applied to the full tree in one pass (meta FakeQuant modules cost nothing).
        self.format_loader.bind_adapter(self.adapter)
        self.format_loader.apply_ir(model)

        # Hydrate shared modules only; decoder layers are hydrated per-layer by WeightManager.
        decoder_prefixes = self.format_loader.decoder_prefixes(model)
        self.format_loader.hydrate(model, skip_prefixes=decoder_prefixes)
        self.buffer_store.restore(model, skip_prefixes=decoder_prefixes)
        get_logger().info("Fake-quant inference shell ready (IR applied, shared hydrated)")
        return model
