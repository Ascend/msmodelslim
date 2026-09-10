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

import time
from typing import Any, List, Optional

import torch
import torch.distributed as dist

from msmodelslim.core.const import DeviceType
from msmodelslim.format.interface import IFormatLoader
from msmodelslim.utils.buffer import RuntimeBufferStore
from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.utils.logging import get_logger, logger_setter

from ..interface import (
    FakeQuantInferenceInterface,
    IInferenceEngine,
    InferenceConfig,
    InferenceResult,
)
from .prefill_loop import PrefillLoop
from .session import Session
from .weight_manager import WeightManager


@logger_setter()
class FakeQuantInferenceEngine(IInferenceEngine):
    """Single-card fake-quant inference engine.

    Wires ``_run_single``: Session builds the empty-weights shell, WeightManager installs
    per-layer hydrate / device-move / offload hooks, then PrefillLoop drives native
    ``model.generate`` (use_cache=False, multi-prefill semantics).

    Multi-card (sample data parallelism) lives in ``DPFakeQuantInferenceEngine``; ``run``
    raises ``UnsupportedError`` when ``len(device_indices) > 1`` so misuse fails loudly.
    """

    def run(
        self,
        adapter: FakeQuantInferenceInterface,
        format_loader: IFormatLoader,
        inputs: List[Any],
        inference_config: InferenceConfig,
        device: DeviceType = DeviceType.NPU,
        device_indices: Optional[List[int]] = None,
    ) -> InferenceResult:
        if device_indices is not None and len(device_indices) > 1:
            raise UnsupportedError(
                "FakeQuantInferenceEngine is single-card only; multi-card requires DPFakeQuantInferenceEngine",
                action="Use multiple --device_id entries (the DP engine is then used) or a single card.",
            )
        return self._run_single(adapter, format_loader, inputs, inference_config, device, device_indices)

    def _run_single(
        self,
        adapter: FakeQuantInferenceInterface,
        format_loader: IFormatLoader,
        inputs: List[Any],
        inference_config: InferenceConfig,
        device: DeviceType,
        device_indices: Optional[List[int]],
        global_sample_indices: Optional[List[int]] = None,
    ) -> InferenceResult:
        device_str = self._resolve_device_str(device, device_indices)

        buffer_store = RuntimeBufferStore()
        try:
            session = Session(
                adapter,
                format_loader,
                buffer_store,
            )
            t_build = time.perf_counter()
            model = session.build()
            get_logger().info("Fake-quant shell built in %.1fs", time.perf_counter() - t_build)

            tokenized_inputs = list(adapter.handle_dataset(inputs, DeviceType.CPU))

            weight_manager = WeightManager(
                format_loader,
                device=device_str,
                buffer_store=buffer_store,
                offload_device="meta",
            )
            weight_manager.install_hooks(model)
            try:
                loop = PrefillLoop(adapter, model)
                # Multi-card DP: all ranks must make the same number of forward passes to keep
                # the MoE collectives synchronized, so disable EOS stopping; PrefillLoop
                # truncates each output at the first EOS token afterwards.
                disable_eos = dist.is_initialized() and dist.get_world_size() > 1
                return loop.run(
                    tokenized_inputs,
                    inference_config.max_new_tokens,
                    disable_eos=disable_eos,
                    global_sample_indices=global_sample_indices,
                )
            finally:
                weight_manager.remove_all_hooks()
        finally:
            buffer_store.clear()

    @staticmethod
    def _resolve_device_str(device: DeviceType, device_indices: Optional[List[int]]) -> str:
        if device == DeviceType.NPU:
            idx = device_indices[0] if device_indices else 0
            if hasattr(torch, "npu"):
                torch.npu.set_device(f"npu:{idx}")
            return f"npu:{idx}"
        # CPU or anything else: use the enum value as-is.
        return device.value if hasattr(device, "value") else str(device)
