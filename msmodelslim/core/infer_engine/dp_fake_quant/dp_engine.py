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

import os
import time
from typing import Any, List, Optional

import torch.distributed as dist
import torch.multiprocessing as mp

from msmodelslim.core.const import DeviceType
from msmodelslim.core.context import ContextManager, get_current_context
from msmodelslim.format.interface import IFormatLoader
from msmodelslim.utils.config import msmodelslim_config
from msmodelslim.utils.distributed import find_free_port, setup_distributed
from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.utils.logging import get_logger, logger_setter, set_logger_level

from ..fake_quant.engine import FakeQuantInferenceEngine
from ..interface import (
    FakeQuantInferenceInterface,
    InferenceConfig,
    InferenceResult,
)
from .dp_sharding import (
    FAKE_QUANT_INFER_NAMESPACE,
    global_indices_for_rank,
    merge_partials_into_context,
    partial_key,
    shard_samples_by_rank,
)


@logger_setter()
class DPFakeQuantInferenceEngine(FakeQuantInferenceEngine):
    """Multi-card fake-quant inference engine (sample DP).

    Inherits the single-card path (``_run_single`` / ``_resolve_device_str``) and adds
    multi-card dispatch via ``run``. Each device is spawned as a worker (``mp.spawn``);
    workers initialize ``torch.distributed`` so MoE adapters can shard experts across ranks
    *inside the model* during ``build_meta_model``. Sample sharding (DP) is round-robin;
    the parent merges partial results back to global sample order after workers finish.

    Whether expert parallelism actually happens is a property of the model, not the engine:
    for non-MoE models ``dist.is_initialized()`` simply has no effect on the build, so the
    flow degrades gracefully to pure sample DP.

    When ``len(samples)`` is not divisible by ``world_size``, ``shard_samples_by_rank`` pads
    by duplicating from the beginning (``DistributedSampler(drop_last=False)`` rule) so every
    rank receives at least one sample; padded results are discarded during merge.

    The entry instantiates this subclass only when ``len(device_indices) > 1``.
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
        if device_indices is None or len(device_indices) <= 1:
            return super().run(adapter, format_loader, inputs, inference_config, device, device_indices)
        return self._run_dp(adapter, format_loader, inputs, inference_config, device, device_indices)

    def _run_dp(
        self,
        adapter: FakeQuantInferenceInterface,
        format_loader: IFormatLoader,
        inputs: List[Any],
        inference_config: InferenceConfig,
        device: DeviceType,
        device_indices: List[int],
    ) -> InferenceResult:
        t_start = time.perf_counter()
        world_size = len(device_indices)
        num_samples = len(inputs)

        # Spawned workers cannot reuse the parent's format_loader (NPU tensors / open handles
        # are invalid post-spawn); each worker rebuilds a CPU loader from the export path.
        model_path = getattr(format_loader, "model_path", None)
        if not model_path:
            raise UnsupportedError(
                "Multi-card fake-quant requires format_loader.model_path to rebuild the loader in each worker",
                action="Provide an AscendV1 export directory as the model path.",
            )

        # Round-robin sharding with padding (DistributedSampler drop_last=False rule);
        # merge uses the matching global_indices_for_rank so shard + merge stay in lockstep.
        shards = [shard_samples_by_rank(inputs, rank, world_size) for rank in range(world_size)]

        shared_ctx = get_current_context()
        # Pre-touch so every worker and the parent resolve the same SharedNamespace before
        # writing / reading partial_{rank}.
        _ = shared_ctx[FAKE_QUANT_INFER_NAMESPACE]

        # Find a free port for HCCL/NCCL process group initialization.
        if "MASTER_PORT" not in os.environ:
            master_port = find_free_port()
            os.environ["MASTER_PORT"] = str(master_port)
        else:
            master_port = int(os.environ["MASTER_PORT"])

        backend = "hccl" if device == DeviceType.NPU else "nccl"

        mp.set_start_method("spawn", force=True)
        get_logger().info(
            "Fake-quant DP inference start: world_size=%d, samples=%d, device_indices=%s, backend=%s",
            world_size,
            num_samples,
            device_indices,
            backend,
        )
        mp.spawn(
            self._dp_worker,
            args=(
                world_size,
                device_indices,
                adapter,
                model_path,
                shards,
                inference_config,
                device,
                shared_ctx,
                backend,
                master_port,
                num_samples,
            ),
            nprocs=world_size,
            join=True,
        )
        merged = merge_partials_into_context(num_samples, world_size)
        get_logger().info(
            "Fake-quant DP inference merged %d samples back to global order in %.1fs",
            num_samples,
            time.perf_counter() - t_start,
        )
        return merged

    @staticmethod
    def _dp_worker(
        rank: int,
        world_size: int,
        device_indices: List[int],
        adapter: FakeQuantInferenceInterface,
        model_path: str,
        shards: List[List[Any]],
        inference_config: InferenceConfig,
        device: DeviceType,
        shared_ctx: Any,
        backend: str,
        master_port: int,
        num_samples: int,
    ) -> None:
        """One DP worker process (spawned by ``mp.spawn``; ``rank`` is auto-prepended).

        Initializes ``torch.distributed`` before the model build so MoE adapters can shard
        experts across ranks inside the model; each rank then runs the single-card inference
        path on its own sample shard. For non-MoE models the initialized group has no effect
        and the flow degrades to pure sample DP.
        """
        # Lazy import for format handling (see IFormatLoader rebuild below).
        from msmodelslim.format.format_handler import build_default_format_chain

        with ContextManager(ctx=shared_ctx):
            # Spawned worker starts at the configured default level (CLI -v/-q are not
            # propagated across processes); keep the previous behaviour.
            set_logger_level(msmodelslim_config.env_vars.log_level)

            # Init the process group BEFORE model build so ep_size / expert range see the rank.
            actual_device_idx = device_indices[rank]
            setup_distributed(rank, world_size, backend, device_index=actual_device_idx, master_port=master_port)
            get_logger().info(
                "Fake-quant DP worker rank=%d/%d on npu:%d (torch.distributed ready, backend=%s)",
                rank,
                world_size,
                actual_device_idx,
                backend,
            )

            # Rebuild a fresh CPU format loader in this process, then run the single-card path
            # (inherited from FakeQuantInferenceEngine) on this rank's shard.
            format_loader = build_default_format_chain().handle(model_path, device="cpu")
            engine = FakeQuantInferenceEngine()
            result = engine._run_single(
                adapter=adapter,
                format_loader=format_loader,
                inputs=shards[rank],
                inference_config=inference_config,
                device=device,
                device_indices=[device_indices[rank]],
                global_sample_indices=global_indices_for_rank(rank, num_samples, world_size),
            )

            # Hand this rank's result to the parent via the shared context; the parent merges
            # all partial_{rank} back to global sample order after spawn join.
            state = get_current_context()[FAKE_QUANT_INFER_NAMESPACE].state
            state[partial_key(rank)] = result.model_dump()
            get_logger().info(
                "Fake-quant DP worker rank=%d/%d done: %d local sample(s) staged to partial_%d",
                rank,
                world_size,
                len(shards[rank]),
                rank,
            )

            if dist.is_initialized():
                dist.destroy_process_group()
