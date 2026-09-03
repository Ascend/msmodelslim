#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
单个 dependency group 内的 IR 任务执行（线程池 + fused cache）。

供主进程 ``ConvertExecutor`` 与子进程 ``convert_dependency_group`` 共用。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field

import torch
from torch import nn

from msmodelslim.core.convert.catalog import TensorCatalog
from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.convert.router import IRRouter
from msmodelslim.core.convert.tasks import (
    IRResult,
    IRTask,
    PortableTensor,
    RoutedTask,
    is_mxfp8_deploy_state,
)
from msmodelslim.core.quant_service.modelslim_convert.virtual_module import (
    ModelFreeModule,
)
from msmodelslim.core.quant_service.modelslim_convert.weight_mapping.fused_cache import (
    FusedTensorCache,
)
from msmodelslim.infra.io.shard_handle_cache import ShardHandleCache


def estimate_task_bytes(task: IRTask, catalog: TensorCatalog | None) -> int:
    """按 float16 粗算单任务权重大小，用于 ``max_inflight_bytes`` 限流。"""
    total = 0
    for ref in task.tensor_bindings.values():
        shape = ref.shape
        if not shape and catalog is not None:
            entry = catalog.get(ref.key)
            shape = entry.shape if entry else ()
        if shape:
            n = 1
            for d in shape:
                n *= int(d)
            total += n * 2
    return max(total, 1)


@dataclass
class TaskTiming:
    lazy_init_s: float = 0.0
    transform_s: float = 0.0
    lookup_s: float = 0.0


@dataclass
class GroupRunStats:
    lazy_init_s: float = 0.0
    transform_s: float = 0.0
    lookup_s: float = 0.0
    pool_wait_s: float = 0.0
    task_count: int = 0
    fused_loads: int = 0
    fused_hits: int = 0
    shard_opens: int = 0
    shard_hits: int = 0


@dataclass
class _TimingCollector:
    lock: threading.Lock = field(default_factory=threading.Lock)
    lazy_init_s: float = 0.0
    transform_s: float = 0.0
    lookup_s: float = 0.0
    task_count: int = 0

    def record(self, timing: TaskTiming) -> None:
        with self.lock:
            self.lazy_init_s += timing.lazy_init_s
            self.transform_s += timing.transform_s
            self.lookup_s += timing.lookup_s
            self.task_count += 1


def prepare_result(result: IRResult, return_mode: str) -> IRResult:
    """
    多进程回传时将 module 转为 CPU state_dict 并把每个 tensor 包成 ``PortableTensor``，
    使其按值（纯 bytes）pickle，避免 torch 共享内存/mmap 导致的 ``Cannot allocate memory``。
    """
    if return_mode != "state_dict" or result.module is None:
        return result
    state_dict = result.module.state_dict()
    is_mxfp8 = is_mxfp8_deploy_state(state_dict)
    packed = {}
    for key, value in state_dict.items():
        if is_mxfp8 and key == "weight" and value.dtype == torch.bfloat16 and value.ndim == 2:
            # MXFP8 量化值精确落在 fp8 网格上（bf16->fp8->bf16 无损），
            # 提前转 fp8 可让跨进程结果队列负载减半（AscendV1 落盘时同样转 fp8）。
            # 需先回 CPU 再转：torch_npu 对 fp8 的 dtype 转换（aclnnInplaceCopy）不可用。
            value = value.detach().cpu().to(torch.float8_e4m3fn)
        packed[key] = PortableTensor.from_tensor(value)
    return IRResult(
        module_path=result.module_path,
        final_ir=result.final_ir,
        module=None,
        state_dict=packed,
        loss_level=result.loss_level,
        route_ir_names=result.route_ir_names,
    )


class DependencyGroupRunner:
    """在单进程内执行一个 dependency group 的全部 IR 任务。"""

    def __init__(self, router: IRRouter) -> None:
        self._router = router
        self._last_stats = GroupRunStats()
        self._last_wall_s = 0.0

    def run_group(
        self,
        context: ConvertContext,
        group: list[RoutedTask],
        max_workers: int,
        budget: int | None,
        catalog: TensorCatalog | None,
        return_mode: str,
        result_sink: Callable[[IRResult], None] | None = None,
    ) -> list[IRResult]:
        group_t0 = time.perf_counter()
        pool_wait_s = 0.0
        reader = context.reader
        fused_cache = FusedTensorCache()
        shard_cache = ShardHandleCache(max_shards=context.config.parallel.shard_cache_size)
        collector = _TimingCollector()
        results: list[IRResult] = []

        def _emit(result: IRResult) -> None:
            prepared = prepare_result(result, return_mode)
            if result_sink is not None:
                result_sink(prepared)
            else:
                results.append(prepared)

        had_fused_attr = reader is not None and hasattr(reader, "fused_tensor_cache")
        had_shard_attr = reader is not None and hasattr(reader, "shard_handle_cache")
        if reader is not None:
            reader.fused_tensor_cache = fused_cache
            reader.shard_handle_cache = shard_cache
        try:
            if max_workers <= 1:
                for rt in group:
                    result, timing = self._run_one(context, rt)
                    collector.record(timing)
                    _emit(result)
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    pending: dict[object, int] = {}
                    submitted = completed = 0
                    while completed < len(group):
                        while submitted < len(group) and len(pending) < max_workers:
                            if budget is not None and pending:
                                used = sum(pending.values())
                                est = estimate_task_bytes(group[submitted].task, catalog)
                                if used + est > budget:
                                    break
                            rt = group[submitted]
                            est = estimate_task_bytes(rt.task, catalog)
                            pending[pool.submit(self._run_one, context, rt)] = est
                            submitted += 1
                        if not pending:
                            break
                        wait_t0 = time.perf_counter()
                        done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
                        pool_wait_s += time.perf_counter() - wait_t0
                        for fut in done:
                            pending.pop(fut)
                            result, timing = fut.result()
                            collector.record(timing)
                            _emit(result)
                            completed += 1
        finally:
            if reader is not None:
                if had_fused_attr:
                    reader.fused_tensor_cache = None
                else:
                    try:
                        delattr(reader, "fused_tensor_cache")
                    except AttributeError:
                        pass
                if had_shard_attr:
                    reader.shard_handle_cache = None
                else:
                    try:
                        delattr(reader, "shard_handle_cache")
                    except AttributeError:
                        pass
            fused_cache.clear()
            shard_cache.clear()

        self._last_stats = GroupRunStats(
            lazy_init_s=collector.lazy_init_s,
            transform_s=collector.transform_s,
            lookup_s=collector.lookup_s,
            pool_wait_s=pool_wait_s,
            task_count=collector.task_count,
            fused_loads=fused_cache.misses,
            fused_hits=fused_cache.hits,
            shard_opens=shard_cache.opens,
            shard_hits=shard_cache.hits,
        )
        self._last_wall_s = time.perf_counter() - group_t0
        return results

    @property
    def last_stats(self) -> GroupRunStats:
        return getattr(self, "_last_stats", GroupRunStats())

    @property
    def last_wall_s(self) -> float:
        return getattr(self, "_last_wall_s", 0.0)

    def _run_one(self, context: ConvertContext, routed: RoutedTask) -> tuple[IRResult, TaskTiming]:
        timing = TaskTiming()
        lookup_t0 = time.perf_counter()
        tree = context.virtual_tree
        if tree is not None:
            mod = tree.get_submodule(routed.task.module_path)
        else:
            mod = routed.task.create_empty_module()
        timing.lookup_s = time.perf_counter() - lookup_t0

        if isinstance(mod, ModelFreeModule) and not mod.lazy_initialized:
            if context.reader is None:
                raise RuntimeError(f"checkpoint reader is required to lazy_init {routed.task.module_path}")
            lazy_t0 = time.perf_counter()
            mod.lazy_init(context.reader, device=context.resolved_worker_device)
            timing.lazy_init_s = time.perf_counter() - lazy_t0

        current: nn.Module = mod
        transform_t0 = time.perf_counter()
        for edge in routed.route:
            current = self._router.get_processor(edge.processor_name).transform(current, context)
        timing.transform_s = time.perf_counter() - transform_t0

        loss = "lossy" if any(e.loss_level.value == "lossy" for e in routed.route) else "lossless"
        result = IRResult(
            module_path=routed.task.module_path,
            final_ir=routed.task.target_ir,
            module=current,
            route_ir_names=routed.route_ir_names,
            loss_level=loss,
        )
        return result, timing
