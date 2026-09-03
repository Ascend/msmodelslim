#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
ConvertExecutor（convert_design.md §10）。

对每个已路由的 IRTask：
  1. ``ModelFreeModule.lazy_init`` 加载源权重；
  2. 按 ``RoutedTask.route`` 顺序调用 ``IIRTransformProcessor.transform``；
  3. 返回 ``IRResult`` 供 Application 写回虚拟树。

并行策略：
  - ``task_granularity=dependency_group``：共享 ``fused_from`` 或同一 safetensors shard 的任务同组；
  - ``ShardHandleCache``：组内复用 ``safe_open`` 句柄，避免重复 mmap 同一 shard；
  - ``worker_backend=process``：组间 ProcessPoolExecutor（纯 CPU），突破 GIL，适合计算瓶颈；
  - ``worker_backend=thread``：组内线程池（兼容旧行为，CPU 受 GIL 限制）；
  - ``max_inflight_bytes``：组内/组间并发任务的粗粒度内存上限。
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from typing import Iterator

from tqdm import tqdm

from msmodelslim.core.convert.catalog import DependencyMap, TensorCatalog
from msmodelslim.core.convert.device import effective_convert_workers
from msmodelslim.core.convert.protocol import ConvertContext, IConvertExecutor
from msmodelslim.core.convert.router import IRRouter
from msmodelslim.core.convert.tasks import IRResult, IRTask, RoutedTask
from msmodelslim.core.quant_service.modelslim_convert.impl.group_runner import (
    DependencyGroupRunner,
    estimate_task_bytes,
)
from msmodelslim.core.quant_service.modelslim_convert.impl.multi_npu import (
    MultiNpuConvertScheduler,
)
from msmodelslim.core.quant_service.modelslim_convert.impl.worker import (
    GroupWorkPayload,
    GroupWorkSummary,
    _init_worker,
    convert_dependency_group,
)
from msmodelslim.utils.logging import get_logger

logger = get_logger()

_GROUP_TIMING_LOG_THRESHOLD = 50
# Manager.Queue：0 表示不限制深度，避免 worker put 阻塞导致死锁
_RESULT_QUEUE_MAXSIZE = 0
_QUEUE_GET_TIMEOUT_S = 0.2


@dataclass
class _GroupTimingRecord:
    group_key: str
    task_count: int
    wall_s: float
    lazy_init_s: float
    transform_s: float
    lookup_s: float
    pool_wait_s: float
    fused_loads: int
    fused_hits: int
    shard_opens: int = 0
    shard_hits: int = 0


@dataclass
class _ConvertRunTiming:
    schedule_s: float = 0.0
    total_s: float = 0.0
    lazy_init_s: float = 0.0
    transform_s: float = 0.0
    lookup_s: float = 0.0
    group_wall_s: float = 0.0
    pool_wait_s: float = 0.0
    task_count: int = 0
    group_count: int = 0
    fused_loads: int = 0
    fused_hits: int = 0
    shard_opens: int = 0
    shard_hits: int = 0
    groups: list[_GroupTimingRecord] = field(default_factory=list)

    def add_group(self, record: _GroupTimingRecord) -> None:
        self.groups.append(record)
        self.lazy_init_s += record.lazy_init_s
        self.transform_s += record.transform_s
        self.lookup_s += record.lookup_s
        self.group_wall_s += record.wall_s
        self.pool_wait_s += record.pool_wait_s
        self.task_count += record.task_count
        self.group_count += 1
        self.fused_loads += record.fused_loads
        self.fused_hits += record.fused_hits
        self.shard_opens += record.shard_opens
        self.shard_hits += record.shard_hits


def _group_key(task: IRTask, dep_map: DependencyMap) -> str:
    """
    依赖组键：

    - 有 ``fused_from`` / catalog 依赖：同 fused 源一组（Qwen MoE fused 缓存）；
    - 否则按 ``inverse_weight_map`` 的 shard 分组：同 safetensors 文件的任务
      进一组，组内并行 + shard handle 缓存，避免 DeepSeek 等「一任务一组」的调度开销。
    """
    deps: set[str] = set()
    for ref in task.tensor_bindings.values():
        if ref.meta.get("fused_from"):
            deps.add(ref.meta["fused_from"])
        deps.update(dep_map.dependencies_of(ref.key))
    if deps:
        return "dep:" + "|".join(sorted(deps))
    shards = sorted(task.inverse_weight_map.keys())
    if shards:
        return "shard:" + "|".join(shards)
    return task.module_path


def _schedule_groups(routed_tasks: list[RoutedTask], dep_map: DependencyMap) -> list[list[RoutedTask]]:
    buckets: dict[str, list[RoutedTask]] = {}
    for rt in routed_tasks:
        buckets.setdefault(_group_key(rt.task, dep_map), []).append(rt)
    return list(buckets.values())


def _split_oversized_groups(
    groups: list[list[RoutedTask]],
    max_group_size: int | None,
) -> list[list[RoutedTask]]:
    """
    把任务数超过 ``max_group_size`` 的大组按任务切成多个子组。

    动机：MoE 一层的 experts（约 512 个 IR 任务）落在同一 dependency group，
    整组只能由单个进程承包，收尾阶段这些大组串行拖尾、多核大量空闲。
    切分后同层 experts 可分散到多个进程并行；各子组各自维护 fused 缓存，
    fused 源数量本就很少（按层计），重复加载代价可忽略。
    """
    if not max_group_size or max_group_size <= 0:
        return groups
    out: list[list[RoutedTask]] = []
    for group in groups:
        if len(group) <= max_group_size:
            out.append(group)
            continue
        for i in range(0, len(group), max_group_size):
            out.append(group[i : i + max_group_size])
    return out


def _short_group_label(group: list[RoutedTask], dep_map: DependencyMap) -> str:
    if not group:
        return ""
    key = _group_key(group[0].task, dep_map)
    if len(key) > 80:
        return key[:77] + "..."
    return key


def _estimate_group_bytes(group: list[RoutedTask], catalog: TensorCatalog | None) -> int:
    return sum(estimate_task_bytes(rt.task, catalog) for rt in group)


def _log_run_timing(stats: _ConvertRunTiming, backend: str) -> None:
    total = stats.total_s or 1.0
    logger.info(
        "ConvertExecutor timing summary (%s): total=%.2fs | schedule=%.3fs | "
        "lazy_init=%.2fs (%.1f%%) | transform=%.2fs (%.1f%%) | lookup=%.2fs (%.1f%%) | "
        "group_wall=%.2fs | pool_wait=%.2fs (%.1f%%) | "
        "tasks=%d groups=%d | fused_loads=%d fused_hits=%d | shard_opens=%d shard_hits=%d",
        backend,
        stats.total_s,
        stats.schedule_s,
        stats.lazy_init_s,
        100.0 * stats.lazy_init_s / total,
        stats.transform_s,
        100.0 * stats.transform_s / total,
        stats.lookup_s,
        100.0 * stats.lookup_s / total,
        stats.group_wall_s,
        stats.pool_wait_s,
        100.0 * stats.pool_wait_s / total,
        stats.task_count,
        stats.group_count,
        stats.fused_loads,
        stats.fused_hits,
        stats.shard_opens,
        stats.shard_hits,
    )
    if stats.groups:
        slowest = sorted(stats.groups, key=lambda g: g.wall_s, reverse=True)[:5]
        for rank, rec in enumerate(slowest, 1):
            logger.info(
                "ConvertExecutor slow group #%d: key=%r tasks=%d wall=%.2fs "
                "lazy_init=%.2fs transform=%.2fs fused=%d/%d shard=%d/%d",
                rank,
                rec.group_key,
                rec.task_count,
                rec.wall_s,
                rec.lazy_init_s,
                rec.transform_s,
                rec.fused_loads,
                rec.fused_hits,
                rec.shard_opens,
                rec.shard_hits,
            )


class ConvertExecutor(IConvertExecutor):
    def __init__(self, router: IRRouter | None = None) -> None:
        self._router = router or IRRouter.default()
        self._group_runner = DependencyGroupRunner(self._router)
        self._direct_worker_metas: list = []

    @property
    def direct_worker_metas(self) -> list:
        """worker 直接写盘后回传的 (staging_subdir, weight_map, desc_map)。"""
        return list(self._direct_worker_metas)

    def run(
        self,
        context: ConvertContext,
        routed_tasks: list[RoutedTask],
    ) -> Iterator[IRResult]:
        run_t0 = time.perf_counter()
        parallel = context.config.parallel
        dep_map = context.preprocess_result.dependency_map if context.preprocess_result is not None else DependencyMap()
        catalog = context.catalog

        schedule_t0 = time.perf_counter()
        groups = (
            [[rt] for rt in routed_tasks]
            if parallel.task_granularity == "ir_task"
            else _schedule_groups(routed_tasks, dep_map)
        )
        # 拆分 MoE 超大组，避免收尾阶段大组单进程串行拖尾、多核空闲。
        groups = _split_oversized_groups(groups, parallel.max_group_size)
        schedule_s = time.perf_counter() - schedule_t0
        run_stats = _ConvertRunTiming(schedule_s=schedule_s)

        backend = parallel.worker_backend
        process_workers = max(1, parallel.max_workers)
        if backend == "process":
            worker_threads = max(1, parallel.worker_threads or parallel.max_workers)
        else:
            worker_threads = effective_convert_workers(
                parallel.worker_threads or parallel.max_workers,
                context.resolved_worker_device,
                parallel.npu_max_workers,
            )
        logger.info(
            "Convert schedule: %d IR tasks, %d groups, backend=%s, process_workers=%d, worker_threads=%d",
            len(routed_tasks),
            len(groups),
            backend,
            process_workers,
            worker_threads,
        )

        budget = parallel.max_inflight_bytes
        pbar = tqdm(total=len(routed_tasks), desc="convert ir tasks")

        if context.parallel_mode == "npu_multi":
            yield from self._run_multinpu(
                context,
                groups,
                budget,
                pbar,
                run_stats,
            )
        elif backend == "process" and process_workers > 1:
            yield from self._run_multiprocess(
                context,
                groups,
                process_workers,
                worker_threads,
                budget,
                catalog,
                dep_map,
                pbar,
                run_stats,
            )
        else:
            thread_workers = effective_convert_workers(
                process_workers,
                context.resolved_worker_device,
                parallel.npu_max_workers,
            )
            for group in groups:
                yield from self._run_group_inprocess(
                    context,
                    group,
                    thread_workers,
                    budget,
                    catalog,
                    dep_map,
                    pbar,
                    run_stats,
                    return_mode="tensor_ref",
                )

        pbar.close()
        run_stats.total_s = time.perf_counter() - run_t0
        _log_run_timing(run_stats, context.parallel_mode)

    @staticmethod
    def _drain_result_queue(result_queue, pbar: tqdm, block: bool) -> Iterator[IRResult]:
        """从结果队列取出已完成的 IRResult；block 时用短 timeout 避免忙等。"""
        while True:
            try:
                if block:
                    result = result_queue.get(timeout=_QUEUE_GET_TIMEOUT_S)
                else:
                    result = result_queue.get_nowait()
            except queue.Empty:
                break
            yield result
            pbar.update(1)

    def _run_multiprocess(
        self,
        context: ConvertContext,
        groups: list[list[RoutedTask]],
        process_workers: int,
        worker_threads: int,
        budget: int | None,
        catalog: TensorCatalog | None,
        dep_map: DependencyMap,
        pbar: tqdm,
        run_stats: _ConvertRunTiming,
    ) -> Iterator[IRResult]:
        mp_ctx = mp.get_context("spawn")
        # 多进程回传必须走 state_dict，避免 pickle 自定义 nn.Module 子类。
        return_mode = "state_dict"
        pending: dict[object, tuple[list[RoutedTask], str, float]] = {}
        submitted = completed = 0
        total_tasks = sum(len(g) for g in groups)

        with mp_ctx.Manager() as manager:
            result_queue = manager.Queue(maxsize=_RESULT_QUEUE_MAXSIZE)
            with ProcessPoolExecutor(
                max_workers=process_workers,
                mp_context=mp_ctx,
                initializer=_init_worker,
                initargs=(result_queue,),
            ) as pool:
                while completed < len(groups) or pending:
                    inflight_bytes = sum(_estimate_group_bytes(group, catalog) for group, _, _ in pending.values())
                    while submitted < len(groups) and len(pending) < process_workers:
                        group = groups[submitted]
                        est = _estimate_group_bytes(group, catalog)
                        if budget is not None and pending and inflight_bytes + est > budget:
                            break
                        group_label = _short_group_label(group, dep_map)
                        payload = GroupWorkPayload(
                            model_path=str(context.model_path),
                            routed_tasks=group,
                            config=context.config,
                            worker_threads=max(1, worker_threads),
                            budget=budget,
                            return_mode=return_mode,
                        )
                        fut = pool.submit(convert_dependency_group, payload)
                        pending[fut] = (group, group_label, time.perf_counter())
                        submitted += 1
                        inflight_bytes += est

                    yield from self._drain_result_queue(result_queue, pbar, block=True)

                    if not pending:
                        continue

                    done, _ = wait(
                        set(pending.keys()),
                        timeout=_QUEUE_GET_TIMEOUT_S,
                        return_when=FIRST_COMPLETED,
                    )
                    for fut in done:
                        group, group_label, group_t0 = pending.pop(fut)
                        wait_t0 = time.perf_counter()
                        summary: GroupWorkSummary = fut.result()
                        pool_wait_s = time.perf_counter() - wait_t0
                        completed += 1
                        main_wall_s = time.perf_counter() - group_t0
                        run_stats.add_group(
                            _GroupTimingRecord(
                                group_key=group_label,
                                task_count=summary.task_count,
                                wall_s=main_wall_s,
                                lazy_init_s=summary.lazy_init_s,
                                transform_s=summary.transform_s,
                                lookup_s=summary.lookup_s,
                                pool_wait_s=pool_wait_s + summary.pool_wait_s,
                                fused_loads=summary.fused_loads,
                                fused_hits=summary.fused_hits,
                                shard_opens=summary.shard_opens,
                                shard_hits=summary.shard_hits,
                            )
                        )
                        if summary.task_count >= _GROUP_TIMING_LOG_THRESHOLD:
                            logger.info(
                                "ConvertExecutor process group done: key=%r tasks=%d "
                                "main_wall=%.2fs worker_wall=%.2fs "
                                "lazy_init=%.2fs transform=%.2fs fused=%d/%d shard=%d/%d",
                                group_label,
                                summary.task_count,
                                main_wall_s,
                                summary.worker_wall_s,
                                summary.lazy_init_s,
                                summary.transform_s,
                                summary.fused_loads,
                                summary.fused_hits,
                                summary.shard_opens,
                                summary.shard_hits,
                            )
                        yield from self._drain_result_queue(result_queue, pbar, block=True)

                # 收尾：不依赖 Queue.empty()（Manager 下不可靠），按 task 数 drain
                while pbar.n < total_tasks:
                    drained = 0
                    for result in self._drain_result_queue(result_queue, pbar, block=True):
                        yield result
                        drained += 1
                    if drained == 0:
                        break

        if pbar.n < total_tasks:
            logger.warning(
                "ConvertExecutor process: expected %d task results, got %d from queue",
                total_tasks,
                pbar.n,
            )

    def _run_multinpu(
        self,
        context: ConvertContext,
        groups: list[list[RoutedTask]],
        budget: int | None,
        pbar: tqdm,
        run_stats: _ConvertRunTiming,
    ) -> Iterator[IRResult]:
        from msmodelslim.core.quant_service.modelslim_convert.impl.save_adapter import (
            is_npu_direct_write,
        )

        direct_write = is_npu_direct_write(context.parallel_mode, context.config.dst_format)
        scheduler = MultiNpuConvertScheduler(
            model_path=str(context.model_path),
            config=context.config,
            device_indices=list(context.config.parallel.device_indices),
            groups=groups,
            save_path=str(context.save_path) if direct_write else None,
            budget=budget,
            # 直接写盘在本进程完成，保留 module 避免 PortableTensor 打包/还原。
            return_mode="module" if direct_write else "state_dict",
        )
        self._direct_worker_metas = []
        for result in scheduler.run():
            pbar.update(1)
            yield result
        self._direct_worker_metas = scheduler.worker_metas
        # 汇总各子进程回传的组摘要，供 timing summary 使用。
        for summary in scheduler.summaries:
            run_stats.add_group(
                _GroupTimingRecord(
                    group_key="npu_multi",
                    task_count=summary.task_count,
                    wall_s=summary.worker_wall_s,
                    lazy_init_s=summary.lazy_init_s,
                    transform_s=summary.transform_s,
                    lookup_s=summary.lookup_s,
                    pool_wait_s=summary.pool_wait_s,
                    fused_loads=summary.fused_loads,
                    fused_hits=summary.fused_hits,
                    shard_opens=summary.shard_opens,
                    shard_hits=summary.shard_hits,
                ),
            )

    def _run_group_inprocess(
        self,
        context: ConvertContext,
        group: list[RoutedTask],
        max_workers: int,
        budget: int | None,
        catalog: TensorCatalog | None,
        dep_map: DependencyMap,
        pbar: tqdm,
        run_stats: _ConvertRunTiming,
        return_mode: str,
    ) -> Iterator[IRResult]:
        group_label = _short_group_label(group, dep_map)
        group_t0 = time.perf_counter()
        results = self._group_runner.run_group(
            context=context,
            group=group,
            max_workers=max_workers,
            budget=budget,
            catalog=catalog,
            return_mode=return_mode,
        )
        group_wall_s = time.perf_counter() - group_t0
        stats = self._group_runner.last_stats
        run_stats.add_group(
            _GroupTimingRecord(
                group_key=group_label,
                task_count=stats.task_count,
                wall_s=group_wall_s,
                lazy_init_s=stats.lazy_init_s,
                transform_s=stats.transform_s,
                lookup_s=stats.lookup_s,
                pool_wait_s=stats.pool_wait_s,
                fused_loads=stats.fused_loads,
                fused_hits=stats.fused_hits,
                shard_opens=stats.shard_opens,
                shard_hits=stats.shard_hits,
            )
        )
        for result in results:
            yield result
            pbar.update(1)
        if stats.task_count >= _GROUP_TIMING_LOG_THRESHOLD:
            logger.info(
                "ConvertExecutor group timing: key=%r tasks=%d wall=%.2fs "
                "lazy_init=%.2fs transform=%.2fs pool_wait=%.2fs fused=%d/%d shard=%d/%d",
                group_label,
                stats.task_count,
                group_wall_s,
                stats.lazy_init_s,
                stats.transform_s,
                stats.pool_wait_s,
                stats.fused_loads,
                stats.fused_hits,
                stats.shard_opens,
                stats.shard_hits,
            )
