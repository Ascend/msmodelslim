#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Convert 子进程 worker：在独立进程中执行一个 dependency group 内的 IR 任务。

ProcessPool 调度粒度为 dependency group，组内仍用 ThreadPoolExecutor + FusedTensorCache；
多进程路径固定纯 CPU，不涉及 NPU。结果经进程级 ``result_queue`` 逐条流式回传，
避免一次性 pickle 大量 nn.Module。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from msmodelslim.core.convert.config import ConvertConfig
from msmodelslim.core.convert.tasks import IRResult, RoutedTask
from msmodelslim.core.quant_service.modelslim_convert.impl.group_runner import (
    DependencyGroupRunner,
    GroupRunStats,
)

if TYPE_CHECKING:
    from msmodelslim.core.convert.protocol import ConvertContext

_PROCESS_DEVICE = "cpu"
# 由 ProcessPoolExecutor.initializer 注入；spawn 下不可通过 payload pickle 传递 Queue
_RESULT_QUEUE: Any | None = None


def _init_worker(result_queue: Any) -> None:
    """子进程启动时绑定主进程创建的 Manager.Queue 代理。

    同时把单个 torch 算子限制为单线程：多进程 + 组内线程已提供并行度，
    若再让每个 torch 算子开满 OpenMP 线程会造成线程超订（oversubscription），
    总线程数远超物理核数而拖慢计算。并行度统一交由 workers × worker_threads 控制。
    """
    global _RESULT_QUEUE
    _RESULT_QUEUE = result_queue

    import torch

    torch.set_num_threads(1)


@dataclass(frozen=True)
class GroupWorkPayload:
    """跨进程传递的最小工作单元（须可 pickle，不含 Queue）。"""

    model_path: str
    routed_tasks: list[RoutedTask]
    config: ConvertConfig
    worker_threads: int
    budget: int | None
    return_mode: str


@dataclass(frozen=True)
class GroupWorkSummary:
    """子进程完成一组任务后的摘要（不含 tensor payload）。"""

    task_count: int
    fused_loads: int
    fused_hits: int
    shard_opens: int = 0
    shard_hits: int = 0
    lazy_init_s: float = 0.0
    transform_s: float = 0.0
    lookup_s: float = 0.0
    pool_wait_s: float = 0.0
    worker_wall_s: float = 0.0

    @classmethod
    def from_runner(cls, runner: DependencyGroupRunner) -> "GroupWorkSummary":
        """从 ``DependencyGroupRunner`` 最近一次 ``run_group`` 的统计构造摘要。"""
        stats: GroupRunStats = runner.last_stats
        return cls(
            task_count=stats.task_count,
            fused_loads=stats.fused_loads,
            fused_hits=stats.fused_hits,
            shard_opens=stats.shard_opens,
            shard_hits=stats.shard_hits,
            lazy_init_s=stats.lazy_init_s,
            transform_s=stats.transform_s,
            lookup_s=stats.lookup_s,
            pool_wait_s=stats.pool_wait_s,
            worker_wall_s=runner.last_wall_s,
        )


def bootstrap_convert_worker(
    model_path: str,
    config: ConvertConfig,
    device: str,
) -> tuple[ConvertContext, DependencyGroupRunner]:
    """
    子进程内构造 reader / router / Context / GroupRunner。

    CPU ProcessPool 与 npu_multi spawn 共用；调用方只负责设备串与调度循环。
    返回的 Context 类型为 ``ConvertContext``（延迟导入，避免父进程提前拉起 IO 栈）。
    """
    from msmodelslim.core.convert.protocol import ConvertContext
    from msmodelslim.infra.io.checkpoint_reader import CheckpointReader
    from msmodelslim.processor.convert.registry import register_convert_processors

    router = register_convert_processors()
    reader = CheckpointReader(model_path)
    context = ConvertContext(config=config, reader=reader)
    context.resolved_worker_device = device
    return context, DependencyGroupRunner(router)


def convert_dependency_group(payload: GroupWorkPayload) -> GroupWorkSummary:
    """
    ProcessPool worker 入口：在子进程内创建 reader/router 并跑完一个 dependency group。

    必须在模块顶层定义以便 ``spawn`` 上下文 pickle。
    """
    if _RESULT_QUEUE is None:
        raise RuntimeError("convert worker result queue is not initialized")

    context, runner = bootstrap_convert_worker(payload.model_path, payload.config, _PROCESS_DEVICE)

    def _sink(result: IRResult) -> None:
        _RESULT_QUEUE.put(result)

    runner.run_group(
        context=context,
        group=payload.routed_tasks,
        max_workers=payload.worker_threads,
        budget=payload.budget,
        catalog=None,
        return_mode=payload.return_mode,
        result_sink=_sink,
    )
    return GroupWorkSummary.from_runner(runner)
