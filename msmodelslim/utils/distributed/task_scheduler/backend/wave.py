#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Wave 调度 backend：按依赖前缀与 parallel 语义分波执行。"""

import contextlib
import queue as py_queue
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pygtrie
import torch
from torch import distributed as dist
from torch import nn

from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.logging import get_logger

from msmodelslim.utils.distributed.task_scheduler.backend.base import DTSBackend
from msmodelslim.utils.distributed.task_scheduler.constants import (
    DISTRIBUTED_TASK_QUEUE_GET_TIMEOUT_S,
    DTS_PERF_LOG_NOT_SUITABLE_FOR_PARALLEL_PREFIX,
    DTS_PERF_LOG_RUN_TIME_SUMMARY_PREFIX,
    DTS_PERF_LOG_SPEEDUP_RATIO_PREFIX,
    DTS_PERF_LOG_SPEEDUP_SKIPPED_PREFIX,
    DTS_USER_LOG_PREFIX,
    get_distributed_task_work_queue,
)
from msmodelslim.utils.distributed.task_scheduler.types import (
    Task,
    TaskExecutionRecord,
    TaskSyncContext,
    _TaskSpec,
)
from msmodelslim.utils.distributed.task_scheduler.payload import (
    task_semantic_hash,
    validate_dependency_paths,
    wave_semantic_hash,
)
from msmodelslim.utils.distributed.task_scheduler.sync import DTSMixin, default_module_state_sync


@contextlib.contextmanager
def _collective_op_guard():
    """检测共享任务执行中是否非法调用集合通信操作。

    DTS 要求共享任务可被任意 rank 独立执行，若任务函数内含有
    ``torch.distributed.broadcast`` / ``all_gather`` 等多 rank 同步操作，
    则该任务在多卡并行时因不同 rank 执行不同任务而导致挂死或错误结果。
    """
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        yield
        return

    _COLLECTIVE_OP_NAMES = frozenset({
        "broadcast",
        "all_reduce",
        "reduce",
        "all_gather",
        "all_gather_into_tensor",
        "gather",
        "scatter",
        "all_to_all",
        "all_to_all_single",
        "reduce_scatter",
        "reduce_scatter_tensor",
        "barrier",
        "monitored_barrier",
        "broadcast_object_list",
        "all_gather_object",
        "gather_object",
        "scatter_object",
    })

    def _make_guard(op_name: str):
        def _guarded(*args, **kwargs):
            # 捕获调用栈，定位用户任务函数的调用位置
            stack = traceback.extract_stack()[:-1]
            caller_lines = ""
            for frame in reversed(stack):
                fname = frame.filename.replace("\\", "/")
                if "_collective_op_guard" in frame.name or "wave.py" in fname:
                    continue
                caller_lines = (
                    f"  File \"{frame.filename}\", line {frame.lineno}, in {frame.name}\n"
                    f"    {frame.line}"
                )
                break

            msg = (
                f"DTS shared task contains illegal cross-rank collective: "
                f"torch.distributed.{op_name}()."
            )
            tip = (
                "DTS shared tasks must be independently executable by any single rank; "
                "they must NOT contain multi-rank synchronization (broadcast, all_gather, etc.). "
                "Remove the collective call from the task function, or restructure the task "
                "so that synchronization happens via DTS sync phase (sync_fn / DTSMixin)."
            )
            if caller_lines:
                tip += f"\n\nCall site:\n{caller_lines}"
            raise SchemaValidateError(msg, action=tip)
        return _guarded

    guards = {name: _make_guard(name) for name in _COLLECTIVE_OP_NAMES if hasattr(dist, name)}
    from unittest.mock import patch
    with patch.multiple(dist, **guards):
        yield


def _dp_tasks_executed_on_this_rank(n_tasks: int, rank: int, world_size: int, all_ranks_execute: bool) -> int:
    """DP 语义下本 rank 预计执行的任务个数（仅用于日志；队列抢跑时实际次数仍可能不同）。"""
    if all_ranks_execute:
        return n_tasks
    return sum(1 for i in range(n_tasks) if (i % world_size) == rank)


class _DtsSingleWaveSchedulerBase(ABC):
    """单波次调度基类（内部）；子类实现并行或本地全量语义。"""

    def __init__(self, model: nn.Module, task_id_prefix: str = "") -> None:
        self.model = model
        self._tasks: List[Task] = []
        self._task_id_prefix = str(task_id_prefix or "")
        self._next_task_seq: int = 0
        self._closed = False
        self._wave_dep_trie: pygtrie.StringTrie = pygtrie.StringTrie(separator=".")
        self._wave_parallel_key: Optional[bool] = None

    def __enter__(self) -> "_DtsSingleWaveSchedulerBase":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _ = (exc_type, exc, tb)
        self._closed = True

    @abstractmethod
    def submit(
            self,
            fn: Callable[..., Any],
            args: Tuple[Any, ...] = (),
            kwargs: Optional[Dict[str, Any]] = None,
            dependencies: Optional[List[str]] = None,
            sync_fn: Optional[Callable[[TaskExecutionRecord, TaskSyncContext], Any]] = None,
            parallel: bool = True,
            wave_parallel_key: Optional[bool] = None,
    ) -> None:
        """提交任务。"""

    @abstractmethod
    def run(self) -> List[TaskExecutionRecord]:
        """执行当前 wave。"""

    @abstractmethod
    def _has_conflict(self, parallel: bool) -> bool:
        """判断新任务是否与当前 wave 的并行语义冲突。"""

    def registered_dependency_paths(self) -> Set[str]:
        """本波次已登记的非空依赖路径集合（供分波调度器与测试观测）。"""
        return set(self._wave_dep_trie.keys())

    def _has_prefix_conflict_in_wave(self, dep: str) -> bool:
        """判断 ``dep`` 是否与本波次已登记路径冲突（同路径 / 前缀 / 子路径）。"""
        d = (dep or "").strip().strip(".")
        if not d:
            return False
        trie = self._wave_dep_trie
        if trie.has_key(d):
            return True
        if trie.has_subtrie(d):
            return True
        for _ in trie.prefixes(d):
            return True
        return False

    def dependencies_conflict_with_wave(self, deps: List[str]) -> bool:
        """候选 ``deps`` 是否与本波次已累积依赖冲突（含前缀冲突）；不校验路径是否在模型上存在。"""
        if not deps or len(self._wave_dep_trie) == 0:
            return False
        for d in deps:
            if self._has_prefix_conflict_in_wave(d):
                return True
        return False

    def submission_conflicts_with_wave(self, deps: List[str], wave_parallel_key: bool) -> bool:
        """依赖前缀冲突或 ``wave_parallel_key`` 与当前波次不一致则冲突。"""
        if self._wave_parallel_key is not None and wave_parallel_key != self._wave_parallel_key:
            return True
        return self.dependencies_conflict_with_wave(deps)

    def _register_wave_dependencies(self, deps: List[str]) -> None:
        for dep in deps:
            d = (dep or "").strip().strip(".")
            if d:
                self._wave_dep_trie[d] = True

    def _submit_common(
            self,
            fn: Callable[..., Any],
            args: Tuple[Any, ...] = (),
            kwargs: Optional[Dict[str, Any]] = None,
            dependencies: Optional[List[str]] = None,
            sync_fn: Optional[Callable[[TaskExecutionRecord, TaskSyncContext], Any]] = None,
            parallel: bool = True,
            wave_parallel_key: Optional[bool] = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("scheduler is closed")
        if self._has_conflict(parallel):
            expected = self._tasks[0].spec.parallel if self._tasks else parallel
            raise SchemaValidateError(
                "DTS submit: parallel must match other tasks in this wave "
                f"(expected {expected!r}, got {parallel!r}).",
                action=(
                    "Split tasks into separate waves by using consistent parallel flags per wave "
                    "or submit via DistributedTaskScheduler which auto-splits conflicting tasks."
                ),
            )
        deps = list(dependencies or [])
        validate_dependency_paths(self.model, deps)
        semantic_hash = task_semantic_hash(
            fn=fn,
            args=tuple(args),
            kwargs=dict(kwargs or {}),
            dependencies=deps,
            parallel=parallel,
            sync_fn=sync_fn,
        )
        if wave_parallel_key is not None:
            if not self._tasks:
                self._wave_parallel_key = wave_parallel_key
            elif self._wave_parallel_key != wave_parallel_key:
                raise SchemaValidateError(
                    "DTS wave submit: wave_parallel_key mismatch within the same wave "
                    f"(expected {self._wave_parallel_key!r}, got {wave_parallel_key!r}).",
                    action="Submit via DistributedTaskScheduler so tasks split across waves automatically.",
                )
        internal_id = f"{self._task_id_prefix}t{self._next_task_seq}"
        self._next_task_seq += 1
        spec = _TaskSpec(
            task_id=internal_id,
            dependencies=deps,
            fn=fn,
            args=tuple(args),
            kwargs=dict(kwargs or {}),
            parallel=parallel,
            semantic_hash=semantic_hash,
        )
        self._tasks.append(Task(spec=spec, sync_fn=sync_fn))
        self._register_wave_dependencies(deps)
        return None

    def _execute_local_task(self, task: Task, rank: int, world_size: int) -> Tuple[Any, float]:
        t0 = time.perf_counter()
        if task.spec.fn is None:
            raise RuntimeError(f"Task {task.spec.task_id} has no fn to execute.")
        out = task.spec.fn(*task.spec.args, **task.spec.kwargs)
        return out, (time.perf_counter() - t0)

    def _run_single_rank(self, rank: int, world_size: int) -> Tuple[Dict[str, Any], Dict[str, int], Dict[str, float]]:
        local_results: Dict[str, Any] = {}
        owner_map: Dict[str, int] = {}
        exec_time_map: Dict[str, float] = {}
        for task in self._tasks:
            spec = task.spec
            out, dur = self._execute_local_task(task, rank=rank, world_size=world_size)
            local_results[spec.task_id] = out
            exec_time_map[spec.task_id] = dur
            owner_map[spec.task_id] = rank
        return local_results, owner_map, exec_time_map

    def _execute_all_rank_tasks(
            self,
            task_indices: List[int],
            queue: Optional[Any],
            rank: int,
            world_size: int,
            local_results: Dict[str, Any],
            local_exec_rank: Dict[str, int],
            local_exec_time_s: Dict[str, float],
            all_ranks_execute_tasks: bool = False,
    ) -> None:
        if not task_indices:
            return
        if all_ranks_execute_tasks:
            for i in task_indices:
                task = self._tasks[i]
                spec = task.spec
                out, dur = self._execute_local_task(task, rank, world_size)
                local_results[spec.task_id] = out
                local_exec_time_s[spec.task_id] = dur
                local_exec_rank[spec.task_id] = 0
                local_exec_time_s[f"idx:{i}"] = dur
                local_exec_rank[f"idx:{i}"] = 0
            return
        if queue is not None:
            if rank == 0:
                for i in task_indices:
                    queue.put(i)
                for _ in range(world_size):
                    queue.put(None)
            while True:
                try:
                    idx = queue.get(timeout=DISTRIBUTED_TASK_QUEUE_GET_TIMEOUT_S)
                except py_queue.Empty as e:
                    raise RuntimeError(
                        f"DistributedTaskScheduler queue.get() timeout after {DISTRIBUTED_TASK_QUEUE_GET_TIMEOUT_S}s. "
                        "Possible shared queue visibility issue or upstream producer stall."
                    ) from e
                if idx is None:
                    break
                task = self._tasks[idx]
                out, dur = self._execute_local_task(task, rank, world_size)
                spec = task.spec
                local_results[spec.task_id] = out
                local_exec_time_s[spec.task_id] = dur
                local_exec_rank[spec.task_id] = rank
                local_exec_time_s[f"idx:{idx}"] = dur
                local_exec_rank[f"idx:{idx}"] = rank
        else:
            for i in task_indices:
                task = self._tasks[i]
                owner = i % world_size
                if rank == owner:
                    out, dur = self._execute_local_task(task, rank, world_size)
                    spec = task.spec
                    local_results[spec.task_id] = out
                    local_exec_time_s[spec.task_id] = dur
                    local_exec_rank[spec.task_id] = owner
                    local_exec_time_s[f"idx:{i}"] = dur
                    local_exec_rank[f"idx:{i}"] = owner

    def _merge_owner_map_multirank(self, local_exec_rank: Dict[str, int], world_size: int) -> Dict[str, int]:
        gathered_exec: List[Optional[Dict[str, int]]] = [None] * world_size
        dist.all_gather_object(gathered_exec, local_exec_rank)
        owner_map: Dict[str, int] = {}
        for d in gathered_exec:
            if d:
                owner_map.update(d)
        return owner_map

    def _merge_exec_time_map_multirank(self, local_exec_time_s: Dict[str, float], world_size: int) -> Dict[str, float]:
        gathered: List[Optional[Dict[str, float]]] = [None] * world_size
        dist.all_gather_object(gathered, local_exec_time_s)
        merged: Dict[str, float] = {}
        for d in gathered:
            if d:
                merged.update(d)
        return merged

    def _sync_all_task(
        self,
        owner_map: Dict[str, int],
        local_results: Dict[str, Any],
        exec_time_map: Dict[str, float],
        rank: int,
        world_size: int,
        sync_enabled: bool = True,
    ) -> List[TaskExecutionRecord]:
        sync_ctx = TaskSyncContext(
            model=self.model,
            rank=rank,
            world_size=world_size,
        )
        records: List[TaskExecutionRecord] = []
        for idx, task in enumerate(self._tasks):
            spec = task.spec
            idx_key = f"idx:{idx}"
            owner_key = spec.task_id if spec.task_id in owner_map else idx_key
            if owner_key not in owner_map:
                raise RuntimeError(f"Task {spec.task_id} has no executor record after run.")
            record = TaskExecutionRecord(
                task_id=spec.task_id,
                executor_rank=owner_map[owner_key],
                result=local_results.get(spec.task_id),
                dependencies=list(spec.dependencies or []),
                exec_time_s=float(exec_time_map.get(spec.task_id, exec_time_map.get(idx_key, 0.0)) or 0.0),
            )
            if not sync_enabled:
                records.append(record)
                continue
            t0 = time.perf_counter()
            self._sync_task(task, record, sync_ctx)
            record.sync_time_s = time.perf_counter() - t0
            records.append(record)
        return records

    def _sync_task(self, task: Task, record: TaskExecutionRecord, sync_ctx: TaskSyncContext) -> None:
        if task.sync_fn is not None:
            task.sync_fn(record, sync_ctx)
            return

        if not task.spec.dependencies:
            return

        for dep in task.spec.dependencies:
            module = self.model.get_submodule(dep)
            for sub in module.modules():
                if isinstance(sub, DTSMixin):
                    sub.distributed_sync(record, sync_ctx)
                else:
                    default_module_state_sync(record, sync_ctx, sub)

    def _build_records_without_sync(
        self,
        owner_map: Dict[str, int],
        local_results: Dict[str, Any],
        exec_time_map: Dict[str, float],
    ) -> List[TaskExecutionRecord]:
        records: List[TaskExecutionRecord] = []
        for task in self._tasks:
            spec = task.spec
            if spec.task_id not in owner_map:
                raise RuntimeError(f"Task {spec.task_id} has no executor record after run.")
            records.append(
                TaskExecutionRecord(
                    task_id=spec.task_id,
                    executor_rank=owner_map[spec.task_id],
                    result=local_results.get(spec.task_id),
                    dependencies=list(spec.dependencies or []),
                    exec_time_s=float(exec_time_map.get(spec.task_id, 0.0) or 0.0),
                )
            )
        return records

    def _finalize_local_wave_no_sync(
            self,
            owner_map: Dict[str, int],
            local_results: Dict[str, Any],
            exec_time_map: Dict[str, float],
            rank: int,
            world_size: int,
            t_run_wall_start: float,
    ) -> Tuple[List[TaskExecutionRecord], float, float]:
        records = self._build_records_without_sync(owner_map, local_results, exec_time_map)
        total_exec_s = sum(float(r.exec_time_s or 0.0) for r in records)
        total_sync_s = sum(float(r.sync_time_s or 0.0) for r in records)
        return records, total_exec_s, total_sync_s


class _DtsMultiRankParallelWaveScheduler(_DtsSingleWaveSchedulerBase):
    def __init__(self, model: nn.Module, task_id_prefix: str = "") -> None:
        super().__init__(model, task_id_prefix)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size <= 1:
            raise SchemaValidateError(
                "DTS parallel wave requires multi-rank environment (world_size > 1).",
                action="Use _DtsSequentialWaveScheduler for single-rank or no-parallel semantics.",
            )

    def _execute_local_task(self, task: Task, rank: int, world_size: int) -> Tuple[Any, float]:
        t0 = time.perf_counter()
        if task.spec.fn is None:
            raise RuntimeError(f"Task {task.spec.task_id} has no fn to execute.")
        with _collective_op_guard():
            out = task.spec.fn(*task.spec.args, **task.spec.kwargs)
        return out, (time.perf_counter() - t0)

    def _has_conflict(self, parallel: bool) -> bool:
        if not self._tasks:
            return parallel is not True
        return self._tasks[0].spec.parallel != parallel

    def submit(
            self,
            fn: Callable[..., Any],
            args: Tuple[Any, ...] = (),
            kwargs: Optional[Dict[str, Any]] = None,
            dependencies: Optional[List[str]] = None,
            sync_fn: Optional[Callable[[TaskExecutionRecord, TaskSyncContext], Any]] = None,
            parallel: bool = True,
            wave_parallel_key: Optional[bool] = None,
    ) -> None:
        return self._submit_common(
            fn,
            args=args,
            kwargs=kwargs,
            dependencies=dependencies,
            sync_fn=sync_fn,
            parallel=parallel,
            wave_parallel_key=wave_parallel_key,
        )

    def run(self) -> List[TaskExecutionRecord]:
        if not self._tasks:
            return []

        t_run_wall_start = time.perf_counter()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        _log = get_logger()

        gathered_counts: List[Optional[int]] = [None] * world_size
        dist.all_gather_object(gathered_counts, int(len(self._tasks)))
        if len(set(gathered_counts)) != 1:
            raise RuntimeError(
                "DTS submit mismatch across ranks when disable_parallel=False. "
                f"rank={rank} gathered_task_counts={gathered_counts}"
            )
        local_hashes = [task.spec.semantic_hash for task in self._tasks]
        local_wave_hash = wave_semantic_hash(local_hashes)
        gathered_wave_hashes: List[Optional[str]] = [None] * world_size
        dist.all_gather_object(gathered_wave_hashes, local_wave_hash)
        base_wave_hash = gathered_wave_hashes[0]
        if any(h != base_wave_hash for h in gathered_wave_hashes):
            gathered_hashes: List[Optional[List[str]]] = [None] * world_size
            dist.all_gather_object(gathered_hashes, local_hashes)
            base_hashes = gathered_hashes[0] or []
            for other_rank, hashes in enumerate(gathered_hashes):
                current = hashes or []
                if current == base_hashes:
                    continue
                mismatch_idx = 0
                for i, (lhs, rhs) in enumerate(zip(base_hashes, current)):
                    if lhs != rhs:
                        mismatch_idx = i
                        break
                else:
                    mismatch_idx = min(len(base_hashes), len(current))
                raise RuntimeError(
                    "DTS submit semantic mismatch across ranks when disable_parallel=False. "
                    f"rank={rank} compare_rank={other_rank} mismatch_task_index={mismatch_idx} "
                    f"base_hash={base_hashes[mismatch_idx] if mismatch_idx < len(base_hashes) else 'NA'} "
                    f"compare_hash={current[mismatch_idx] if mismatch_idx < len(current) else 'NA'}"
                )

        n_tasks = len(self._tasks)
        task_indices = list(range(n_tasks))
        local_results = {}
        local_exec_rank: Dict[str, int] = {}
        local_exec_time_s: Dict[str, float] = {}
        queue = get_distributed_task_work_queue()

        self._execute_all_rank_tasks(
            task_indices,
            queue,
            rank,
            world_size,
            local_results,
            local_exec_rank,
            local_exec_time_s,
            all_ranks_execute_tasks=False,
        )
        owner_map = self._merge_owner_map_multirank(local_exec_rank, world_size)
        exec_time_map = self._merge_exec_time_map_multirank(local_exec_time_s, world_size)

        records = self._sync_all_task(
            owner_map,
            local_results,
            exec_time_map,
            rank,
            world_size,
            sync_enabled=True,
        )

        t_run_wall_s = time.perf_counter() - t_run_wall_start
        total_exec_s = sum(float(r.exec_time_s or 0.0) for r in records)
        total_sync_s = sum(float(r.sync_time_s or 0.0) for r in records)

        per_rank_task_count: Dict[int, int] = {}
        per_rank_exec_time: Dict[int, float] = {}
        per_rank_sync_time: Dict[int, float] = {}
        for r_rec in records:
            er = r_rec.executor_rank
            per_rank_task_count[er] = per_rank_task_count.get(er, 0) + 1
            per_rank_exec_time[er] = per_rank_exec_time.get(er, 0.0) + float(r_rec.exec_time_s or 0.0)
            per_rank_sync_time[er] = per_rank_sync_time.get(er, 0.0) + float(r_rec.sync_time_s or 0.0)

        my_tasks = per_rank_task_count.get(rank, 0)
        my_exec_s = per_rank_exec_time.get(rank, 0.0)
        my_sync_s = per_rank_sync_time.get(rank, 0.0)
        if total_sync_s > 0:
            ratio = total_exec_s / total_sync_s
            _log.debug(
                DTS_USER_LOG_PREFIX
                + DTS_PERF_LOG_RUN_TIME_SUMMARY_PREFIX
                + " rank=%s world_size=%s total_exec_s=%.6f total_sync_s=%.6f exec_over_sync=%.6f",
                rank, world_size, total_exec_s, total_sync_s, ratio,
            )
            if ratio <= 1.0:
                _log.debug(
                    DTS_USER_LOG_PREFIX
                    + DTS_PERF_LOG_NOT_SUITABLE_FOR_PARALLEL_PREFIX
                    + ": exec_over_sync=%.6f (<=1). Sync cost dominates.",
                    ratio,
                )
        queue_str = "shared" if queue is not None else "static_rr"
        speedup = float(total_exec_s) / t_run_wall_s if t_run_wall_s > 0 and total_exec_s > 0 else 0.0
        _log.info(
            DTS_USER_LOG_PREFIX + "Summary: tasks=%d world_size=%d rank=%d my_tasks=%d exec_s=%.4f my_exec_s=%.4f sync_s=%.4f my_sync_s=%.4f t_run_wall_s=%.4f speedup=%.4f queue=%s",
            n_tasks, world_size, rank, my_tasks,
            total_exec_s, my_exec_s, total_sync_s, my_sync_s, t_run_wall_s, speedup, queue_str,
        )
        return records


class _DtsSequentialWaveScheduler(_DtsSingleWaveSchedulerBase):
    def _has_conflict(self, parallel: bool) -> bool:
        _ = parallel
        return False

    def submit(
            self,
            fn: Callable[..., Any],
            args: Tuple[Any, ...] = (),
            kwargs: Optional[Dict[str, Any]] = None,
            dependencies: Optional[List[str]] = None,
            sync_fn: Optional[Callable[[TaskExecutionRecord, TaskSyncContext], Any]] = None,
            parallel: bool = True,
            wave_parallel_key: Optional[bool] = None,
    ) -> None:
        _ = parallel
        return self._submit_common(
            fn,
            args=args,
            kwargs=kwargs,
            dependencies=dependencies,
            sync_fn=sync_fn,
            parallel=False,
            wave_parallel_key=wave_parallel_key,
        )

    def run(self) -> List[TaskExecutionRecord]:
        if not self._tasks:
            return []

        t_run_wall_start = time.perf_counter()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        _log = get_logger()

        if world_size == 1:
            local_results, owner_map, exec_time_map = self._run_single_rank(0, 1)
        else:
            n_tasks = len(self._tasks)
            task_indices = list(range(n_tasks))
            local_results: Dict[str, Any] = {}
            local_exec_rank: Dict[str, int] = {}
            local_exec_time_s: Dict[str, float] = {}
            self._execute_all_rank_tasks(
                task_indices,
                None,
                rank,
                world_size,
                local_results,
                local_exec_rank,
                local_exec_time_s,
                all_ranks_execute_tasks=True,
            )
            owner_map = dict(local_exec_rank)
            exec_time_map = dict(local_exec_time_s)

        records, total_exec_s, total_sync_s = self._finalize_local_wave_no_sync(
            owner_map,
            local_results,
            exec_time_map,
            rank,
            world_size,
            t_run_wall_start,
        )
        t_run_wall_s = time.perf_counter() - t_run_wall_start
        my_tasks = len(records)
        mode = "single-rank" if world_size == 1 else "no-parallel"
        _log.info(
            DTS_USER_LOG_PREFIX + "Summary: tasks=%d world_size=%d rank=%d my_tasks=%d exec_s=%.4f my_exec_s=%.4f sync_s=%.4f my_sync_s=%.4f t_run_wall_s=%.4f mode=%s",
            len(self._tasks), world_size, rank, my_tasks,
            total_exec_s, total_exec_s, total_sync_s, total_sync_s, t_run_wall_s, mode,
        )
        return records


class WaveDTSBackend(DTSBackend):
    """默认 wave backend：依赖前缀与 parallel 语义冲突则新开波次。"""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._waves: List[_DtsSingleWaveSchedulerBase] = []
        self._closed = False

    def close(self) -> None:
        self._closed = True

    def submit(
            self,
            fn: Callable[..., Any],
            args: Tuple[Any, ...] = (),
            kwargs: Optional[Dict[str, Any]] = None,
            dependencies: Optional[List[str]] = None,
            sync_fn: Optional[Callable[[TaskExecutionRecord, TaskSyncContext], Any]] = None,
            parallel: bool = True,
            *,
            scheduler_disable_parallel: bool,
            global_disable_parallel: bool,
    ) -> None:
        if self._closed:
            raise RuntimeError("WaveDTSBackend is closed")

        deps = list(dependencies or [])
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        single_rank = world_size <= 1
        effective_local_only = (
            single_rank or scheduler_disable_parallel or global_disable_parallel or (not parallel)
        )
        wave_parallel_key = bool(parallel)
        inner_parallel = not effective_local_only
        if self._must_start_new_wave(deps, wave_parallel_key):
            wave_idx = len(self._waves)
            if effective_local_only:
                new_wave = _DtsSequentialWaveScheduler(self.model, task_id_prefix=f"w{wave_idx}_")
            else:
                new_wave = _DtsMultiRankParallelWaveScheduler(self.model, task_id_prefix=f"w{wave_idx}_")
            new_wave.submit(
                fn,
                args=args,
                kwargs=kwargs,
                dependencies=deps,
                sync_fn=sync_fn,
                parallel=inner_parallel,
                wave_parallel_key=wave_parallel_key,
            )
            self._waves.append(new_wave)
        else:
            self._waves[-1].submit(
                fn,
                args=args,
                kwargs=kwargs,
                dependencies=deps,
                sync_fn=sync_fn,
                parallel=inner_parallel,
                wave_parallel_key=wave_parallel_key,
            )

    def _must_start_new_wave(self, deps: List[str], wave_parallel_key: bool) -> bool:
        if not self._waves:
            return True
        return self._waves[-1].submission_conflicts_with_wave(deps, wave_parallel_key)

    def run(self) -> List[TaskExecutionRecord]:
        if not self._waves:
            return []

        all_records: List[TaskExecutionRecord] = []
        for wave in self._waves:
            with wave:
                all_records.extend(wave.run())

        return all_records
