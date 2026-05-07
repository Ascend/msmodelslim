#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# -------------------------------------------------------------------------
# 分布式单测公共入口：与常见 ``mp.spawn`` + ``distributed_test`` 写法一致；
# DTS 要求并行用例使用 **spawn + gloo**，并用 **file://** 初始化以避免固定 TCP 端口冲突。
# -------------------------------------------------------------------------

from __future__ import annotations

import os
import queue as py_queue
import shutil
import tempfile
import threading
from functools import wraps
from typing import Any, Callable, List, Optional, Tuple

import torch.multiprocessing as mp


def _dts_prepare_spawn() -> None:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


def spawn_queue():
    """父进程侧 Queue：必须与 ``torch.multiprocessing.spawn`` 同属 spawn 上下文（Linux 默认 fork 会触发 SemLock 错误）。"""
    return mp.get_context("spawn").Queue()


def _dts_spawn_entry(
    rank: int,
    world_size: int,
    backend: str,
    init_file_abs: str,
    user_fn: Callable[..., Any],
    user_args: Tuple[Any, ...],
    user_kwargs: Optional[dict],
) -> None:
    import torch.distributed as dist

    init_url = "file://" + init_file_abs.replace("\\", "/")
    dist.init_process_group(
        backend=backend,
        init_method=init_url,
        rank=rank,
        world_size=world_size,
    )
    try:
        user_fn(rank, world_size, *user_args, **(user_kwargs or {}))
        if dist.is_initialized() and world_size > 1:
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_distributed_spawn(
    world_size: int,
    fn: Callable[..., Any],
    fn_args: Tuple[Any, ...] = (),
    fn_kwargs: Optional[dict] = None,
    *,
    backend: str = "gloo",
    init_dir_prefix: str = "dts_dist_spawn_",
) -> None:
    """使用 ``torch.multiprocessing.spawn`` 启动多进程，并在各进程内初始化进程组后调用 ``fn``。

    调用约定：``fn(rank, world_size, *fn_args, **fn_kwargs)``（``rank`` / ``world_size`` 与 PyTorch spawn 一致）。

    临时目录仅用于 ``file://`` store；全部子进程退出后删除。
    """
    fn_kwargs = fn_kwargs or {}
    init_dir = tempfile.mkdtemp(prefix=init_dir_prefix)
    init_file_abs = os.path.join(init_dir, "store")
    open(init_file_abs, "w").close()
    abs_path = os.path.abspath(init_file_abs)
    try:
        _dts_prepare_spawn()
        mp.spawn(
            _dts_spawn_entry,
            args=(world_size, backend, abs_path, fn, fn_args, fn_kwargs),
            nprocs=world_size,
            join=True,
        )
    finally:
        shutil.rmtree(init_dir, ignore_errors=True)


class QueueResultFuture:
    """父进程侧 queue 结果收集 future。"""

    def __init__(self, results_queue, expected_results: int, poll_interval_s: float = 0.2):
        self._queue = results_queue
        self._expected = int(expected_results)
        self._poll_s = float(poll_interval_s)
        self._results: List[Any] = []
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._consume, name="dts-results-consumer", daemon=True)
        self._thread.start()

    def _consume(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=self._poll_s)
            except py_queue.Empty:
                continue
            with self._lock:
                self._results.append(item)
                if len(self._results) >= self._expected:
                    self._done.set()
                    self._stop.set()
                    return

    def result(self, timeout_s: float = 120.0) -> List[Any]:
        if self._expected <= 0:
            return []
        ok = self._done.wait(timeout=max(float(timeout_s), 1.0))
        self._stop.set()
        self._thread.join(timeout=max(1.0, self._poll_s * 5))
        if not ok:
            raise TimeoutError(
                f"QueueResultFuture timeout after {timeout_s}s "
                f"(collected={len(self._results)}/{self._expected})."
            )
        with self._lock:
            return list(self._results[: self._expected])


def start_queue_result_collector(results_queue, expected_results: int, *, poll_interval_s: float = 0.2) -> QueueResultFuture:
    """启动父进程消费线程并返回结果 future。"""
    return QueueResultFuture(results_queue, expected_results=expected_results, poll_interval_s=poll_interval_s)


def distributed_test(world_size: int = 2, backend: str = "gloo"):
    """pytest 装饰器：被装饰函数签名为 ``def test_xxx(rank, world_size, ...):``。

    注意：需要 ``pytest`` 运行时导入本模块；与 ``unittest.TestCase`` 混用时请直接调用
    :func:`run_distributed_spawn`。
    """

    def decorator(func: Callable[..., Any]):
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            def _target(rank: int, wsize: int) -> None:
                func(rank, wsize, *args, **kwargs)

            run_distributed_spawn(world_size, _target, backend=backend)

        return wrapper

    return decorator
