#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""DTS 稳定门面：``with`` / ``submit`` / ``run``；具体调度由 backend 实现。"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from torch import nn

from msmodelslim.utils.distributed.task_scheduler.backend.base import DTSBackend
from msmodelslim.utils.distributed.task_scheduler.backend.wave import WaveDTSBackend
from msmodelslim.utils.distributed.task_scheduler.types import TaskExecutionRecord, TaskSyncContext

class DistributedTaskScheduler:
    """分波调度（默认 wave backend）；依赖/前缀与 parallel 语义冲突则新开波次。"""

    _global_disable_parallel: bool = False

    @classmethod
    def set_global_disable_parallel(cls, val: bool) -> None:
        """设置全局 disable_parallel 标志，由模型适配器按需调用。"""
        cls._global_disable_parallel = val

    @classmethod
    def get_global_disable_parallel(cls) -> bool:
        """获取全局 disable_parallel 标志。"""
        return cls._global_disable_parallel

    def __init__(
            self,
            model: nn.Module,
            disable_parallel: bool = False,
            backend: Optional[DTSBackend] = None,
    ) -> None:
        self.model = model
        self.disable_parallel = disable_parallel
        self._backend: DTSBackend = backend if backend is not None else WaveDTSBackend(model)
        self._closed = False

    def __enter__(self) -> "DistributedTaskScheduler":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _ = (exc_type, exc, tb)
        self._closed = True
        close_fn = getattr(self._backend, "close", None)
        if callable(close_fn):
            close_fn()

    def submit(
            self,
            fn: Callable[..., Any],
            args: Tuple[Any, ...] = (),
            kwargs: Optional[Dict[str, Any]] = None,
            dependencies: Optional[List[str]] = None,
            sync_fn: Optional[Callable[[TaskExecutionRecord, TaskSyncContext], Any]] = None,
            parallel: bool = True,
    ) -> None:
        if self._closed:
            raise RuntimeError("DistributedTaskScheduler is closed")
        self._backend.submit(
            fn,
            args=args,
            kwargs=kwargs,
            dependencies=dependencies,
            sync_fn=sync_fn,
            parallel=parallel,
            scheduler_disable_parallel=self.disable_parallel,
            global_disable_parallel=self.get_global_disable_parallel(),
        )

    def run(self) -> List[TaskExecutionRecord]:
        return self._backend.run()



