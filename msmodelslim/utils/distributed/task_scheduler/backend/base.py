#!/usr/bin/env python
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

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple


from torch import nn  # noqa: F401
from msmodelslim.utils.distributed.task_scheduler.types import TaskExecutionRecord, TaskSyncContext


class DTSBackend(ABC):
    """DTS 调度后端：在保持 submit/run 语义一致的前提下，可替换具体调度实现。"""

    @abstractmethod
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
        """由 ``DistributedTaskScheduler`` 调用；``scheduler_disable_parallel`` / ``global_disable_parallel`` 来自门面。"""

    @abstractmethod
    def run(self) -> List[TaskExecutionRecord]:
        """执行已提交任务并返回与提交顺序一致的记录。"""
