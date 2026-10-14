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

DTS 对外公开接口（仅导出公开符号）。
"""

from msmodelslim.utils.distributed.task_scheduler.backend import DTSBackend, WaveDTSBackend
from msmodelslim.utils.distributed.task_scheduler.constants import (
    clear_distributed_task_work_queue,
    get_distributed_task_work_queue,
    set_distributed_task_work_queue,
)
from msmodelslim.utils.distributed.task_scheduler.types import TaskExecutionRecord, TaskSyncContext
from msmodelslim.utils.distributed.task_scheduler.scheduler import DistributedTaskScheduler
from msmodelslim.utils.distributed.task_scheduler.sync import DTSMixin

__all__ = [
    "DTSBackend",
    "DTSMixin",
    "DistributedTaskScheduler",
    "TaskExecutionRecord",
    "TaskSyncContext",
    "WaveDTSBackend",
    "set_distributed_task_work_queue",
    "get_distributed_task_work_queue",
    "clear_distributed_task_work_queue",
]
