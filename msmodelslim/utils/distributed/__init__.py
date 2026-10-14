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

__all__ = [
    "DistHelper",
    "find_free_port",
    "sync_base_operation",
    "sync_gather_tensors",
    "setup_distributed",
    "is_rank_zero",
    "DistributedTaskScheduler",
    "DTSMixin",
    "TaskExecutionRecord",
    "TaskSyncContext",
    "set_distributed_task_work_queue",
    "get_distributed_task_work_queue",
    "clear_distributed_task_work_queue",
    "DTSBackend",
    "WaveDTSBackend",
    "DTS_PERF_LOG_RUN_TIME_SUMMARY_PREFIX",
    "DTS_PERF_LOG_NOT_SUITABLE_FOR_PARALLEL_PREFIX",
    "DTS_PERF_LOG_SPEEDUP_RATIO_PREFIX",
    "DTS_PERF_LOG_SPEEDUP_SKIPPED_PREFIX",
]

from msmodelslim.utils.distributed.dist_helper import DistHelper, is_rank_zero
from msmodelslim.utils.distributed.dist_ops import sync_base_operation, sync_gather_tensors
from msmodelslim.utils.distributed.dist_setup import find_free_port, setup_distributed
from msmodelslim.utils.distributed.task_scheduler import (
    DTSBackend,
    DTSMixin,
    DistributedTaskScheduler,
    TaskExecutionRecord,
    TaskSyncContext,
    WaveDTSBackend,
    clear_distributed_task_work_queue,
    get_distributed_task_work_queue,
    set_distributed_task_work_queue,
)
from msmodelslim.utils.distributed.task_scheduler.constants import (
    DTS_PERF_LOG_NOT_SUITABLE_FOR_PARALLEL_PREFIX,
    DTS_PERF_LOG_RUN_TIME_SUMMARY_PREFIX,
    DTS_PERF_LOG_SPEEDUP_RATIO_PREFIX,
    DTS_PERF_LOG_SPEEDUP_SKIPPED_PREFIX,
)
