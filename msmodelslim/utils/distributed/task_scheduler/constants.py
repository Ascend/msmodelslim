#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""DTS 运行时常量与进程级共享队列（与具体调度 backend 无关）。"""

from typing import Any, Optional

# 共享队列 ``queue.get`` 超时（秒），避免进程间异常时无限阻塞。
DISTRIBUTED_TASK_QUEUE_GET_TIMEOUT_S = 300

# 用户可见调度日志统一前缀（便于过滤）；性能指引常量勿改文案。
DTS_USER_LOG_PREFIX = "【DTS】 "

# 性能指引日志前缀（勿改文案；``test.cases.utils.distributed.test_dts_performance`` 依赖）。
DTS_PERF_LOG_RUN_TIME_SUMMARY_PREFIX = "DTS run time summary:"
DTS_PERF_LOG_NOT_SUITABLE_FOR_PARALLEL_PREFIX = "DTS not suitable for parallel"
DTS_PERF_LOG_SPEEDUP_RATIO_PREFIX = "DTS speedup ratio (total_exec / t_run_wall)"
DTS_PERF_LOG_SPEEDUP_SKIPPED_PREFIX = "DTS speedup ratio skipped"

# 由 runner 在 spawn 前创建后注入子进程，供 scheduler 共享任务抢占使用。
_DISTRIBUTED_TASK_WORK_QUEUE: Optional[Any] = None


def set_distributed_task_work_queue(q: Optional[Any]) -> None:
    """在分布式 worker 进程内注册共享任务队列（仅 ``mp.spawn`` 子进程路径，由 ``DPLayerWiseRunner`` 调用）。"""
    global _DISTRIBUTED_TASK_WORK_QUEUE
    _DISTRIBUTED_TASK_WORK_QUEUE = q


def get_distributed_task_work_queue() -> Optional[Any]:
    """返回当前进程已注入的共享队列；未注入时返回 ``None``（按 ``idx % world_size`` 静态指派 owner）。"""
    return _DISTRIBUTED_TASK_WORK_QUEUE


def clear_distributed_task_work_queue() -> None:
    """清理当前进程的共享队列引用（通常在 worker ``finally`` 中调用）。"""
    global _DISTRIBUTED_TASK_WORK_QUEUE
    _DISTRIBUTED_TASK_WORK_QUEUE = None
