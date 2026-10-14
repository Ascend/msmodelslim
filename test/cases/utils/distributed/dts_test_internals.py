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

DTS 测试统一内部导入入口。
"""

from typing import TYPE_CHECKING

from msmodelslim.utils.distributed.task_scheduler.backend.wave import (
    WaveDTSBackend,
    _DtsMultiRankParallelWaveScheduler,
    _DtsSequentialWaveScheduler,
)
from msmodelslim.utils.distributed.task_scheduler.sync import default_module_state_sync
from msmodelslim.utils.distributed.task_scheduler.types import _TaskSpec

if TYPE_CHECKING:
    from msmodelslim.utils.distributed.task_scheduler.scheduler import DistributedTaskScheduler


def get_default_wave_backend(scheduler: "DistributedTaskScheduler") -> WaveDTSBackend:
    """默认 ``DistributedTaskScheduler`` 使用 ``WaveDTSBackend``；其它 backend 下测试应显式构造 backend。"""
    b = scheduler._backend
    if not isinstance(b, WaveDTSBackend):
        raise TypeError(f"expected WaveDTSBackend, got {type(b)!r}")
    return b


def dts_waves(scheduler: "DistributedTaskScheduler"):
    """默认 wave backend 下的波次列表（仅测试侧检视）。"""
    return get_default_wave_backend(scheduler)._waves


__all__ = [
    "_TaskSpec",
    "default_module_state_sync",
    "get_default_wave_backend",
    "dts_waves",
    "_DtsMultiRankParallelWaveScheduler",
    "_DtsSequentialWaveScheduler",
]
