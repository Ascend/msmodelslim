#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------

msmodelslim.core.quant_service.modelslim_convert.impl.multi_npu 调度器竞态时序的单元测试。

覆盖 PR 审查指出的 P1 竞态：
- worker 快于主进程落盘、全部退出而 result_queue 尚有未读结果/摘要时，不得误判失败；
- 仅当 result_queue 为空且所有 worker 已退出、summary 仍未收满时，才判定失败；
- worker 报错经 error_queue 优先抛出。
"""

import queue as _queue
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from msmodelslim.core.convert.config import ConvertConfig
from msmodelslim.core.quant_service.modelslim_convert.impl.multi_npu import MultiNpuConvertScheduler


class _FakeQueue:
    def __init__(self, items):
        self._items = list(items)

    def put(self, *args, **kwargs):
        return None

    def get(self, timeout=None):
        if self._items:
            return self._items.pop(0)
        raise _queue.Empty

    def get_nowait(self):
        return self.get()


class _FakeProc:
    def __init__(self, alive=False, pid=1):
        self._alive = alive
        self.pid = pid

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        return None

    def terminate(self):
        self._alive = False


class _FakeProcs:
    def __init__(self, procs):
        self.processes = procs


@contextmanager
def _patch_queues(q_task, q_result, q_error, procs_alive):
    """mock spawn 的 task/result/error 三队列与 worker 进程存活状态。"""
    queues = iter([q_task, q_result, q_error])

    def fake_get_context(*args, **kwargs):
        ctx = MagicMock()
        ctx.Queue = lambda *a, **k: next(queues)
        return ctx

    procs = _FakeProcs([_FakeProc(alive=procs_alive), _FakeProc(alive=procs_alive)])
    with (
        patch("torch.multiprocessing.get_context", side_effect=fake_get_context),
        patch("msmodelslim.core.quant_service.modelslim_convert.impl.multi_npu.spawn", return_value=procs),
    ):
        yield


def _make_scheduler(groups):
    return MultiNpuConvertScheduler(
        model_path="/m",
        config=ConvertConfig(model_path="/m", save_path="/o"),
        device_indices=[0, 1],
        groups=groups,
        budget=None,
        return_mode="state_dict",
    )


class TestMultiNpuRace:
    def test_completes_when_all_workers_exit_but_queue_still_has_results(self):
        """P1 回归：worker 全部退出、队列尚有未读 result/summary 时不得误判失败。"""
        groups = [["g0"], ["g1"]]
        q_result = _FakeQueue(
            [
                ("result", "r0"),
                ("result", "r1"),
                ("summary", "s0"),
                ("summary", "s1"),
            ]
        )
        with _patch_queues(_FakeQueue([]), q_result, _FakeQueue([]), procs_alive=False):
            scheduler = _make_scheduler(groups)
            yielded = list(scheduler.run())
        # 收满 2 个 summary 后正常退出，不抛 "All NPU workers exited" 误报
        assert [y for y in yielded if y in ("r0", "r1")] == ["r0", "r1"]
        assert len(scheduler.summaries) == 2

    def test_raises_when_queue_empty_and_all_workers_exited_before_all_summaries(self):
        """队列为空且所有 worker 已退出、summary 未收满 -> 判定失败。"""
        groups = [["g0"], ["g1"]]
        q_result = _FakeQueue([("result", "r0"), ("summary", "s0")])  # 缺第 2 个 summary
        with _patch_queues(_FakeQueue([]), q_result, _FakeQueue([]), procs_alive=False):
            scheduler = _make_scheduler(groups)
            with pytest.raises(RuntimeError, match="All NPU workers exited"):
                list(scheduler.run())

    def test_collect_saved_and_worker_meta_when_direct_write(self):
        """直接写盘：saved 进度标记推进结果，收满 summary 后继续收集 worker_meta。"""
        groups = [["g0"], ["g1"]]
        q_result = _FakeQueue(
            [
                ("saved", "mod0"),
                ("saved", "mod1"),
                ("summary", "s0"),
                ("summary", "s1"),
                ("worker_meta", (".direct_0", {"k": "f0"}, {})),
                ("worker_meta", (".direct_1", {"k2": "f1"}, {})),
            ]
        )
        with _patch_queues(_FakeQueue([]), q_result, _FakeQueue([]), procs_alive=False):
            scheduler = MultiNpuConvertScheduler(
                model_path="/m",
                config=ConvertConfig(model_path="/m", save_path="/o"),
                device_indices=[0, 1],
                groups=groups,
                save_path="/o",
                return_mode="module",
            )
            yielded = list(scheduler.run())
        assert [y.module_path for y in yielded] == ["mod0", "mod1"]
        assert all(y.already_saved for y in yielded)
        assert len(scheduler.summaries) == 2
        assert len(scheduler.worker_metas) == 2
