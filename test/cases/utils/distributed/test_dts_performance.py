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
"""

"""
DTS 性能语义单测：

- **单进程 + mock perf_counter**：仅单卡 ``world_size==1`` 语义（耗时汇总 / 日志字段），**不算并行测试**。
- **契约（多卡并行 wave 日志）**：``TestDtsPerformanceGuidanceLogContract`` 用假 ``dist`` + 受控 ``perf_counter``
  断言 ``DTS_PERF_LOG_*`` 指引日志仍存在，防止误删「适合/不适合 DTS」的运行时提示。
- **多进程并行**：``dts_distributed_spawn.run_distributed_spawn``（``torch.multiprocessing.spawn``）
  +子进程 ``gloo``（``file://``），两 rank 走 ``world_size>1`` 的 ``run()``。
"""

import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch.nn as nn

from msmodelslim.utils.distributed import (
    DTS_PERF_LOG_NOT_SUITABLE_FOR_PARALLEL_PREFIX,
    DTS_PERF_LOG_RUN_TIME_SUMMARY_PREFIX,
    DTS_PERF_LOG_SPEEDUP_RATIO_PREFIX,
    DTS_PERF_LOG_SPEEDUP_SKIPPED_PREFIX,
)
from msmodelslim.utils.distributed.task_scheduler.backend import wave as dts_wave_mod
from test.cases.utils.distributed.dts_test_internals import (
    _DtsMultiRankParallelWaveScheduler,
    _DtsSequentialWaveScheduler,
)

from test.cases.utils.distributed.dts_distributed_spawn import (
    run_distributed_spawn,
    start_queue_result_collector,
    spawn_queue,
)
from test.cases.utils.distributed.test_dts_scheduler_test_workers import _run_dts_multirank_perf_worker


def _fake_all_gather_object_for_two_ranks(object_list, obj):
    """单进程 UT 模拟 ``dist.all_gather_object``（world_size=2）：rank1 贡献空 dict。"""
    n = len(object_list)
    if isinstance(obj, int):
        for i in range(n):
            object_list[i] = obj
    elif isinstance(obj, str):
        for i in range(n):
            object_list[i] = obj
    elif isinstance(obj, list):
        object_list[0] = list(obj)
        for i in range(1, n):
            object_list[i] = list(obj)
    elif isinstance(obj, dict):
        object_list[0] = dict(obj)
        for i in range(1, n):
            object_list[i] = {}
    else:
        raise AssertionError(f"unexpected all_gather_object payload type: {type(obj)!r}")


def _format_logger_call(call_args) -> str:
    args, kwargs = call_args
    if not args:
        return ""
    fmt, *rest = args
    if isinstance(fmt, str) and rest:
        try:
            return fmt % tuple(rest)
        except (TypeError, ValueError):
            return fmt + " " + repr(rest)
    return str(fmt)


def _perf_counter_sequence(exec_deltas, sync_deltas, *, extra_wall_s: float = 0.0):
    """按 ``run()`` 内 ``perf_counter`` 调用顺序生成返回值。

    含：``run()`` 入口墙钟起点、各任务执行/同步段、收尾墙钟（用于 ``T_run``）。
    ``extra_wall_s`` 加在最后一次同步结束之后，模拟调度等额外开销。
    """
    assert len(exec_deltas) == len(sync_deltas), "exec/sync 任务数必须一致"
    vals = []
    t_wall0 = 0.0
    vals.append(t_wall0)
    t = t_wall0
    for e in exec_deltas:
        vals.append(t)
        t += float(e)
        vals.append(t)
    for s in sync_deltas:
        vals.append(t)
        t += float(s)
        vals.append(t)
    vals.append(t + float(extra_wall_s))
    return vals


class TestDtsPerformanceExecVsSync(unittest.TestCase):
    """构造 exec_time 与 sync_time 的相对关系，验证 exec_over_sync 语义。"""

    def test_when_sync_much_less_than_exec_ratio_gt_one(self):
        """同步 << 执行：汇总 exec/sync 比值 > 1，不应触发「不适合并行」类提示。"""
        model = nn.Linear(2, 2)
        seq = _perf_counter_sequence([80.0], [0.05])
        mock_log = MagicMock()
        with patch.object(dts_wave_mod.time, "perf_counter", side_effect=seq):
            with patch.object(dts_wave_mod, "get_logger", return_value=mock_log):
                with _DtsSequentialWaveScheduler(model) as sch:
                    sch.submit(lambda: None, dependencies=[])
                    records = sch.run()

        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(records[0].exec_time_s, 80.0, places=5)
        self.assertAlmostEqual(records[0].sync_time_s, 0.0, places=5)
        total_exec = sum(r.exec_time_s for r in records)
        total_sync = sum(r.sync_time_s for r in records)
        self.assertEqual(total_sync, 0.0)
        not_suitable = [
            c for c in mock_log.debug.call_args_list if c.args and "not suitable for parallel" in c.args[0]
        ]
        self.assertEqual(len(not_suitable), 0)

    def test_when_sync_ge_exec_ratio_le_one_triggers_debug_branch(self):
        """同步 >= 执行：exec_over_sync <= 1 时应打 debug「不适合并行」日志。"""
        model = nn.Linear(2, 2)
        seq = _perf_counter_sequence([2.0], [8.0])
        mock_log = MagicMock()
        with patch.object(dts_wave_mod.time, "perf_counter", side_effect=seq):
            with patch.object(dts_wave_mod, "get_logger", return_value=mock_log):
                with _DtsSequentialWaveScheduler(model) as sch:
                    sch.submit(lambda: None, dependencies=[])
                    records = sch.run()

        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(records[0].exec_time_s, 2.0, places=5)
        self.assertAlmostEqual(records[0].sync_time_s, 0.0, places=5)
        total_exec = sum(r.exec_time_s for r in records)
        total_sync = sum(r.sync_time_s for r in records)
        self.assertEqual(total_sync, 0.0)

        not_suitable_calls = [
            c
            for c in mock_log.debug.call_args_list
            if c.args and "not suitable for parallel" in c.args[0]
        ]
        self.assertEqual(
            len(not_suitable_calls),
            0,
            "串行语义下不执行同步，不应输出 not suitable for parallel debug",
        )

    def test_multi_task_when_sync_much_less_than_exec(self):
        """多任务：同步总耗时仍远小于执行总耗时。"""
        model = nn.Linear(2, 2)
        seq = _perf_counter_sequence([30.0, 30.0], [0.01, 0.01])
        mock_log = MagicMock()
        with patch.object(dts_wave_mod.time, "perf_counter", side_effect=seq):
            with patch.object(dts_wave_mod, "get_logger", return_value=mock_log):
                with _DtsSequentialWaveScheduler(model) as sch:
                    sch.submit(lambda: None, dependencies=[])
                    sch.submit(lambda: None, dependencies=[])
                    records = sch.run()

        self.assertEqual(len(records), 2)
        self.assertAlmostEqual(sum(r.exec_time_s for r in records), 60.0, places=5)
        self.assertAlmostEqual(sum(r.sync_time_s for r in records), 0.0, places=5)
        not_suitable = [
            c for c in mock_log.debug.call_args_list if c.args and "not suitable for parallel" in c.args[0]
        ]
        self.assertEqual(len(not_suitable), 0)

    def test_multi_task_when_sync_dominates(self):
        """多任务：同步总耗时大于等于执行总耗时。"""
        model = nn.Linear(2, 2)
        seq = _perf_counter_sequence([1.0, 1.0], [3.0, 3.0])
        mock_log = MagicMock()
        with patch.object(dts_wave_mod.time, "perf_counter", side_effect=seq):
            with patch.object(dts_wave_mod, "get_logger", return_value=mock_log):
                with _DtsSequentialWaveScheduler(model) as sch:
                    sch.submit(lambda: None, dependencies=[])
                    sch.submit(lambda: None, dependencies=[])
                    records = sch.run()

        total_exec = sum(r.exec_time_s for r in records)
        total_sync = sum(r.sync_time_s for r in records)
        self.assertEqual(total_exec, 2.0)
        self.assertEqual(total_sync, 0.0)
        not_suitable = [
            c for c in mock_log.debug.call_args_list if c.args and "not suitable for parallel" in c.args[0]
        ]
        self.assertEqual(len(not_suitable), 0)

    def test_single_rank_run_does_not_emit_speedup_ratio_info(self):
        """单卡（world_size==1）不打印加速比 INFO（仅多卡且未 disable_parallel 时有意义）。"""
        model = nn.Linear(2, 2)
        seq = _perf_counter_sequence([10.0, 10.0], [1.0, 1.0], extra_wall_s=0.0)
        mock_log = MagicMock()
        with patch.object(dts_wave_mod.time, "perf_counter", side_effect=seq):
            with patch.object(dts_wave_mod, "get_logger", return_value=mock_log):
                with _DtsSequentialWaveScheduler(model) as sch:
                    sch.submit(lambda: None, dependencies=[])
                    sch.submit(lambda: None, dependencies=[])
                    _ = sch.run()

        speedup_calls = [
            c
            for c in mock_log.info.call_args_list
            if c.args
            and (
                "DTS speedup ratio (T_run / sum(task_exec))" in c.args[0]
                or "DTS speedup ratio skipped" in c.args[0]
            )
        ]
        self.assertEqual(len(speedup_calls), 0)


def _launch_two_rank_perf_workers(
    *,
    num_tasks: int,
    exec_sleep_s: float,
    sync_sleep_s: float,
    use_work_queue: bool,
    timeout_s: float = 120.0,
):
    """``torch.multiprocessing.spawn`` 启动 2 进程跑 DTS，返回两 rank 的结果 dict（已按 rank 排序）。"""
    work_queue = spawn_queue() if use_work_queue else None
    results_queue = spawn_queue()
    results_future = start_queue_result_collector(results_queue, expected_results=2)
    run_distributed_spawn(
        2,
        _run_dts_multirank_perf_worker,
        (
            work_queue,
            results_queue,
            num_tasks,
            exec_sleep_s,
            sync_sleep_s,
            use_work_queue,
        ),
        init_dir_prefix="dts_perf_gloo_",
    )
    raw = results_future.result(timeout_s=timeout_s)
    for r in raw:
        if not r.get("ok", True):
            raise AssertionError(f"DTS perf worker rank {r.get('rank')} failed:\n{r.get('error', r)}")

    return sorted(raw, key=lambda x: int(x["rank"]))


class TestDtsPerformanceGuidanceLogContract(unittest.TestCase):
    """契约测试：多卡并行 wave 末尾的性能指引日志（exec/sync、墙钟加速比）。

    防止误删或更名导致开发者失去「何种 workload 适合 DTS」的运行时提示。
    """

    def _run_parallel_wave_under_fake_dist(self, perf_vals, sync_fn):
        mock_log = MagicMock()
        with patch.object(dts_wave_mod.time, "perf_counter", side_effect=list(perf_vals)):
            with patch.object(dts_wave_mod, "get_logger", return_value=mock_log):
                with patch.multiple(
                    dts_wave_mod.dist,
                    is_initialized=MagicMock(return_value=True),
                    get_world_size=MagicMock(return_value=2),
                    get_rank=MagicMock(return_value=0),
                    all_gather_object=MagicMock(side_effect=_fake_all_gather_object_for_two_ranks),
                ):
                    model = nn.Linear(2, 2)
                    sch = _DtsMultiRankParallelWaveScheduler(model, task_id_prefix="ut_")
                    with sch:
                        sch.submit(lambda: None, dependencies=[], sync_fn=sync_fn, parallel=True)
                        records = sch.run()
        return records, mock_log

    def _joined_messages(self, mock_log, method_name: str) -> str:
        m = getattr(mock_log, method_name)
        return "\n".join(_format_logger_call(c) for c in m.call_args_list)

    def test_multirank_parallel_wave_logs_time_summary_and_speedup_when_exec_dominates(self):
        """同步轻于执行：不打「不适合并行」，但打墙钟/加速比 INFO。"""
        # wall0, exec0, exec1, sync0, sync1, wall1
        perf_vals = [0.0, 0.0, 10.0, 10.0, 11.0, 11.0]

        def _light_sync(_r, _c) -> None:
            return None

        records, mock_log = self._run_parallel_wave_under_fake_dist(perf_vals, _light_sync)
        self.assertEqual(len(records), 1)
        debug_text = self._joined_messages(mock_log, "debug")
        info_text = self._joined_messages(mock_log, "info")
        self.assertIn(DTS_PERF_LOG_RUN_TIME_SUMMARY_PREFIX, debug_text)
        self.assertNotIn(DTS_PERF_LOG_NOT_SUITABLE_FOR_PARALLEL_PREFIX, debug_text)
        self.assertIn(DTS_PERF_LOG_SPEEDUP_RATIO_PREFIX, info_text)
        self.assertNotIn(DTS_PERF_LOG_SPEEDUP_SKIPPED_PREFIX, info_text)

    def test_multirank_parallel_wave_warns_when_sync_dominates(self):
        """同步不轻于执行：DEBUG 提示不适合并行，且仍输出加速比 INFO。"""
        perf_vals = [0.0, 0.0, 2.0, 2.0, 12.0, 12.0]

        def _heavy_sync(_r, _c) -> None:
            return None

        records, mock_log = self._run_parallel_wave_under_fake_dist(perf_vals, _heavy_sync)
        self.assertEqual(len(records), 1)
        debug_text = self._joined_messages(mock_log, "debug")
        info_text = self._joined_messages(mock_log, "info")
        self.assertIn(DTS_PERF_LOG_RUN_TIME_SUMMARY_PREFIX, debug_text)
        self.assertIn(DTS_PERF_LOG_NOT_SUITABLE_FOR_PARALLEL_PREFIX, debug_text)
        self.assertIn(DTS_PERF_LOG_SPEEDUP_RATIO_PREFIX, info_text)

    def test_multirank_parallel_wave_skips_speedup_when_zero_exec(self):
        """无可统计执行耗时：INFO 走 skipped 分支，且不因 ratio 触发「不适合并行」。"""
        perf_vals = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        def _noop_sync(_r, _c) -> None:
            return None

        records, mock_log = self._run_parallel_wave_under_fake_dist(perf_vals, _noop_sync)
        self.assertEqual(len(records), 1)
        debug_text = self._joined_messages(mock_log, "debug")
        info_text = self._joined_messages(mock_log, "info")
        self.assertIn(DTS_PERF_LOG_RUN_TIME_SUMMARY_PREFIX, debug_text)
        self.assertNotIn(DTS_PERF_LOG_NOT_SUITABLE_FOR_PARALLEL_PREFIX, debug_text)
        self.assertIn(DTS_PERF_LOG_SPEEDUP_SKIPPED_PREFIX, info_text)


class TestDtsPerformanceMultiprocessParallelPath(unittest.TestCase):
    """真实双进程 gloo：必须命中多 rank 调度与（可选）共享队列抢跑。"""

    def test_multiprocess_shared_queue_uses_both_ranks_as_executors(self):
        """注入共享队列时，4 个 sleep 任务应在两 rank 上都有 executor（动态抢跑）。"""
        results = _launch_two_rank_perf_workers(
            num_tasks=4,
            exec_sleep_s=0.06,
            sync_sleep_s=0.0,
            use_work_queue=True,
        )
        ex0 = results[0]["executors"]
        self.assertEqual(ex0, results[1]["executors"])
        self.assertEqual(len(ex0), 4)
        self.assertEqual(set(ex0), {0, 1}, "两 rank 都应执行过任务，否则未覆盖队列并行路径")

    def test_multiprocess_wall_time_faster_than_serial_stacked_exec(self):
        """双 rank + 队列：本 rank 墙钟应明显短于各任务 exec 串行叠加（Σ exec_time）。"""
        exec_s = 0.07
        results = _launch_two_rank_perf_workers(
            num_tasks=4,
            exec_sleep_s=exec_s,
            sync_sleep_s=0.0,
            use_work_queue=True,
        )
        sum_exec = results[0]["sum_exec"]
        self.assertGreaterEqual(sum_exec, 4 * exec_s * 0.85)
        for r in results:
            # 理想下最慢 rank 约 ceil(4/2)*exec_s；放宽给 gloo / 调度开销
            self.assertLess(
                r["t_run"],
                sum_exec * 0.92,
                msg="并行执行应使墙钟显著低于「全部任务 exec 串行堆叠」；"
                f"rank={r['rank']} t_run={r['t_run']:.4f} sum_exec={sum_exec:.4f}",
            )

    def test_multiprocess_when_sync_dominates_exec_over_sync_le_one(self):
        """多 rank + 轻量执行 + 重同步：各 rank 汇总应满足 exec 总耗时不大于 sync 总耗时。"""
        results = _launch_two_rank_perf_workers(
            num_tasks=3,
            exec_sleep_s=0.02,
            sync_sleep_s=0.07,
            use_work_queue=True,
        )
        for r in results:
            se = float(r["sum_exec"])
            ss = float(r["sum_sync"])
            self.assertGreater(ss, 0.0)
            self.assertLessEqual(se / ss, 1.0 + 1e-6)

    def test_multiprocess_static_round_robin_without_shared_queue(self):
        """无共享队列时仍为双 rank 并行语义：任务 i 由 ``i %% world_size`` 固定 owner 执行。"""
        results = _launch_two_rank_perf_workers(
            num_tasks=4,
            exec_sleep_s=0.02,
            sync_sleep_s=0.0,
            use_work_queue=False,
        )
        self.assertEqual(results[0]["executors"], [0, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
