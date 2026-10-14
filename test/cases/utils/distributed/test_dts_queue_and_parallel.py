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
# pylint: disable=consider-using-from-import,no-name-in-module,pointless-string-statement

"""
DistributedTaskScheduler（分波调度器）的单元测试

测试范围：
    - 分布式任务队列与并行相关语义

多 rank「并行」相关用例：``dts_distributed_spawn.run_distributed_spawn``（内部 ``torch.multiprocessing.spawn``）
+子进程 ``gloo`` + ``file://``。
"""

import unittest  # noqa: E402
from typing import Any, Dict, List, Tuple  # noqa: E402, F401
from unittest.mock import MagicMock  # noqa: E402

from msmodelslim.utils.distributed import DistributedTaskScheduler  # noqa: E402, F401
import torch.nn as nn  # noqa: E402, F401
import torch  # noqa: E402, F401

from test.cases.utils.distributed.dts_distributed_spawn import (  # noqa: E402
    run_distributed_spawn,
    start_queue_result_collector,
    spawn_queue,
)
from test.cases.utils.distributed.test_dts_scheduler_test_workers import (  # noqa: E402
    _run_disable_parallel_worker,
    _run_dts_dp_serial_oracle_equivalence_worker,
    _run_dts_heterogeneous_submit_supported_worker,
    _run_dts_submit_hash_mismatch_worker,
    _run_dts_submit_hash_tensor_meta_worker,
)


class TestDistributedTaskSchedulerQueue(unittest.TestCase):
    """分布式任务队列操作单元测试

    测试范围：
        - 队列设置与获取
        - 队列清理
        - 初始状态
    """

    def setUp(self):
        """测试前置准备：确保队列初始为 None"""
        from msmodelslim.utils.distributed import clear_distributed_task_work_queue

        clear_distributed_task_work_queue()

    def tearDown(self):
        """测试后置清理"""
        from msmodelslim.utils.distributed import clear_distributed_task_work_queue

        clear_distributed_task_work_queue()

    def test_get_distributed_task_work_queue_returns_set_queue_when_set(self):
        """测试：设置和获取分布式任务工作队列"""
        from msmodelslim.utils.distributed import (
            get_distributed_task_work_queue,
            set_distributed_task_work_queue,
        )

        mock_queue = MagicMock()

        set_distributed_task_work_queue(mock_queue)
        result = get_distributed_task_work_queue()

        self.assertEqual(result, mock_queue)

    def test_clear_distributed_task_work_queue_returns_none_when_cleared(self):
        """测试：清理队列重置全局变量"""
        from msmodelslim.utils.distributed import (
            clear_distributed_task_work_queue,
            get_distributed_task_work_queue,
            set_distributed_task_work_queue,
        )

        mock_queue = MagicMock()
        set_distributed_task_work_queue(mock_queue)

        clear_distributed_task_work_queue()
        result = get_distributed_task_work_queue()

        self.assertIsNone(result)

    def test_get_distributed_task_work_queue_returns_none_when_never_set(self):
        """测试：初始状态队列为 None"""
        from msmodelslim.utils.distributed import get_distributed_task_work_queue

        result = get_distributed_task_work_queue()

        self.assertIsNone(result)


class TestSharedTaskParallelControl(unittest.TestCase):
    """内部单波调度器与 DistributedTaskScheduler 的 disable_parallel 行为验证（spawn + gloo）。"""

    def test_disable_parallel_skips_sync_when_disable_parallel_true(self):
        world_size = 2
        results_queue = spawn_queue()
        results_future = start_queue_result_collector(results_queue, expected_results=world_size)
        run_distributed_spawn(
            world_size,
            _run_disable_parallel_worker,
            (results_queue,),
            init_dir_prefix="dts_disable_parallel_gloo_",
        )
        results = results_future.result(timeout_s=120.0)

        results_by_rank = {r["rank"]: r for r in results}

        # default：owner-only 执行 + 同步到 all ranks
        self.assertEqual(results_by_rank[0]["impl_default"]["shared1"], 100.0)
        self.assertEqual(results_by_rank[1]["impl_default"]["shared1"], 100.0)
        self.assertEqual(results_by_rank[0]["impl_default"]["shared2"], 201.0)
        self.assertEqual(results_by_rank[1]["impl_default"]["shared2"], 201.0)

        # disable：所有 rank 都执行，且跳过同步 => shared 保持各自不同
        self.assertEqual(results_by_rank[0]["impl_disable"]["shared1"], 100.0)
        self.assertEqual(results_by_rank[1]["impl_disable"]["shared1"], 101.0)
        self.assertEqual(results_by_rank[0]["impl_disable"]["shared2"], 200.0)
        self.assertEqual(results_by_rank[1]["impl_disable"]["shared2"], 201.0)

        # Wave 包装一致性（该用例无 deps 冲突，等价于单波次）
        self.assertEqual(results_by_rank[0]["wave_default"]["shared1"], 100.0)
        self.assertEqual(results_by_rank[1]["wave_default"]["shared1"], 100.0)
        self.assertEqual(results_by_rank[0]["wave_default"]["shared2"], 201.0)
        self.assertEqual(results_by_rank[1]["wave_default"]["shared2"], 201.0)
        self.assertEqual(results_by_rank[0]["wave_disable"]["shared1"], 100.0)
        self.assertEqual(results_by_rank[1]["wave_disable"]["shared1"], 101.0)
        self.assertEqual(results_by_rank[0]["wave_disable"]["shared2"], 200.0)
        self.assertEqual(results_by_rank[1]["wave_disable"]["shared2"], 201.0)


class TestDtsDpSerialOracleEquivalence(unittest.TestCase):
    """DP 语义：多卡并行 DTS 终态与严格串行 oracle 一致（顺序敏感就地更新）。"""

    _SEED_PAIRS = (
        (20260211, 424242),
        (77331, 991881),
    )

    def test_multirank_parallel_dts_matches_serial_oracle_state_dict(self):
        world_size = 2
        for task_seed, init_seed in self._SEED_PAIRS:
            with self.subTest(task_seed=task_seed, init_seed=init_seed):
                results_queue = spawn_queue()
                results_future = start_queue_result_collector(results_queue, expected_results=world_size)
                run_distributed_spawn(
                    world_size,
                    _run_dts_dp_serial_oracle_equivalence_worker,
                    (results_queue, task_seed, init_seed),
                    init_dir_prefix=f"dts_dp_oracle_equiv_gloo_{task_seed}_{init_seed}_",
                )
                results = results_future.result(timeout_s=120.0)
                by_rank = {int(r["rank"]): r for r in results}
                for r in range(world_size):
                    self.assertIn(r, by_rank, msg=f"missing rank {r} in results")
                    self.assertTrue(by_rank[r].get("ok"), msg=by_rank[r])
                self.assertEqual(by_rank[0]["worst_key"], by_rank[1]["worst_key"])
                self.assertEqual(by_rank[0]["worst"], by_rank[1]["worst"])


class TestDtsSubmitSemanticHash(unittest.TestCase):
    def test_multirank_submit_hash_mismatch_raises(self):
        world_size = 2
        results_queue = spawn_queue()
        results_future = start_queue_result_collector(results_queue, expected_results=world_size)
        run_distributed_spawn(
            world_size,
            _run_dts_submit_hash_mismatch_worker,
            (results_queue,),
            init_dir_prefix="dts_submit_hash_mismatch_",
        )
        results = results_future.result(timeout_s=120.0)
        for r in results:
            self.assertTrue(r.get("ok"), msg=r)
            self.assertIn("semantic mismatch", (r.get("error") or "").lower())

    def test_multirank_tensor_value_diff_keeps_same_hash_semantics(self):
        world_size = 2
        results_queue = spawn_queue()
        results_future = start_queue_result_collector(results_queue, expected_results=world_size)
        run_distributed_spawn(
            world_size,
            _run_dts_submit_hash_tensor_meta_worker,
            (results_queue,),
            init_dir_prefix="dts_submit_hash_tensor_meta_",
        )
        results = results_future.result(timeout_s=120.0)
        for r in results:
            self.assertTrue(r.get("ok"), msg=r)
            self.assertEqual(r.get("record_count"), 1)


class TestDtsHeterogeneousSubmitSupport(unittest.TestCase):
    def test_multirank_heterogeneous_submit_order_is_supported_without_error(self):
        world_size = 2
        results_queue = spawn_queue()
        results_future = start_queue_result_collector(results_queue, expected_results=world_size)
        run_distributed_spawn(
            world_size,
            _run_dts_heterogeneous_submit_supported_worker,
            (results_queue,),
            init_dir_prefix="dts_heterogeneous_submit_supported_",
        )
        results = results_future.result(timeout_s=120.0)
        by_rank = {int(r["rank"]): r for r in results}
        self.assertEqual(set(by_rank.keys()), {0, 1})
        for r in by_rank.values():
            self.assertTrue(r.get("ok"), msg=r)
            self.assertEqual(r.get("record_count"), 10)
            self.assertEqual(r.get("wave_count"), 2)
            # 至少有 shared 任务走 owner 语义执行（不是 all-local 全同 rank）
            self.assertTrue(len(set(r.get("executor_ranks", []))) >= 2, msg=r)
        # rank0: local->share, rank1: share->local，分波顺序不同但均可执行
        self.assertEqual(by_rank[0].get("wave_task_counts"), [2, 8])
        self.assertEqual(by_rank[1].get("wave_task_counts"), [8, 2])


if __name__ == "__main__":
    unittest.main()
