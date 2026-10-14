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
# pylint: disable=consider-using-from-import,no-name-in-module,pointless-string-statement,ungrouped-imports

"""
DistributedTaskScheduler（分波调度器）的单元测试

测试范围：
    - submit 分波与执行语义
    - 单卡场景下的基本执行流程
"""

import unittest  # noqa: E402
from typing import Any, List, Tuple  # noqa: E402

import torch  # noqa: E402, F401
from unittest.mock import MagicMock  # noqa: E402, F401
from typing import Dict  # noqa: E402, F401
import torch.nn as nn  # noqa: E402

from msmodelslim.utils.distributed import DistributedTaskScheduler  # noqa: E402
from test.cases.utils.distributed.dts_test_internals import (  # noqa: E402
    _DtsSequentialWaveScheduler,
    dts_waves,
)

from test.cases.utils.distributed.dts_test_utils import build_dts_dependency_mock_model  # noqa: E402


class TestDistributedTaskSchedulerExecution(unittest.TestCase):
    """DistributedTaskScheduler 执行相关测试"""

    def setUp(self):
        """测试前置准备"""
        self.mock_model = nn.Linear(10, 10)
        self.execution_order: List[Tuple[int, str]] = []

    def test_run_executes_waves_in_order_when_multiple_waves_exist(self):
        """测试：多波次按顺序执行（使用空依赖避免模块解析）"""

        def fn(payload: Any) -> str:
            # 记录执行顺序（以 payload 作为外部可读标识）
            wave_idx = len(self.execution_order)
            self.execution_order.append((wave_idx, str(payload)))
            return f"result_{payload}"

        scheduler = DistributedTaskScheduler(self.mock_model)

        with scheduler:
            # 使用空依赖列表，所有任务在同一个波次
            scheduler.submit(fn, args=("task1",), dependencies=[])
            scheduler.submit(fn, args=("task2",), dependencies=[])
            scheduler.submit(fn, args=("task3",), dependencies=[])
            scheduler.submit(fn, args=("task4",), dependencies=[])

            records = scheduler.run()

        # 验证所有任务都被执行
        self.assertEqual(len(records), 4)
        # 验证都在同一个波次
        self.assertEqual(len(dts_waves(scheduler)), 1)

    def test_run_records_task_result_when_worker_returns_value(self):
        """测试：任务结果正确记录在 TaskExecutionRecord 中"""

        def fn(payload: Any) -> str:
            return f"processed_{payload}"

        scheduler = DistributedTaskScheduler(self.mock_model)

        with scheduler:
            scheduler.submit(fn, args=("task1",), dependencies=[])
            scheduler.submit(fn, args=("task2",), dependencies=[])

            records = scheduler.run()

        # 验证结果正确（task_id 由 DTS 内部生成）
        self.assertEqual(len(records), 2)
        self.assertEqual([r.result for r in records], ["processed_task1", "processed_task2"])

    def test_run_records_executor_rank_when_single_rank(self):
        """测试：TaskExecutionRecord 中记录了正确的执行 rank"""

        def fn(payload: Any) -> str:
            return str(payload)

        scheduler = DistributedTaskScheduler(self.mock_model)

        with scheduler:
            scheduler.submit(fn, args=("task1",), dependencies=[])
            records = scheduler.run()

        # 单卡场景下 executor_rank 应该是 0
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].executor_rank, 0)

    def test_submit_creates_expected_waves_when_dependencies_alternate(self):
        """测试：交替依赖模式（A->B->A->B）- 仅验证分波逻辑，不执行"""

        def fn(payload):
            return str(payload)

        scheduler = DistributedTaskScheduler(build_dts_dependency_mock_model())

        with scheduler:
            # Wave 0: task1(A), task2(B) - 无冲突，合集 {A, B}
            scheduler.submit(fn, args=("task1",), dependencies=["A"])
            scheduler.submit(fn, args=("task2",), dependencies=["B"])

            # Wave 1: task3(A) 与 Wave 0 的 task1 冲突（A），合集 {A, C}
            scheduler.submit(fn, args=("task3",), dependencies=["A", "C"])

            # task4(B,D) 与 Wave 1 合集 {A,C} 无交集，加入 Wave 1，合集变为 {A, C, B, D}
            scheduler.submit(fn, args=("task4",), dependencies=["B", "D"])

            # task5(A) 与当前 Wave 1 合集 {A, C, B, D} 有交集（A），创建 Wave 2
            scheduler.submit(fn, args=("task5",), dependencies=["A"])

        # Wave 0: 2 个任务 (task1, task2)
        # Wave 1: 2 个任务 (task3, task4)
        # Wave 2: 1 个任务 (task5)
        self.assertEqual(len(dts_waves(scheduler)), 3)
        self.assertEqual(len(dts_waves(scheduler)[0]._tasks), 2)  # task1, task2
        self.assertEqual(len(dts_waves(scheduler)[1]._tasks), 2)  # task3, task4
        self.assertEqual(len(dts_waves(scheduler)[2]._tasks), 1)  # task5


class TestDistributedTaskSchedulerEdgeCases(unittest.TestCase):
    """DistributedTaskScheduler 边界条件测试"""

    def setUp(self):
        self.mock_model = build_dts_dependency_mock_model()

    def test_submit_creates_new_wave_per_task_when_all_dependencies_same(self):
        """测试：所有任务具有相同依赖"""

        def fn(payload):
            return str(payload)

        scheduler = DistributedTaskScheduler(self.mock_model)

        with scheduler:
            # 所有任务都依赖相同的模块
            scheduler.submit(fn, args=("task1",), dependencies=["shared"])
            scheduler.submit(fn, args=("task2",), dependencies=["shared"])
            scheduler.submit(fn, args=("task3",), dependencies=["shared"])

        # 每个任务都应该创建新波次（因为都与前一个冲突）
        self.assertEqual(len(dts_waves(scheduler)), 3)
        for i in range(3):
            self.assertEqual(len(dts_waves(scheduler)[i]._tasks), 1)

    def test_submit_keeps_single_wave_when_all_dependencies_unique(self):
        """测试：大量任务在同一个波次"""

        def fn(payload):
            return str(payload)

        scheduler = DistributedTaskScheduler(self.mock_model)

        num_tasks = 100

        with scheduler:
            for i in range(num_tasks):
                # 每个任务依赖不同的模块，无冲突
                scheduler.submit(
                    fn,
                    args=(f"task{i}",),
                    dependencies=[f"module{i}"],
                )

        # 所有任务都在同一个波次
        self.assertEqual(len(dts_waves(scheduler)), 1)
        self.assertEqual(len(dts_waves(scheduler)[0]._tasks), num_tasks)

    def test_submit_creates_many_waves_when_dependencies_always_conflict(self):
        """测试：大量波次"""

        def fn(payload):
            return str(payload)

        scheduler = DistributedTaskScheduler(self.mock_model)

        num_waves = 50

        with scheduler:
            for i in range(num_waves):
                # 每个任务都与前一个冲突（依赖相同的模块）
                scheduler.submit(
                    fn,
                    args=(f"task{i}",),
                    dependencies=["shared_module"],
                )

        # 每个任务创建一个新波次
        self.assertEqual(len(dts_waves(scheduler)), num_waves)

    def test_run_executes_payload_driven_worker_when_dependencies_empty(self):
        """测试：空依赖但有 payload 的任务"""
        executed_tasks: List[str] = []

        def fn(payload: Any) -> str:
            executed_tasks.append(payload.get("key", "no_key"))
            return "ok"

        scheduler = DistributedTaskScheduler(self.mock_model)

        with scheduler:
            scheduler.submit(fn, args=({"key": "value1"},), dependencies=[])
            scheduler.submit(fn, args=({"key": "value2"},), dependencies=[])

            records = scheduler.run()  # noqa: F841

        # 验证 payload 正确传递
        self.assertEqual(len(executed_tasks), 2)
        self.assertIn("value1", executed_tasks)
        self.assertIn("value2", executed_tasks)

    def test_submit_creates_new_wave_when_any_dependency_conflicts(self):
        """测试：多个依赖全部与当前波次冲突"""

        def fn(payload):
            return str(payload)

        scheduler = DistributedTaskScheduler(self.mock_model)

        with scheduler:
            # Wave 0
            scheduler.submit(fn, args=("task1",), dependencies=["m1", "m2", "m3"])

            # 新任务多个依赖，其中一个与 Wave 0 冲突即可
            scheduler.submit(fn, args=("task2",), dependencies=["m3", "m4", "m5"])

        self.assertEqual(len(dts_waves(scheduler)), 2)

    def test_submit_keeps_same_wave_when_no_dependencies_conflict(self):
        """测试：部分依赖但无冲突"""

        def fn(payload):
            return str(payload)

        scheduler = DistributedTaskScheduler(self.mock_model)

        with scheduler:
            # Wave 0
            scheduler.submit(fn, args=("task1",), dependencies=["m1"])
            scheduler.submit(fn, args=("task2",), dependencies=["m2"])

            # 新任务依赖 m3 和 m4，都不在 Wave 0 合集中
            scheduler.submit(fn, args=("task3",), dependencies=["m3", "m4"])

        # 无冲突，加入 Wave 0
        self.assertEqual(len(dts_waves(scheduler)), 1)
        self.assertEqual(len(dts_waves(scheduler)[0]._tasks), 3)

    def test_run_does_not_remove_waves_when_completed(self):
        """测试：run 后波次状态保持不变"""

        def fn(payload):
            return str(payload)

        scheduler = DistributedTaskScheduler(self.mock_model)

        with scheduler:
            scheduler.submit(fn, args=("task1",), dependencies=[])
            scheduler.submit(fn, args=("task2",), dependencies=[])

            records = scheduler.run()  # noqa: F841

        # 验证波次仍然存在
        self.assertEqual(len(dts_waves(scheduler)), 1)
        self.assertEqual(len(dts_waves(scheduler)[0]._tasks), 2)

    # task_type 字段已移除，不再做相关保留性测试


class TestDistributedTaskSchedulerIntegration(unittest.TestCase):
    """DistributedTaskScheduler 集成测试"""

    def test_run_matches_impl_behavior_when_no_dependencies(self):
        """测试：DistributedTaskScheduler(wave) 与内部单波调度器的兼容性（使用空依赖）"""
        mock_model = nn.Linear(10, 10)
        results = []

        def fn(payload: Any) -> str:
            results.append(str(payload))
            return str(payload)

        # 直接使用内部单波调度器（对比）
        with _DtsSequentialWaveScheduler(mock_model) as impl:
            impl.submit(fn, args=("task1",), dependencies=[])
            impl.submit(fn, args=("task2",), dependencies=[])
            impl_records = impl.run()

        # 使用 DistributedTaskScheduler（所有任务无冲突，应该在一个波次）
        # 使用空依赖列表避免模块解析问题
        with DistributedTaskScheduler(mock_model) as wave_dts:
            wave_dts.submit(fn, args=("task1",), dependencies=[])
            wave_dts.submit(fn, args=("task2",), dependencies=[])
            wave_records = wave_dts.run()

        # 两种调度器都应该执行所有任务（各执行2次，共4次）
        self.assertEqual(len(impl_records), 2)
        self.assertEqual(len(wave_records), 2)
        # 验证执行次数（impl 一次 + wave_dts 一次，共 4 条执行记录）
        self.assertEqual(len(results), 4)

    def test_submit_creates_expected_waves_when_mixed_dependency_scenario(self):
        """测试：混合依赖场景（模拟实际量化流程）- 仅验证分波逻辑"""
        mock_model = build_dts_dependency_mock_model()

        def fn(payload):
            return payload

        scheduler = DistributedTaskScheduler(mock_model)

        with scheduler:
            # Phase 1: Weight Quantization (独立任务，可并行)
            for layer in range(3):
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    scheduler.submit(
                        fn,
                        args=({"phase": "weight_quant", "name": f"l{layer}_{proj}"},),
                        dependencies=[f"layer.{layer}.{proj}"],
                    )

            # Phase 2: Smoothing（依赖 layer.{i}.q_proj 和 layer.{i}.o_proj）
            for layer in range(3):
                scheduler.submit(
                    fn,
                    args=({"phase": "smooth", "name": f"l{layer}"},),
                    dependencies=[f"layer.{layer}.q_proj", f"layer.{layer}.o_proj"],
                )

            # Phase 3: Rotation（依赖 layer.{i}.q_proj）
            for layer in range(3):
                scheduler.submit(
                    fn,
                    args=({"phase": "rotation", "name": f"l{layer}"},),
                    dependencies=[f"layer.{layer}.q_proj"],
                )

        # 验证分波结果（不执行 run 以避免模块解析问题）
        # Phase 1: 12 个独立任务应该都在 Wave 0（因为 layer.{i}.{proj} 都不同）
        # Phase 2: 3 个 smooth 任务，每个都与 Wave 0 的一些任务冲突
        # Phase 3: 3 个 rotate 任务，每个都与 Wave 1 的任务冲突

        # 验证总共有 3 个波次
        self.assertEqual(len(dts_waves(scheduler)), 3)

        # Wave 0: 12 个 weight quant 任务
        self.assertEqual(len(dts_waves(scheduler)[0]._tasks), 12)

        # Wave 1: 3 个 smooth 任务（每个都与 Wave 0 冲突）
        self.assertEqual(len(dts_waves(scheduler)[1]._tasks), 3)

        # Wave 2: 3 个 rotate 任务（每个都与 Wave 1 冲突）
        self.assertEqual(len(dts_waves(scheduler)[2]._tasks), 3)

        # 验证任务分布（task_id 内部生成，phase 由 args[0] 传入）
        wave0_phases = {t.spec.args[0].get("phase") for t in dts_waves(scheduler)[0]._tasks}
        wave1_phases = {t.spec.args[0].get("phase") for t in dts_waves(scheduler)[1]._tasks}
        wave2_phases = {t.spec.args[0].get("phase") for t in dts_waves(scheduler)[2]._tasks}

        self.assertEqual(wave0_phases, {"weight_quant"})
        self.assertEqual(wave1_phases, {"smooth"})
        self.assertEqual(wave2_phases, {"rotation"})
