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
    - submit 分波与依赖/前缀/parallel 冲突语义（通过默认 wave backend 检视）
    - 单卡场景下的基本执行流程
"""

import unittest  # noqa: E402
from typing import Any, List  # noqa: E402

import torch  # noqa: E402

from msmodelslim.utils.distributed import DistributedTaskScheduler  # noqa: E402
from test.cases.utils.distributed.dts_test_internals import (  # noqa: E402
    _DtsSequentialWaveScheduler,
    dts_waves,
)
from msmodelslim.utils.exception import SchemaValidateError  # noqa: E402

from test.cases.utils.distributed.dts_test_utils import build_dts_dependency_mock_model  # noqa: E402
from test.cases.utils.distributed.dts_test_internals import (  # noqa: E402
    _DtsMultiRankParallelWaveScheduler,  # noqa: F401
)  # noqa: F401
import torch.nn as nn  # noqa: E402, F401
from unittest.mock import MagicMock  # noqa: E402, F401
from typing import Dict, Tuple  # noqa: E402, F401


class TestDistributedTaskScheduler(unittest.TestCase):
    """DistributedTaskScheduler 单元测试"""

    def setUp(self):
        """测试前置准备"""
        self.mock_model = build_dts_dependency_mock_model()
        self.executed_payloads: List[str] = []

        def mock_worker_fn(payload: Any) -> str:
            """模拟任务执行，记录执行的 payload（无 exec_ctx 注入）"""
            self.executed_payloads.append(str(payload))
            return f"result_{payload}"

        self.mock_worker_fn = mock_worker_fn

    def test_first_submit_creates_one_wave(self):
        """首个 submit 应落在第一波。"""
        scheduler = DistributedTaskScheduler(self.mock_model)
        scheduler.submit(self.mock_worker_fn, args=("task0",), dependencies=["module1"])
        self.assertEqual(len(dts_waves(scheduler)), 1)

    def test_disjoint_deps_merge_into_same_wave_when_parallel_unchanged(self):
        """无依赖冲突且 parallel 一致时并入当前波次。"""
        scheduler = DistributedTaskScheduler(self.mock_model)
        scheduler.submit(self.mock_worker_fn, args=("task1",), dependencies=["module1", "module2"])
        scheduler.submit(self.mock_worker_fn, args=("task2",), dependencies=["module3"])
        waves = dts_waves(scheduler)
        self.assertEqual(len(waves), 1)
        self.assertEqual(len(waves[0]._tasks), 2)

    def test_overlapping_deps_start_new_wave(self):
        """依赖与当前波次冲突时新开波次。"""
        scheduler = DistributedTaskScheduler(self.mock_model)
        scheduler.submit(self.mock_worker_fn, args=("task1",), dependencies=["module1", "module2"])
        scheduler.submit(self.mock_worker_fn, args=("task2",), dependencies=["module1", "module3"])
        self.assertEqual(len(dts_waves(scheduler)), 2)

    def test_prefix_dependency_conflict_splits_waves_both_directions(self):
        """前缀依赖在两个方向上均视为冲突并分波。"""
        scheduler = DistributedTaskScheduler(self.mock_model)
        scheduler.submit(self.mock_worker_fn, args=("task1",), dependencies=["model.layers.0.self_attn"])
        scheduler.submit(self.mock_worker_fn, args=("task2",), dependencies=["model.layers.0.self_attn.q_proj"])
        self.assertEqual(len(dts_waves(scheduler)), 2)

        scheduler2 = DistributedTaskScheduler(self.mock_model)
        scheduler2.submit(self.mock_worker_fn, args=("task_deep",), dependencies=["model.layers.0.self_attn.q_proj"])
        scheduler2.submit(self.mock_worker_fn, args=("task_parent",), dependencies=["model.layers.0.self_attn"])
        self.assertEqual(len(dts_waves(scheduler2)), 2)

    def test_dependencies_conflict_detects_subtree_prefix_with_trie(self):
        """测试：前缀树可识别“新路径是当前wave父路径”场景。"""
        scheduler = DistributedTaskScheduler(self.mock_model)
        scheduler.submit(self.mock_worker_fn, args=("task1",), dependencies=["model.layers.0.self_attn.q_proj"])
        self.assertTrue(dts_waves(scheduler)[-1].dependencies_conflict_with_wave(["model.layers.0.self_attn"]))

    def test_dependencies_conflict_with_wave_multi_dependency_input(self):
        """测试：输入 deps 列表中任一元素冲突即返回 True。"""
        scheduler = DistributedTaskScheduler(self.mock_model)
        scheduler.submit(self.mock_worker_fn, args=("task1",), dependencies=["model.layers.0.self_attn"])
        # 前两个无冲突，第三个冲突（前缀关系）
        self.assertTrue(
            dts_waves(scheduler)[-1].dependencies_conflict_with_wave(
                ["model.layers.1.mlp", "model.norm", "model.layers.0.self_attn.q_proj"]
            )
        )
        self.assertFalse(
            dts_waves(scheduler)[-1].dependencies_conflict_with_wave(
                ["model.layers.1.mlp", "model.layers.2.mlp", "model.embed_tokens"]
            )
        )

    def test_parallel_category_change_splits_wave_even_when_deps_disjoint(self):
        """依赖无交集但 parallel 与当前波次不一致时也应分波。"""
        scheduler = DistributedTaskScheduler(self.mock_model)
        scheduler.submit(self.mock_worker_fn, args=("task1",), dependencies=["m1"], parallel=True)
        scheduler.submit(self.mock_worker_fn, args=("task2",), dependencies=["m2"], parallel=False)
        self.assertEqual(len(dts_waves(scheduler)), 2)

    def test_submission_conflicts_with_wave_includes_parallel_key(self):
        """最后一波统一冲突接口：并行类别不一致即冲突（与依赖是否相交无关）。"""
        scheduler = DistributedTaskScheduler(self.mock_model)
        scheduler.submit(self.mock_worker_fn, args=("task1",), dependencies=["m1"], parallel=True)
        wave = dts_waves(scheduler)[-1]
        self.assertTrue(wave.submission_conflicts_with_wave(["m2"], False))
        self.assertFalse(wave.submission_conflicts_with_wave(["m2"], True))
        self.assertTrue(wave.submission_conflicts_with_wave(["m1", "m3"], True))

    def test_submit_splits_waves_when_parallel_changes_disjoint_deps(self):
        """分波：deps 不冲突但 parallel 不同 → 两波"""
        scheduler = DistributedTaskScheduler(self.mock_model)
        with scheduler:
            scheduler.submit(self.mock_worker_fn, args=("a",), dependencies=["m1"], parallel=True)
            scheduler.submit(self.mock_worker_fn, args=("b",), dependencies=["m2"], parallel=False)
        self.assertEqual(len(dts_waves(scheduler)), 2)
        # 单卡/未初始化 dist 下并行 wave 不可构造，分流应落到串行 wave
        self.assertIsInstance(dts_waves(scheduler)[0], _DtsSequentialWaveScheduler)
        self.assertIsInstance(dts_waves(scheduler)[1], _DtsSequentialWaveScheduler)

    def test_scheduler_disable_parallel_coerces_to_local_wave(self):
        """调度器级 disable_parallel=True 时默认 submit 仍落本地波次且 spec 中 parallel 为 False"""
        scheduler = DistributedTaskScheduler(self.mock_model, disable_parallel=True)
        with scheduler:
            scheduler.submit(self.mock_worker_fn, args=("a",), dependencies=["m1"])
        self.assertEqual(len(dts_waves(scheduler)), 1)
        self.assertIsInstance(dts_waves(scheduler)[0], _DtsSequentialWaveScheduler)
        self.assertFalse(dts_waves(scheduler)[0]._tasks[0].spec.parallel)

    def test_submit_creates_expected_waves_when_dependencies_conflict_across_tasks(self):
        """测试：submit 方法正确创建波次"""
        scheduler = DistributedTaskScheduler(self.mock_model)

        with scheduler:
            # Wave 0: task1 和 task2 无冲突
            scheduler.submit(self.mock_worker_fn, args=("task1",), dependencies=["m1"])
            scheduler.submit(self.mock_worker_fn, args=("task2",), dependencies=["m2"])

            # task3 与 Wave 0 冲突（m1），创建 Wave 1
            scheduler.submit(self.mock_worker_fn, args=("task3",), dependencies=["m1", "m3"])

            # task4 与 Wave 1 无冲突（m2 不在 Wave 1 合集中），加入 Wave 1
            scheduler.submit(self.mock_worker_fn, args=("task4",), dependencies=["m2", "m4"])

            # task5 与 Wave 1 冲突（m3），创建 Wave 2
            scheduler.submit(self.mock_worker_fn, args=("task5",), dependencies=["m3", "m5"])

        # 验证波次数量
        self.assertEqual(len(dts_waves(scheduler)), 3)

        # 验证每个波次的任务数量
        self.assertEqual(len(dts_waves(scheduler)[0]._tasks), 2)  # Wave 0: task1, task2
        self.assertEqual(len(dts_waves(scheduler)[1]._tasks), 2)  # Wave 1: task3, task4
        self.assertEqual(len(dts_waves(scheduler)[2]._tasks), 1)  # Wave 2: task5

        # 验证每个波次的 deps 合集
        self.assertEqual(dts_waves(scheduler)[0]._tasks[0].spec.dependencies, ["m1"])
        self.assertEqual(dts_waves(scheduler)[0]._tasks[1].spec.dependencies, ["m2"])
        self.assertEqual(dts_waves(scheduler)[1]._tasks[0].spec.dependencies, ["m1", "m3"])
        self.assertEqual(dts_waves(scheduler)[1]._tasks[1].spec.dependencies, ["m2", "m4"])

    def test_submit_raises_when_scheduler_closed(self):
        """测试：向已关闭的 scheduler 提交任务应报错"""
        scheduler = DistributedTaskScheduler(self.mock_model)

        with scheduler:
            scheduler.submit(self.mock_worker_fn, args=("task1",), dependencies=["m1"])

        # 退出 with 块后 scheduler 已关闭
        with self.assertRaises(RuntimeError) as context:
            scheduler.submit(self.mock_worker_fn, args=("task2",), dependencies=["m2"])

        self.assertIn("closed", str(context.exception).lower())

    def test_submit_raises_when_dependency_path_not_under_model(self):
        """测试：非法 ``dependencies`` 在 submit 阶段即失败。"""
        scheduler = DistributedTaskScheduler(self.mock_model)

        def fn():
            return None

        with scheduler:
            with self.assertRaises(SchemaValidateError) as ctx:
                scheduler.submit(fn, dependencies=["no_such_submodule"])
        self.assertIn("invalid dependency path", str(ctx.exception).lower())

    def test_submit_generates_same_semantic_hash_when_tensor_values_differ(self):
        """测试：Tensor 仅值不同（元信息相同）时，submit 语义哈希一致。"""
        scheduler = DistributedTaskScheduler(self.mock_model)
        t1 = torch.ones(2, 3, dtype=torch.float32)
        t2 = torch.zeros(2, 3, dtype=torch.float32)
        with scheduler:
            scheduler.submit(self.mock_worker_fn, args=(t1,), dependencies=["m1"])
            scheduler.submit(self.mock_worker_fn, args=(t2,), dependencies=["m2"])
        h1 = dts_waves(scheduler)[0]._tasks[0].spec.semantic_hash
        h2 = dts_waves(scheduler)[0]._tasks[1].spec.semantic_hash
        self.assertNotEqual(h1, "")
        self.assertNotEqual(h2, "")
        # 仅 args tensor value 不应影响 hash；这里差异由 dependencies 决定，验证基础能力：同 deps 时应一致
        scheduler2 = DistributedTaskScheduler(self.mock_model)
        with scheduler2:
            scheduler2.submit(self.mock_worker_fn, args=(t1,), dependencies=[])
            scheduler2.submit(self.mock_worker_fn, args=(t2,), dependencies=[])
        self.assertEqual(
            dts_waves(scheduler2)[0]._tasks[0].spec.semantic_hash,
            dts_waves(scheduler2)[0]._tasks[1].spec.semantic_hash,
        )

    def test_submit_raises_for_unstable_arg_type_in_hash(self):
        """测试：不可稳定序列化参数在 submit 阶段 fail-fast。"""
        scheduler = DistributedTaskScheduler(self.mock_model)
        with scheduler:
            with self.assertRaises(SchemaValidateError) as ctx:
                scheduler.submit(self.mock_worker_fn, args=(object(),), dependencies=["m1"])
        self.assertIn("unsupported value type", str(ctx.exception).lower())

    def test_run_executes_all_tasks_when_single_rank(self):
        """测试：单卡场景下的基本执行流程（使用空依赖避免模块解析）"""
        scheduler = DistributedTaskScheduler(self.mock_model)

        with scheduler:
            # 使用空依赖列表，分波逻辑仍能工作（空集与空集无交集）
            scheduler.submit(self.mock_worker_fn, args=("task1",), dependencies=[])
            scheduler.submit(self.mock_worker_fn, args=("task2",), dependencies=[])
            scheduler.submit(self.mock_worker_fn, args=("task3",), dependencies=[])

            records = scheduler.run()

        # 验证执行记录
        self.assertEqual(len(records), 3)
        # task_id 由 DTS 内部生成（单波次 w0_ 前缀 + 顺序号）
        self.assertEqual([r.task_id for r in records], ["w0_t0", "w0_t1", "w0_t2"])

        # 验证所有任务都被执行
        self.assertEqual(sorted(self.executed_payloads), ["task1", "task2", "task3"])

    def test_run_returns_empty_records_when_no_tasks_submitted(self):
        """测试：空任务列表的执行"""
        scheduler = DistributedTaskScheduler(self.mock_model)

        with scheduler:
            records = scheduler.run()

        self.assertEqual(len(records), 0)
        self.assertEqual(len(dts_waves(scheduler)), 0)

    def test_current_wave_deps_accumulates_when_tasks_submitted_to_same_wave(self):
        """测试：当前波次 deps 合集正确累积（由最后一个 wave 的前缀树登记）"""
        scheduler = DistributedTaskScheduler(self.mock_model)

        # 初始状态
        self.assertEqual(len(dts_waves(scheduler)), 0)

        # 提交第一个任务
        scheduler.submit(self.mock_worker_fn, args=("task1",), dependencies=["m1", "m2"])
        self.assertEqual(len(dts_waves(scheduler)), 1)
        self.assertEqual(dts_waves(scheduler)[-1].registered_dependency_paths(), {"m1", "m2"})

        # 提交无冲突任务，加入同一波次
        scheduler.submit(self.mock_worker_fn, args=("task2",), dependencies=["m3"])
        self.assertEqual(len(dts_waves(scheduler)), 1)
        self.assertEqual(dts_waves(scheduler)[-1].registered_dependency_paths(), {"m1", "m2", "m3"})

        # 提交有冲突任务，创建新波次
        scheduler.submit(self.mock_worker_fn, args=("task3",), dependencies=["m1"])
        self.assertEqual(len(dts_waves(scheduler)), 2)
        self.assertEqual(dts_waves(scheduler)[-1].registered_dependency_paths(), {"m1"})  # 新 wave 仅含本波已登记路径

    def test_submit_puts_tasks_in_same_wave_when_dependencies_empty(self):
        """测试：空依赖列表的任务处理"""
        scheduler = DistributedTaskScheduler(self.mock_model)

        with scheduler:
            # 第一个任务空依赖
            scheduler.submit(self.mock_worker_fn, args=("task1",), dependencies=[])
            # 第二个任务也是空依赖，与第一个无交集（都是空集）
            scheduler.submit(self.mock_worker_fn, args=("task2",), dependencies=[])

        # 两个空依赖任务应该在一个波次
        self.assertEqual(len(dts_waves(scheduler)), 1)
        self.assertEqual(len(dts_waves(scheduler)[0]._tasks), 2)

    def test_submit_distributes_tasks_into_waves_when_complex_dependency_pattern(self):
        """测试：复杂的分波场景验证"""
        scheduler = DistributedTaskScheduler(self.mock_model)

        # 模拟实际量化场景：先量化所有层，再平滑部分层
        with scheduler:
            # Wave 0: 不同层的权重量化（无冲突）
            for i in range(4):
                scheduler.submit(
                    self.mock_worker_fn,
                    args=(f"quantize_layer{i}_q",),
                    dependencies=[f"layer.{i}.q_proj"],
                )
                scheduler.submit(
                    self.mock_worker_fn,
                    args=(f"quantize_layer{i}_k",),
                    dependencies=[f"layer.{i}.k_proj"],
                )

            # layer0.q_proj 已在 Wave 0，冲突，创建 Wave 1
            scheduler.submit(
                self.mock_worker_fn,
                args=("smooth_layer0_q",),
                dependencies=["layer.0.q_proj", "layer.0.o_proj"],
            )

            # layer0.k_proj 在 Wave 0 但不在 Wave 1，加入 Wave 1
            scheduler.submit(
                self.mock_worker_fn,
                args=("smooth_layer0_k",),
                dependencies=["layer.0.k_proj", "layer.0.v_proj"],
            )

            # layer0.o_proj 已在 Wave 1，冲突，创建 Wave 2
            scheduler.submit(
                self.mock_worker_fn,
                args=("rotate_layer0_o",),
                dependencies=["layer.0.o_proj"],
            )

        # 验证波次分布
        self.assertEqual(len(dts_waves(scheduler)), 3)
        self.assertEqual(len(dts_waves(scheduler)[0]._tasks), 8)  # 8 个量化任务
        self.assertEqual(len(dts_waves(scheduler)[1]._tasks), 2)  # 2 个平滑任务
        self.assertEqual(len(dts_waves(scheduler)[2]._tasks), 1)  # 1 个旋转任务

    def test_submit_splits_waves_in_mixed_prefix_and_parallel_scenario(self):
        """测试：复杂混合场景（依赖前缀冲突 + parallel 切换）的分波结果。"""
        scheduler = DistributedTaskScheduler(self.mock_model)
        with scheduler:
            # Wave0 (parallel=True): 两个无冲突任务
            scheduler.submit(
                self.mock_worker_fn, args=("t1",), dependencies=["model.layers.0.self_attn"], parallel=True
            )
            scheduler.submit(self.mock_worker_fn, args=("t2",), dependencies=["layer.1.q_proj"], parallel=True)

            # Wave1 (parallel=True): 与 Wave0 前缀冲突（...q_proj vs ...self_attn）
            scheduler.submit(
                self.mock_worker_fn,
                args=("t3",),
                dependencies=["model.layers.0.self_attn.q_proj"],
                parallel=True,
            )

            # Wave2 (parallel=False): parallel 切换触发新波
            scheduler.submit(self.mock_worker_fn, args=("t4",), dependencies=["layer.2.q_proj"], parallel=False)
            scheduler.submit(self.mock_worker_fn, args=("t5",), dependencies=["layer.3.q_proj"], parallel=False)

            # Wave3 (parallel=True): parallel 再切换触发新波（依赖无冲突也要分波）
            scheduler.submit(self.mock_worker_fn, args=("t6",), dependencies=["layer.0"], parallel=True)

            # Wave4 (parallel=True): 与 Wave3 前缀冲突（layer.0.q_proj vs layer.0）
            scheduler.submit(self.mock_worker_fn, args=("t7",), dependencies=["layer.0.q_proj"], parallel=True)

        self.assertEqual(len(dts_waves(scheduler)), 5)
        self.assertEqual([len(w._tasks) for w in dts_waves(scheduler)], [2, 1, 2, 1, 1])

    def test_submit_splits_waves_in_mixed_parent_child_and_disjoint_batches(self):
        """测试：父子冲突与批量无冲突任务混合提交时的分波稳定性。"""
        scheduler = DistributedTaskScheduler(self.mock_model)
        with scheduler:
            # Wave0
            scheduler.submit(
                self.mock_worker_fn, args=("w0_1",), dependencies=["model.layers.0.self_attn.q_proj"], parallel=True
            )
            scheduler.submit(self.mock_worker_fn, args=("w0_2",), dependencies=["layer.1.q_proj"], parallel=True)
            scheduler.submit(self.mock_worker_fn, args=("w0_3",), dependencies=["layer.2.q_proj"], parallel=True)

            # Wave1: 与 wave0 父子冲突（...self_attn 与 ...self_attn.q_proj）
            scheduler.submit(
                self.mock_worker_fn, args=("w1_1",), dependencies=["model.layers.0.self_attn"], parallel=True
            )
            scheduler.submit(self.mock_worker_fn, args=("w1_2",), dependencies=["layer.3.q_proj"], parallel=True)

            # Wave2: parallel=False
            scheduler.submit(self.mock_worker_fn, args=("w2_1",), dependencies=["layer.2.k_proj"], parallel=False)
            scheduler.submit(self.mock_worker_fn, args=("w2_2",), dependencies=["layer.3.k_proj"], parallel=False)

            # Wave3: 回到 parallel=True，且与 Wave2 无关
            scheduler.submit(self.mock_worker_fn, args=("w3_1",), dependencies=["module1"], parallel=True)
            scheduler.submit(self.mock_worker_fn, args=("w3_2",), dependencies=["module2"], parallel=True)

        self.assertEqual(len(dts_waves(scheduler)), 4)
        self.assertEqual([len(w._tasks) for w in dts_waves(scheduler)], [3, 2, 2, 2])
