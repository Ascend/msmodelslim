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
DistributedTaskScheduler 集合通信检测单元测试

测试范围：
    - _collective_op_guard 在非分布式环境下的行为（no-op）
    - _collective_op_guard 在多 rank 下检测非法集合通信
    - 通过 DTS 调度器执行含集合通信的任务时正确报错
"""

import inspect
import os
import unittest
from typing import Any, Dict, List, Tuple

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
import torch.distributed as dist
import torch.nn as nn

from msmodelslim.utils.distributed import DistributedTaskScheduler
from msmodelslim.utils.distributed.task_scheduler.backend.wave import _collective_op_guard
from msmodelslim.utils.exception import SchemaValidateError
from test.cases.utils.distributed.dts_distributed_spawn import (
    distributed_test,
    run_distributed_spawn,
)
from test.cases.utils.distributed.dts_test_internals import (
    _TaskSpec,
    _DtsMultiRankParallelWaveScheduler,
    _DtsSequentialWaveScheduler,
)


class TestCollectiveOpGuardNonDistributed(unittest.TestCase):
    """_collective_op_guard 在非分布式环境下的行为测试。"""

    def test_noop_when_not_initialized(self):
        """未初始化 dist 时 guard 不应拦截任何操作。"""
        with _collective_op_guard():
            result = 1 + 1
        self.assertEqual(result, 2)

    def test_noop_with_torch_ops(self):
        """未初始化 dist 时 torch 张量操作正常执行。"""
        with _collective_op_guard():
            t = torch.ones(3, 3)
        self.assertEqual(t.shape, (3, 3))

    def test_noop_on_function_call(self):
        """未初始化 dist 时任意函数调用正常执行。"""
        def _helper(x):
            return x * 2

        with _collective_op_guard():
            out = _helper(42)
        self.assertEqual(out, 84)

    def test_noop_with_model_forward(self):
        """未初始化 dist 时模型前向正常执行。"""
        model = nn.Linear(4, 4)
        x = torch.randn(2, 4)
        with _collective_op_guard():
            y = model(x)
        self.assertEqual(y.shape, (2, 4))


def _verify_guard_detects_collective(rank: int, world_size: int, collective_name: str):
    """通用验证函数：在 guard 内调用指定集合通信应抛出 SchemaValidateError。"""
    t = torch.ones(3)

    with _collective_op_guard():
        # 每次通过 getattr 获取 dist 上的函数，确保 guard 的替换生效
        fn = getattr(dist, collective_name, None)
        if fn is None:
            return  # 跳过该环境不支持的集合通信

        if collective_name == "broadcast":
            fn(t, src=0)
        elif collective_name in ("barrier", "monitored_barrier"):
            fn()
        elif collective_name == "all_reduce":
            fn(t)
        elif collective_name in ("all_gather", "all_gather_into_tensor"):
            out = [torch.zeros(3) for _ in range(world_size)]
            fn(out, t)
        elif collective_name == "reduce_scatter":
            out = torch.zeros(3)
            fn(out, [torch.ones(3) for _ in range(world_size)])
        elif collective_name == "all_to_all":
            out = [torch.zeros(1) for _ in range(world_size)]
            fn(out, [torch.ones(1) for _ in range(world_size)])
        elif collective_name == "all_to_all_single":
            out = torch.zeros(world_size)
            fn(out, torch.ones(world_size))
        elif collective_name in ("gather",):
            if rank == 0:
                out = [torch.zeros(3) for _ in range(world_size)]
                fn(out, t, dst=0)
            else:
                fn(None, t, dst=0)
        elif collective_name in ("scatter",):
            if rank == 0:
                in_tensors = [torch.ones(3) for _ in range(world_size)]
                fn(t, scatter_list=in_tensors, src=0)
            else:
                fn(t, scatter_list=None, src=0)
        elif collective_name == "broadcast_object_list":
            fn([t], src=0)
        elif collective_name == "all_gather_object":
            out = [None] * world_size
            fn(out, t)
        elif collective_name == "gather_object":
            if rank == 0:
                out = [None] * world_size
                fn(out, t, dst=0)
            else:
                fn(None, t, dst=0)
        elif collective_name == "scatter_object":
            if rank == 0:
                in_objs = [torch.ones(3) for _ in range(world_size)]
                fn(t, scatter_objects=in_objs, src=0)
            else:
                fn(t, scatter_objects=None, src=0)
        else:
            fn(t)


def _run_collective_detection_test(rank: int, world_size: int, collective_name: str):
    """在子进程中执行集合通信检测验证。"""
    # Test 1: guard does not interfere with normal execution
    with _collective_op_guard():
        t = torch.ones(3)
    assert t.sum().item() == 3.0, "Normal execution should work inside guard"

    # Test 2: guard detects the specified collective
    try:
        _verify_guard_detects_collective(rank, world_size, collective_name)
        raise AssertionError(
            f"Expected SchemaValidateError for dist.{collective_name}, but no exception was raised"
        )
    except SchemaValidateError:
        pass  # Expected
    except Exception as e:
        raise AssertionError(
            f"Expected SchemaValidateError for dist.{collective_name}, got {type(e).__name__}: {e}"
        ) from e

    # Test 3: after guard exits, collectives work again
    t = torch.ones(3)
    dist.broadcast(t, src=0)
    assert t.sum().item() == 3.0, "Collectives should work after guard exits"


class TestCollectiveOpGuardDistributed(unittest.TestCase):
    """_collective_op_guard 多 rank 检测测试。"""

    def test_broadcast_detected(self):
        """broadcast 操作应被检测并抛出 SchemaValidateError。"""
        run_distributed_spawn(2, _run_collective_detection_test,
                              fn_args=("broadcast",))

    def test_all_reduce_detected(self):
        """all_reduce 操作应被检测并抛出 SchemaValidateError。"""
        run_distributed_spawn(2, _run_collective_detection_test,
                              fn_args=("all_reduce",))

    def test_barrier_detected(self):
        """barrier 操作应被检测并抛出 SchemaValidateError。"""
        run_distributed_spawn(2, _run_collective_detection_test,
                              fn_args=("barrier",))

    def test_all_gather_detected(self):
        """all_gather 操作应被检测并抛出 SchemaValidateError。"""
        run_distributed_spawn(2, _run_collective_detection_test,
                              fn_args=("all_gather",))

    def test_broadcast_object_list_detected(self):
        """broadcast_object_list 操作应被检测并抛出 SchemaValidateError。"""
        run_distributed_spawn(2, _run_collective_detection_test,
                              fn_args=("broadcast_object_list",))

    def test_all_gather_object_detected(self):
        """all_gather_object 操作应被检测并抛出 SchemaValidateError。"""
        run_distributed_spawn(2, _run_collective_detection_test,
                              fn_args=("all_gather_object",))

    def test_reduce_detected(self):
        """reduce 操作应被检测并抛出 SchemaValidateError。"""
        run_distributed_spawn(2, _run_collective_detection_test,
                              fn_args=("reduce",))

    def test_gather_detected(self):
        """gather 操作应被检测并抛出 SchemaValidateError。"""
        run_distributed_spawn(2, _run_collective_detection_test,
                              fn_args=("gather",))

    def test_scatter_detected(self):
        """scatter 操作应被检测并抛出 SchemaValidateError。"""
        run_distributed_spawn(2, _run_collective_detection_test,
                              fn_args=("scatter",))

    def test_reduce_scatter_detected(self):
        """reduce_scatter 操作应被检测并抛出 SchemaValidateError。"""
        run_distributed_spawn(2, _run_collective_detection_test,
                              fn_args=("reduce_scatter",))


def _run_parallel_scheduler_legal_task(rank: int, world_size: int):
    """验证 parallel 调度器中合法任务正常执行。"""
    model = nn.Module()
    model.shared = nn.Linear(4, 4)

    def legal_fn():
        return torch.ones(3).sum().item()

    scheduler = DistributedTaskScheduler(model)
    with scheduler:
        scheduler.submit(legal_fn, dependencies=["shared"])
    records = scheduler.run()
    my_result = next((r.result for r in records if r.executor_rank == rank), None)
    if my_result is not None:
        assert my_result == 3.0, f"Legal task should produce correct result, got {my_result}"


class TestDTSCollectiveDetectionIntegration(unittest.TestCase):
    """DTS 调度器集合通信检测集成测试。"""

    def test_parallel_scheduler_legal_task(self):
        """parallel 调度器中合法任务正常执行。"""
        run_distributed_spawn(2, _run_parallel_scheduler_legal_task)

    def test_parallel_execute_local_task_has_guard(self):
        """验证 _DtsMultiRankParallelWaveScheduler 的 _execute_local_task 包含 guard。"""
        # Parallel scheduler should wrap with guard
        parallel_source = inspect.getsource(_DtsMultiRankParallelWaveScheduler._execute_local_task)
        self.assertIn("_collective_op_guard", parallel_source,
                      "_DtsMultiRankParallelWaveScheduler._execute_local_task must use _collective_op_guard")

        # Sequential scheduler should NOT have guard (inherits base class)
        seq_source = inspect.getsource(_DtsSequentialWaveScheduler._execute_local_task)
        self.assertNotIn("_collective_op_guard", seq_source,
                         "_DtsSequentialWaveScheduler._execute_local_task must NOT use _collective_op_guard")


if __name__ == "__main__":
    unittest.main()
