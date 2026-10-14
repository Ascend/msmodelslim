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
# pylint: disable=assignment-from-none,consider-using-from-import,no-name-in-module,pointless-string-statement

"""
DistributedTaskScheduler（分波调度器）的单元测试

测试范围：
    - 默认模块状态同步（单卡/多卡）

多 rank 同步验证：``run_distributed_spawn``（``torch.multiprocessing.spawn``）+子进程 ``gloo`` + ``file://``。
"""

import os  # noqa: E402
import unittest  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402, F401
from typing import Any, Dict, List, Tuple  # noqa: E402, F401

# 部分环境装了 torch_npu 但缺少运行时库时，torch 的 device backend auto-load 会导致 import torch 失败。
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from msmodelslim.utils.distributed import TaskExecutionRecord, TaskSyncContext  # noqa: E402

from test.cases.utils.distributed.dts_distributed_spawn import (  # noqa: E402
    run_distributed_spawn,
    start_queue_result_collector,
    spawn_queue,
)
from test.cases.utils.distributed.test_dts_scheduler_test_workers import _run_sync_worker_fn  # noqa: E402


def _sync_np_to_tensor(x):
    return torch.as_tensor(x)


class TestSyncFunctions(unittest.TestCase):
    """同步函数单元测试

    测试范围：
        - default_module_state_sync 单卡/多卡场景
        - Mock 分布式环境验证
    """

    def setUp(self):
        """测试前置准备"""
        self.mock_model = nn.Linear(10, 10)

    def test_default_module_state_sync_noops_when_single_rank(self):
        """测试：单卡场景不执行通信"""
        from test.cases.utils.distributed.dts_test_internals import default_module_state_sync

        record = TaskExecutionRecord(task_id="task1", executor_rank=0)
        sync_ctx = TaskSyncContext(
            model=self.mock_model,
            rank=0,
            world_size=1,
        )

        # 单卡场景不应抛出异常，也不应执行通信
        result = default_module_state_sync(record, sync_ctx, self.mock_model)

        # 无返回值，但不应抛出异常
        self.assertIsNone(result)

    def _test_module_sync_with_real_dist(self, module_cls, module_kwargs, world_size=2, owner_rank=0):
        """使用真实分布式环境测试模块同步

        Args:
            module_cls: 要测试的模块类
            module_kwargs: 模块构造参数
            world_size: 进程数
            owner_rank: owner rank

        Returns:
            list: 各 rank 的同步结果
        """
        results_queue = spawn_queue()
        results_future = start_queue_result_collector(results_queue, expected_results=world_size)
        run_distributed_spawn(
            world_size,
            _run_sync_worker_fn,
            (module_cls, module_kwargs, owner_rank, results_queue),
            init_dir_prefix="dts_gloo_sync_",
        )
        results = results_future.result(timeout_s=120.0)

        for r in results:
            if not r.get("ok", True):
                self.fail(f"DTS sync worker rank {r.get('rank')} failed:\n{r.get('error', r)}")

        return results

    def test_default_module_state_sync_syncs_linear_when_multirank_real_dist(self):
        """测试：使用真实分布式环境验证 nn.Linear 参数同步正确性

        场景：
            - 2 个 rank，rank 0 为 owner
            - rank 0 的 weight/bias = 1.0
            - rank 1 的 weight/bias = 0.0
        预期：
            - 同步后 rank 1 的 weight/bias 应该变为 1.0
        """
        results = self._test_module_sync_with_real_dist(
            nn.Linear,
            {"in_features": 10, "out_features": 20},
            world_size=2,
            owner_rank=0,
        )

        # 按 rank 组织结果
        results_by_rank = {r["rank"]: r for r in results}

        # 验证 rank 0 (owner) 的值保持不变
        rank0_result = results_by_rank[0]
        for key, value in rank0_result["post_sync"].items():
            t = _sync_np_to_tensor(value)
            self.assertTrue(
                torch.allclose(t, torch.tensor(1.0)),
                f"Rank 0 {key} should remain 1.0 after sync, got {t.mean().item()}",
            )

        # 验证 rank 1 的值被同步为 owner 的值 (1.0)
        rank1_result = results_by_rank[1]
        for key, value in rank1_result["post_sync"].items():
            t = _sync_np_to_tensor(value)
            self.assertTrue(
                torch.allclose(t, torch.tensor(1.0)),
                f"Rank 1 {key} should be synced to 1.0, got {t.mean().item()}",
            )

    def test_default_module_state_sync_syncs_embedding_when_multirank_real_dist(self):
        """测试：使用真实分布式环境验证 nn.Embedding 参数同步正确性

        场景：
            - 2 个 rank，rank 0 为 owner
            - rank 0 的 weight = 1.0
            - rank 1 的 weight = 0.0
        预期：
            - 同步后 rank 1 的 weight 应该变为 1.0
        """
        results = self._test_module_sync_with_real_dist(
            nn.Embedding,
            {"num_embeddings": 100, "embedding_dim": 64},
            world_size=2,
            owner_rank=0,
        )

        results_by_rank = {r["rank"]: r for r in results}

        # 验证 rank 0 (owner) 的值保持不变
        rank0_result = results_by_rank[0]
        for key, value in rank0_result["post_sync"].items():
            t = _sync_np_to_tensor(value)
            self.assertTrue(
                torch.allclose(t, torch.tensor(1.0)),
                f"Rank 0 {key} should remain 1.0 after sync, got {t.mean().item()}",
            )

        # 验证 rank 1 的值被同步
        rank1_result = results_by_rank[1]
        for key, value in rank1_result["post_sync"].items():
            t = _sync_np_to_tensor(value)
            self.assertTrue(
                torch.allclose(t, torch.tensor(1.0)),
                f"Rank 1 {key} should be synced to 1.0, got {t.mean().item()}",
            )

    def test_default_module_state_sync_syncs_layernorm_when_multirank_real_dist(self):
        """测试：使用真实分布式环境验证 nn.LayerNorm 参数同步正确性

        场景：
            - 2 个 rank，rank 0 为 owner
            - rank 0 的 weight/bias = 1.0/2.0
            - rank 1 的 weight/bias = 0.0
        预期：
            - 同步后 rank 1 的 weight/bias 应该与 rank 0 相同
        """
        results = self._test_module_sync_with_real_dist(
            nn.LayerNorm,
            {"normalized_shape": 128},
            world_size=2,
            owner_rank=0,
        )

        results_by_rank = {r["rank"]: r for r in results}
        rank0_result = results_by_rank[0]
        rank1_result = results_by_rank[1]

        # 验证所有参数都被同步
        for key in rank0_result["post_sync"].keys():
            owner_value = _sync_np_to_tensor(rank0_result["post_sync"][key])
            synced_value = _sync_np_to_tensor(rank1_result["post_sync"][key])
            self.assertTrue(
                torch.allclose(owner_value, synced_value),
                f"{key} not synced correctly: rank0={owner_value.mean().item()}, rank1={synced_value.mean().item()}",
            )

    def test_default_module_state_sync_syncs_conv2d_when_multirank_real_dist(self):
        """测试：使用真实分布式环境验证 nn.Conv2d 参数同步正确性

        场景：
            - 2 个 rank，rank 0 为 owner
            - rank 0 的 weight/bias 有特定值
            - rank 1 的 weight/bias = 0.0
        预期：
            - 同步后 rank 1 的值应该与 rank 0 相同
        """
        results = self._test_module_sync_with_real_dist(
            nn.Conv2d,
            {"in_channels": 3, "out_channels": 64, "kernel_size": 3},
            world_size=2,
            owner_rank=0,
        )

        results_by_rank = {r["rank"]: r for r in results}
        rank0_result = results_by_rank[0]
        rank1_result = results_by_rank[1]

        # 验证所有参数都被同步
        for key in rank0_result["post_sync"].keys():
            owner_value = _sync_np_to_tensor(rank0_result["post_sync"][key])
            synced_value = _sync_np_to_tensor(rank1_result["post_sync"][key])
            self.assertTrue(
                torch.allclose(owner_value, synced_value),
                f"{key} not synced correctly: rank0={owner_value.mean().item()}, rank1={synced_value.mean().item()}",
            )

    def test_default_module_state_sync_syncs_from_owner_rank1_when_multirank_real_dist(self):
        """测试：owner rank 为 1 时的同步正确性

        场景：
            - 2 个 rank，rank 1 为 owner
            - rank 1 的值为 1.0
            - rank 0 的值为 0.0
        预期：
            - 同步后 rank 0 的值应该与 rank 1 相同 (1.0)
        """
        results = self._test_module_sync_with_real_dist(
            nn.Linear,
            {"in_features": 10, "out_features": 10},
            world_size=2,
            owner_rank=1,
        )

        results_by_rank = {r["rank"]: r for r in results}

        # 验证 rank 0 的值被同步为 rank 1 (owner) 的值
        rank0_result = results_by_rank[0]
        for key, value in rank0_result["post_sync"].items():
            t = _sync_np_to_tensor(value)
            self.assertTrue(
                torch.allclose(t, torch.tensor(1.0)),
                f"Rank 0 {key} should be synced to owner (rank 1) value 1.0, got {t.mean().item()}",
            )
