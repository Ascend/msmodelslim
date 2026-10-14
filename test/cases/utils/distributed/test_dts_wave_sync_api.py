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
    - DTS 同步 API 与默认/自定义同步路径
"""

import unittest  # noqa: E402
from typing import Any, Dict, List, Tuple  # noqa: E402, F401
from unittest.mock import patch  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402, F401

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from msmodelslim.utils.distributed import (  # noqa: E402
    DTSMixin,
    DistributedTaskScheduler,
    TaskExecutionRecord,
    TaskSyncContext,
)
from test.cases.utils.distributed.dts_test_internals import dts_waves  # noqa: E402


class TestDistributedTaskSchedulerSync(unittest.TestCase):
    """DistributedTaskScheduler 同步逻辑单元测试

    测试范围：
        - 自定义 sync_fn 调用
        - DTSMixin.distributed_sync 调用
        - 回退到默认同步函数
        - sync_meta 记录
    """

    def setUp(self):
        """测试前置准备"""
        self.mock_model = nn.Linear(10, 10)

    def test_sync_task_does_not_call_custom_sync_fn_on_single_rank_sequential(self):
        """测试：单卡串行语义下不执行自定义 sync_fn（仅并行可执行条件下才同步）"""
        custom_sync_called = False
        received_record = None
        received_ctx = None

        def custom_sync_fn(record, sync_ctx):
            nonlocal custom_sync_called, received_record, received_ctx
            custom_sync_called = True
            received_record = record
            received_ctx = sync_ctx

        def fn():
            return "ok"

        scheduler = DistributedTaskScheduler(self.mock_model)

        with scheduler:
            scheduler.submit(fn, dependencies=[], sync_fn=custom_sync_fn)
            records = scheduler.run()  # noqa: F841

        self.assertFalse(custom_sync_called)
        self.assertIsNone(received_record)
        self.assertIsNone(received_ctx)

    def test_sync_task_does_not_call_module_distributed_sync_on_single_rank_sequential(self):
        """测试：单卡串行语义下不执行 DTSMixin.distributed_sync"""
        sync_called = False

        class SyncableModule(nn.Linear, DTSMixin):
            def distributed_sync(self, record, sync_ctx):
                nonlocal sync_called
                sync_called = True

        class Root(nn.Module):
            def __init__(self):
                super().__init__()
                self.m1 = SyncableModule(10, 10)

        syncable_model = Root()

        def fn():
            return "ok"

        scheduler = DistributedTaskScheduler(syncable_model)

        with scheduler:
            # 注意：dependencies 使用模型路径
            scheduler.submit(fn, dependencies=["m1"])
            records = scheduler.run()

        self.assertFalse(sync_called)
        self.assertEqual(len(records), 1)

    def test_sync_task_falls_back_to_default_sync_when_no_task_or_instance_sync(self):
        """测试：无自定义 sync_fn / DTSMixin 时回退到默认同步函数（不应抛错）"""

        def fn():
            return "ok"

        scheduler = DistributedTaskScheduler(self.mock_model)

        with scheduler:
            scheduler.submit(fn, dependencies=[])
            records = scheduler.run()

        self.assertEqual(len(records), 1)

    def test_sync_all_task_returns_records_with_sync_meta_when_sync_runs(self):
        """测试：同步记录包含 sync_meta"""

        def fn():
            return "ok"

        scheduler = DistributedTaskScheduler(self.mock_model)

        with scheduler:
            scheduler.submit(fn, dependencies=[])
            records = scheduler.run()

        self.assertEqual(len(records), 1)
        self.assertIsInstance(records[0].sync_meta, dict)

    def test_sync_task_parent_dependency_syncs_parent_and_children(self):
        """测试：dependencies 指向父模块时，按 modules() 语义同步父模块及子模块"""
        dts_sync_calls = 0

        class SyncLeaf(nn.Module, DTSMixin):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(1))

            def distributed_sync(self, record, sync_ctx):
                _ = (record, sync_ctx)
                nonlocal dts_sync_calls
                dts_sync_calls += 1

        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.sync_leaf = SyncLeaf()
                self.plain_leaf = nn.Linear(2, 2)

        class Root(nn.Module):
            def __init__(self):
                super().__init__()
                self.parent = Parent()

        model = Root()
        scheduler = DistributedTaskScheduler(model)
        with scheduler:
            scheduler.submit(lambda: "ok", dependencies=["parent"])
            wave = dts_waves(scheduler)[0]
            task = wave._tasks[0]
            record = TaskExecutionRecord(task_id=task.spec.task_id, executor_rank=0)
            sync_ctx = TaskSyncContext(model=model, rank=0, world_size=2)

            synced_modules = []

            def _fake_default_sync(rec, ctx, module):
                _ = (rec, ctx)
                synced_modules.append(module)

            with patch(
                "msmodelslim.utils.distributed.task_scheduler.backend.wave.default_module_state_sync",
                side_effect=_fake_default_sync,
            ):
                wave._sync_task(task, record, sync_ctx)

        self.assertEqual(dts_sync_calls, 1)
        self.assertEqual(len(synced_modules), 2)
        self.assertIs(synced_modules[0], model.parent)
        self.assertIs(synced_modules[1], model.parent.plain_leaf)

    def test_sync_task_leaf_dependency_default_syncs_once(self):
        """测试：dependencies 直接指向叶子模块时，默认同步仍执行且仅一次"""

        class Root(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3, 3)

        model = Root()
        leaf = model.fc
        scheduler = DistributedTaskScheduler(model)
        with scheduler:
            scheduler.submit(lambda: "ok", dependencies=["fc"])
            wave = dts_waves(scheduler)[0]
            task = wave._tasks[0]
            record = TaskExecutionRecord(task_id=task.spec.task_id, executor_rank=0)
            sync_ctx = TaskSyncContext(model=model, rank=0, world_size=2)

            synced_modules = []

            def _fake_default_sync(rec, ctx, module):
                _ = (rec, ctx)
                synced_modules.append(module)

            with patch(
                "msmodelslim.utils.distributed.task_scheduler.backend.wave.default_module_state_sync",
                side_effect=_fake_default_sync,
            ):
                wave._sync_task(task, record, sync_ctx)

        self.assertEqual(len(synced_modules), 1)
        self.assertIs(synced_modules[0], leaf)
