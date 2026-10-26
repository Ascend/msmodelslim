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

import unittest

from torch import nn

from msmodelslim.utils.distributed.task_scheduler.types import (
    _TaskSpec,
    Task,
    TaskExecutionRecord,
    TaskSyncContext,
)


class TestTaskSpecDefaults(unittest.TestCase):
    """测试_TaskSpec数据类的默认值"""

    def test_task_spec_defaults_when_created_with_minimal_args_then_use_defaults(self):
        """正常：仅提供task_id时应使用默认值"""
        spec = _TaskSpec(task_id="t0")
        self.assertEqual(spec.task_id, "t0")
        self.assertEqual(spec.dependencies, [])
        self.assertIsNone(spec.fn)
        self.assertEqual(spec.args, ())
        self.assertEqual(spec.kwargs, {})
        self.assertTrue(spec.parallel)
        self.assertEqual(spec.semantic_hash, "")

    def test_task_spec_dependencies_default_is_independent_when_multiple_instances(self):
        """边界：多个实例的默认dependencies应互不影响"""
        spec1 = _TaskSpec(task_id="t1")
        spec2 = _TaskSpec(task_id="t2")
        spec1.dependencies.append("dep1")
        self.assertNotIn("dep1", spec2.dependencies)


class TestTaskDefaults(unittest.TestCase):
    """测试Task数据类的默认值"""

    def test_task_sync_fn_is_none_when_not_provided(self):
        """正常：未提供sync_fn时应为None"""
        spec = _TaskSpec(task_id="t0")
        task = Task(spec=spec)
        self.assertIsNone(task.sync_fn)


class TestTaskExecutionRecordDefaults(unittest.TestCase):
    """测试TaskExecutionRecord数据类的默认值"""

    def test_record_defaults_when_created_with_minimal_args_then_use_defaults(self):
        """正常：仅提供task_id和executor_rank时应使用默认值"""
        record = TaskExecutionRecord(task_id="t0", executor_rank=0)
        self.assertEqual(record.task_id, "t0")
        self.assertEqual(record.executor_rank, 0)
        self.assertIsNone(record.result)
        self.assertEqual(record.dependencies, [])
        self.assertIsNone(record.exception)
        self.assertEqual(record.exec_time_s, 0.0)
        self.assertEqual(record.sync_time_s, 0.0)
        self.assertEqual(record.sync_meta, {})

    def test_record_dependencies_default_is_independent_when_multiple_instances(self):
        """边界：多个实例的默认dependencies应互不影响"""
        r1 = TaskExecutionRecord(task_id="t1", executor_rank=0)
        r2 = TaskExecutionRecord(task_id="t2", executor_rank=1)
        r1.dependencies.append("dep")
        self.assertNotIn("dep", r2.dependencies)


class TestTaskSyncContext(unittest.TestCase):
    """测试TaskSyncContext数据类"""

    def test_sync_context_when_created_then_stores_all_fields(self):
        """正常：应正确存储所有字段"""
        model = nn.Linear(4, 8)
        ctx = TaskSyncContext(model=model, rank=1, world_size=4)
        self.assertIs(ctx.model, model)
        self.assertEqual(ctx.rank, 1)
        self.assertEqual(ctx.world_size, 4)
