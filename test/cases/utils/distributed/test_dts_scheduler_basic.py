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
from unittest.mock import MagicMock

from torch import nn

from msmodelslim.utils.distributed.task_scheduler.scheduler import DistributedTaskScheduler


class TestDistributedTaskSchedulerGlobalDisableParallel(unittest.TestCase):
    """测试DistributedTaskScheduler的全局disable_parallel标志"""

    def tearDown(self):
        DistributedTaskScheduler.set_global_disable_parallel(False)

    def test_set_global_disable_parallel_when_true_then_get_returns_true(self):
        """正常：设置True后应返回True"""
        DistributedTaskScheduler.set_global_disable_parallel(True)
        self.assertTrue(DistributedTaskScheduler.get_global_disable_parallel())

    def test_set_global_disable_parallel_when_false_then_get_returns_false(self):
        """正常：设置False后应返回False"""
        DistributedTaskScheduler.set_global_disable_parallel(False)
        self.assertFalse(DistributedTaskScheduler.get_global_disable_parallel())

    def test_get_global_disable_parallel_default_when_never_set_then_returns_false(self):
        """边界：默认值应为False"""
        DistributedTaskScheduler._global_disable_parallel = False
        self.assertFalse(DistributedTaskScheduler.get_global_disable_parallel())


class TestDistributedTaskSchedulerContextManager(unittest.TestCase):
    """测试DistributedTaskScheduler的上下文管理器"""

    def test_context_manager_returns_self_when_enter(self):
        """正常：__enter__应返回自身"""
        model = nn.Linear(4, 8)
        mock_backend = MagicMock()
        scheduler = DistributedTaskScheduler(model, backend=mock_backend)

        with scheduler as s:
            self.assertIs(s, scheduler)

    def test_context_manager_closes_backend_when_exit(self):
        """正常：__exit__应关闭backend"""
        model = nn.Linear(4, 8)
        mock_backend = MagicMock()
        scheduler = DistributedTaskScheduler(model, backend=mock_backend)

        with scheduler:
            pass

        mock_backend.close.assert_called_once()

    def test_context_manager_sets_closed_flag_when_exit(self):
        """正常：__exit__后应设置_closed标志"""
        model = nn.Linear(4, 8)
        mock_backend = MagicMock()
        scheduler = DistributedTaskScheduler(model, backend=mock_backend)

        with scheduler:
            self.assertFalse(scheduler._closed)

        self.assertTrue(scheduler._closed)


class TestDistributedTaskSchedulerSubmitAfterClose(unittest.TestCase):
    """测试DistributedTaskScheduler关闭后提交的行为"""

    def test_submit_raises_runtime_error_when_scheduler_is_closed(self):
        """异常：关闭后提交应抛出RuntimeError"""
        model = nn.Linear(4, 8)
        mock_backend = MagicMock()
        scheduler = DistributedTaskScheduler(model, backend=mock_backend)

        with scheduler:
            pass

        with self.assertRaises(RuntimeError) as ctx:
            scheduler.submit(lambda: None)
        self.assertIn("closed", str(ctx.exception))


class TestDistributedTaskSchedulerInit(unittest.TestCase):
    """测试DistributedTaskScheduler的初始化"""

    def test_init_stores_model_when_created(self):
        """正常：应存储model"""
        model = nn.Linear(4, 8)
        mock_backend = MagicMock()
        scheduler = DistributedTaskScheduler(model, backend=mock_backend)
        self.assertIs(scheduler.model, model)

    def test_init_stores_disable_parallel_when_created(self):
        """正常：应存储disable_parallel"""
        model = nn.Linear(4, 8)
        mock_backend = MagicMock()
        scheduler = DistributedTaskScheduler(model, disable_parallel=True, backend=mock_backend)
        self.assertTrue(scheduler.disable_parallel)

    def test_init_uses_provided_backend_when_given(self):
        """正常：应使用提供的backend"""
        model = nn.Linear(4, 8)
        mock_backend = MagicMock()
        scheduler = DistributedTaskScheduler(model, backend=mock_backend)
        self.assertIs(scheduler._backend, mock_backend)

    def test_init_creates_wave_backend_when_none(self):
        """边界：未提供backend时应创建WaveDTSBackend"""
        from msmodelslim.utils.distributed.task_scheduler.backend.wave import WaveDTSBackend

        model = nn.Linear(4, 8)
        scheduler = DistributedTaskScheduler(model)
        self.assertIsInstance(scheduler._backend, WaveDTSBackend)


class TestDistributedTaskSchedulerSubmit(unittest.TestCase):
    """测试DistributedTaskScheduler的submit方法"""

    def test_submit_delegates_to_backend_when_not_closed(self):
        """正常：submit应委托给backend"""
        model = nn.Linear(4, 8)
        mock_backend = MagicMock()
        scheduler = DistributedTaskScheduler(model, disable_parallel=False, backend=mock_backend)

        def dummy_fn():
            pass

        scheduler.submit(dummy_fn, args=(1,), kwargs={"k": "v"}, dependencies=["dep"], parallel=True)

        mock_backend.submit.assert_called_once_with(
            dummy_fn,
            args=(1,),
            kwargs={"k": "v"},
            dependencies=["dep"],
            sync_fn=None,
            parallel=True,
            scheduler_disable_parallel=False,
            global_disable_parallel=False,
        )


class TestDistributedTaskSchedulerRun(unittest.TestCase):
    """测试DistributedTaskScheduler的run方法"""

    def test_run_delegates_to_backend_when_called(self):
        """正常：run应委托给backend"""
        model = nn.Linear(4, 8)
        mock_backend = MagicMock()
        mock_backend.run.return_value = []
        scheduler = DistributedTaskScheduler(model, backend=mock_backend)

        result = scheduler.run()

        mock_backend.run.assert_called_once()
        self.assertEqual(result, [])
