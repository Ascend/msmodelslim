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

from msmodelslim.utils.distributed.task_scheduler.constants import (
    DISTRIBUTED_TASK_QUEUE_GET_TIMEOUT_S,
    DTS_USER_LOG_PREFIX,
    DTS_PERF_LOG_RUN_TIME_SUMMARY_PREFIX,
    DTS_PERF_LOG_NOT_SUITABLE_FOR_PARALLEL_PREFIX,
    DTS_PERF_LOG_SPEEDUP_RATIO_PREFIX,
    DTS_PERF_LOG_SPEEDUP_SKIPPED_PREFIX,
    set_distributed_task_work_queue,
    get_distributed_task_work_queue,
    clear_distributed_task_work_queue,
)


class TestDtsConstantsValues(unittest.TestCase):
    """测试DTS常量值"""

    def test_timeout_when_accessed_then_is_positive_int(self):
        """DISTRIBUTED_TASK_QUEUE_GET_TIMEOUT_S-访问-应为正整数"""
        self.assertIsInstance(DISTRIBUTED_TASK_QUEUE_GET_TIMEOUT_S, int)
        self.assertGreater(DISTRIBUTED_TASK_QUEUE_GET_TIMEOUT_S, 0)

    def test_user_log_prefix_when_accessed_then_is_non_empty_string(self):
        """DTS_USER_LOG_PREFIX-访问-应为非空字符串"""
        self.assertIsInstance(DTS_USER_LOG_PREFIX, str)
        self.assertTrue(len(DTS_USER_LOG_PREFIX) > 0)

    def test_perf_log_prefixes_when_accessed_then_are_strings(self):
        """性能日志前缀-访问-应均为字符串"""
        for prefix in [
            DTS_PERF_LOG_RUN_TIME_SUMMARY_PREFIX,
            DTS_PERF_LOG_NOT_SUITABLE_FOR_PARALLEL_PREFIX,
            DTS_PERF_LOG_SPEEDUP_RATIO_PREFIX,
            DTS_PERF_LOG_SPEEDUP_SKIPPED_PREFIX,
        ]:
            self.assertIsInstance(prefix, str)
            self.assertTrue(len(prefix) > 0)


class TestDistributedTaskWorkQueue(unittest.TestCase):
    """测试分布式任务工作队列的set/get/clear"""

    def tearDown(self):
        clear_distributed_task_work_queue()

    def test_get_returns_none_when_never_set(self):
        """边界：从未设置时应返回None"""
        clear_distributed_task_work_queue()
        result = get_distributed_task_work_queue()
        self.assertIsNone(result)

    def test_get_returns_queue_when_set(self):
        """正常：设置后应返回对应队列"""
        mock_queue = {"type": "test_queue"}
        set_distributed_task_work_queue(mock_queue)
        result = get_distributed_task_work_queue()
        self.assertIs(result, mock_queue)

    def test_clear_sets_queue_to_none(self):
        """正常：清理后应返回None"""
        set_distributed_task_work_queue("some_queue")
        clear_distributed_task_work_queue()
        result = get_distributed_task_work_queue()
        self.assertIsNone(result)

    def test_set_overwrites_previous_queue(self):
        """边界：重复设置应覆盖前一个队列"""
        set_distributed_task_work_queue("queue_1")
        set_distributed_task_work_queue("queue_2")
        result = get_distributed_task_work_queue()
        self.assertEqual(result, "queue_2")

    def test_clear_is_idempotent_when_already_cleared(self):
        """边界：重复清理不应报错"""
        clear_distributed_task_work_queue()
        clear_distributed_task_work_queue()
        self.assertIsNone(get_distributed_task_work_queue())
