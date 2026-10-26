#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# pylint: disable=abstract-class-instantiated

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

import inspect

import pytest

from msmodelslim.utils.distributed.task_scheduler.backend.base import DTSBackend


class _UnfinishedBackend(DTSBackend):
    pass


class _RecordingBackend(DTSBackend):
    def __init__(self):
        self.submissions = []

    def submit(
        self,
        fn,
        args=(),
        kwargs=None,
        dependencies=None,
        sync_fn=None,
        parallel=True,
        *,
        scheduler_disable_parallel,
        global_disable_parallel,
    ):
        self.submissions.append(
            {
                "fn": fn,
                "args": args,
                "kwargs": kwargs,
                "dependencies": dependencies,
                "sync_fn": sync_fn,
                "parallel": parallel,
                "scheduler_disable_parallel": scheduler_disable_parallel,
                "global_disable_parallel": global_disable_parallel,
            }
        )

    def run(self):
        return [self.submissions]


def test_dts_backend_abstract_contract_requires_submit_and_run():
    assert inspect.isabstract(DTSBackend)
    assert DTSBackend.__abstractmethods__ == {"submit", "run"}

    with pytest.raises(TypeError, match="submit"):
        _UnfinishedBackend()


def test_concrete_backend_records_and_returns_submission():
    backend = _RecordingBackend()
    backend.submit(
        lambda value: value,
        args=(3,),
        scheduler_disable_parallel=True,
        global_disable_parallel=False,
    )

    result = backend.run()
    assert result[0][0]["args"] == (3,)
    assert result[0][0]["scheduler_disable_parallel"] is True
    assert result[0][0]["global_disable_parallel"] is False
