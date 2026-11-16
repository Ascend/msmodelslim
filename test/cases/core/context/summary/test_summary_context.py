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

from msmodelslim.core.context import ContextFactory, ContextManager
from msmodelslim.core.context.summary import QuantSummary, SUMMARY_NAMESPACE, SummaryContext


def test_summary_context_accepts_plain_dict_when_recording():
    state = {}
    context = SummaryContext(state)

    QuantSummary().time.record(context, "model.layers.0", 1.0)

    assert state["time"]["model.layers.0"] == {"elapsed": 1.0, "avg": 1.0}


def test_summary_context_exposes_injected_mapping_when_state_is_given():
    state = {}

    assert SummaryContext(state).state is state
    assert SummaryContext().state is None


def test_summary_context_binds_namespace_state_when_created_from_context():
    ctx = ContextFactory().create(is_distributed=False)

    # 命名空间按需惰性创建，初始为空，不需要 begin 初始化。
    assert len(SummaryContext.from_context(ctx).state) == 0

    with ContextManager(ctx):
        summary = QuantSummary()
        summary.time.record(SummaryContext.from_context(ctx), "model.layers.0", 2.0)
        summary.finish(ctx)

    state = ctx[SUMMARY_NAMESPACE].state
    assert list(state.keys()) == ["time"]
    assert state["time"]["model.layers.0"]["elapsed"] == 2.0


def test_summary_context_ignores_observations_when_state_is_missing():
    summary = QuantSummary()
    summary.time.record(SummaryContext(), "layer0", 1.0)

    assert not summary.finish({})
    assert SummaryContext().state is None


def test_summary_context_never_breaks_quantization_when_context_is_broken():
    class _BrokenContext:
        def get(self, key, default=None):
            raise RuntimeError("shared context unavailable")

        def __getitem__(self, key):
            raise RuntimeError("shared context unavailable")

    context = SummaryContext.from_context(_BrokenContext())
    assert context.enabled is False

    summary = QuantSummary()
    summary.time.record(context, "layer0", 1.0)

    assert not summary.finish(_BrokenContext())


def test_summary_context_is_disabled_when_no_current_context_exists():
    assert SummaryContext.current().enabled is False


def test_summary_context_lazily_binds_empty_state_when_created_from_current_context(monkeypatch):
    ctx = ContextFactory().create(is_distributed=False)
    monkeypatch.setattr("msmodelslim.core.context.summary.summary_context.get_current_context", lambda: ctx)

    context = SummaryContext.current()

    assert context.enabled is True
    assert context.state is ctx[SUMMARY_NAMESPACE].state
    assert len(context.state) == 0
