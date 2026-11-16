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

import pytest

from msmodelslim.core.context.summary import SummaryContext, TimeQuantStatisticsObserver, TimeStats


class _RecordingLogger:
    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        self.messages.append(message % args if args else message)

    def warning(self, message, *args):
        self.messages.append(message % args if args else message)


def _make_observer():
    return TimeQuantStatisticsObserver()


def test_time_observer_records_elapsed_and_running_average_when_layers_are_recorded(summary_state):
    context = SummaryContext(summary_state)
    observer = _make_observer()

    observer.record(context, "model.layers.0", 1.0)
    observer.record(context, "model.layers.1", 3.0)

    time_state = summary_state["time"]
    assert time_state["model.layers.0"] == {"elapsed": 1.0, "avg": 1.0}
    assert time_state["model.layers.1"] == {"elapsed": 3.0, "avg": 2.0}


def test_time_observer_does_not_write_when_rank_is_not_zero(summary_state, monkeypatch):
    monkeypatch.setattr("msmodelslim.core.context.summary.observers.time_observer.is_rank_zero", lambda: False)
    context = SummaryContext(summary_state)
    observer = _make_observer()

    observer.record(context, "model.layers.0", 1.0)

    assert summary_state == {}


def test_time_observer_prints_layer_line_when_layer_is_recorded(summary_state, monkeypatch):
    recorder = _RecordingLogger()
    monkeypatch.setattr("msmodelslim.core.context.summary.observers.time_observer.logger", recorder)
    context = SummaryContext(summary_state)
    observer = _make_observer()

    observer.record(context, "model.layers.0", 1.0)
    observer.record(context, "model.layers.1", 3.0)

    assert recorder.messages == [
        "model.layers.0 quant done, this layer elapsed 1.00s, overall avg 1.00s",
        "model.layers.1 quant done, this layer elapsed 3.00s, overall avg 2.00s",
    ]


def test_time_observer_aggregate_sums_layers_and_elapsed_when_data_exists(summary_state):
    context = SummaryContext(summary_state)
    observer = _make_observer()

    observer.record(context, "model.layers.0", 1.0)
    observer.record(context, "model.layers.1", 3.0)

    result = observer.aggregate(context)

    assert result == TimeStats(layer_count=2, total_elapsed=4.0)
    assert result.avg_elapsed == pytest.approx(2.0)


def test_time_observer_aggregate_returns_none_when_no_data(summary_state):
    assert _make_observer().aggregate(SummaryContext(summary_state)) is None
    assert TimeQuantStatisticsObserver().aggregate(SummaryContext(None)) is None


def test_time_observer_format_reports_layers_total_and_average_when_stats_exist():
    observer = TimeQuantStatisticsObserver()

    assert observer.format(TimeStats(layer_count=2, total_elapsed=4.0)) == (
        "[Time] 2 layers total 4.00s, avg 2.00s/layer"
    )


def test_time_observer_record_is_a_no_op_when_container_is_missing():
    context = SummaryContext(None)
    observer = TimeQuantStatisticsObserver()

    observer.record(context, "model.layers.0", 1.0)
