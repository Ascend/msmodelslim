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

from msmodelslim.app.auto_tuning.practice_history_infra import (
    TuningHistoryInfra,
    TuningHistoryManagerInfra,
)


class _ConcreteHistory(TuningHistoryInfra):
    def __init__(self):
        self.records = []
        self.cleared = False

    def append_history(self, practice, evaluation):
        self.records.append((practice, evaluation))

    def clear_records(self):
        self.records = []
        self.cleared = True


class _ConcreteHistoryManager(TuningHistoryManagerInfra):
    def __init__(self, history):
        self._history = history
        self.loaded_from = None

    def load_history(self, database: str):
        self.loaded_from = database
        return self._history


class TestTuningHistoryInfra:
    """Test suite for TuningHistoryInfra — ABC 接口契约。"""

    # ---------- 正常情形 ----------

    def test_append_history_records_tuple_when_called_with_practice_and_evaluation(self):
        """主路径：append_history 应记录 (practice, evaluation) 元组。"""
        history = _ConcreteHistory()
        history.append_history("p1", "e1")
        history.append_history("p2", "e2")

        assert history.records == [("p1", "e1"), ("p2", "e2")]

    def test_clear_records_empties_records_and_sets_cleared_flag_when_called(self):
        """主路径：clear_records 应清空 records 并标记 cleared。"""
        history = _ConcreteHistory()
        history.append_history("p1", "e1")
        history.clear_records()

        assert not history.records
        assert history.cleared is True

    # ---------- 边界情形 ----------

    def test_clear_records_on_empty_history_is_noop_when_no_records(self):
        """边界：空 history 调用 clear_records 不应报错。"""
        history = _ConcreteHistory()
        history.clear_records()

        assert not history.records
        assert history.cleared is True

    # ---------- 异常情形 ----------

    def test_tuning_history_infra_raises_type_error_when_instantiated_directly(self):
        """异常：直接实例化抽象类应抛 TypeError。"""
        with pytest.raises(TypeError):
            TuningHistoryInfra()  # pylint: disable=abstract-class-instantiated

    def test_subclass_raises_type_error_when_append_history_not_implemented(self):
        """异常：子类未实现 append_history 应抛 TypeError。"""

        class _Incomplete(TuningHistoryInfra):
            def clear_records(self):
                pass

        with pytest.raises(TypeError):
            _Incomplete()  # pylint: disable=abstract-class-instantiated


class TestTuningHistoryManagerInfra:
    """Test suite for TuningHistoryManagerInfra — ABC 接口契约。"""

    # ---------- 正常情形 ----------

    def test_load_history_returns_history_and_records_path_when_subclass_implements(self):
        """主路径：load_history 应返回 history 实例并记录调用路径。"""
        history = _ConcreteHistory()
        manager = _ConcreteHistoryManager(history)

        result = manager.load_history("/path/to/db")

        assert result is history
        assert manager.loaded_from == "/path/to/db"

    # ---------- 异常情形 ----------

    def test_tuning_history_manager_infra_raises_type_error_when_instantiated_directly(self):
        """异常：直接实例化抽象类应抛 TypeError。"""
        with pytest.raises(TypeError):
            TuningHistoryManagerInfra()  # pylint: disable=abstract-class-instantiated
