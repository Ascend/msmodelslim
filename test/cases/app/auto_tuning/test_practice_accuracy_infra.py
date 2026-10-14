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

from msmodelslim.app.auto_tuning.practice_accuracy_infra import (
    TuningAccuracyInfra,
    TuningAccuracyManagerInfra,
)


class _ConcreteAccuracy(TuningAccuracyInfra):
    def __init__(self):
        self.records = []
        self._count = 0

    def get_accuracy(self, practice, evaluation_config):
        for p, e, r in self.records:
            if p == practice and e == evaluation_config:
                return r
        return None

    def append_accuracy(self, practice, evaluation_config, evaluation):
        self.records.append((practice, evaluation_config, evaluation))
        self._count += 1

    def get_accuracy_count(self):
        return self._count


class _ConcreteAccuracyManager(TuningAccuracyManagerInfra):
    def __init__(self, accuracy):
        self._accuracy = accuracy
        self.loaded_from = None

    def load_accuracy(self, database: str):
        self.loaded_from = database
        return self._accuracy


class TestTuningAccuracyInfra:
    """Test suite for TuningAccuracyInfra — ABC 接口契约。"""

    # ---------- 正常情形 ----------

    def test_get_accuracy_returns_record_when_matching_practice_and_config(self):
        """主路径：匹配的 (practice, evaluation_config) 应返回对应 record。"""
        acc = _ConcreteAccuracy()
        acc.append_accuracy("p1", "cfg1", "result1")

        result = acc.get_accuracy("p1", "cfg1")

        assert result == "result1"

    def test_get_accuracy_returns_none_when_no_matching_record(self):
        """主路径：未匹配时返回 None。"""
        acc = _ConcreteAccuracy()

        result = acc.get_accuracy("p1", "cfg1")

        assert result is None

    def test_get_accuracy_count_returns_zero_when_no_records(self):
        """主路径：无记录时 count 应为 0。"""
        acc = _ConcreteAccuracy()

        assert acc.get_accuracy_count() == 0

    def test_get_accuracy_count_increments_after_each_append(self):
        """主路径：每次 append 后 count 应 +1。"""
        acc = _ConcreteAccuracy()
        acc.append_accuracy("p1", "cfg1", "r1")
        acc.append_accuracy("p2", "cfg2", "r2")

        assert acc.get_accuracy_count() == 2

    # ---------- 边界情形 ----------

    def test_get_accuracy_returns_first_match_when_multiple_records_share_practice(self):
        """边界：同一 practice 多 config 时，匹配 config 即返回（不依赖顺序）。"""
        acc = _ConcreteAccuracy()
        acc.append_accuracy("p1", "cfg1", "r1")
        acc.append_accuracy("p1", "cfg2", "r2")

        assert acc.get_accuracy("p1", "cfg2") == "r2"

    # ---------- 异常情形 ----------

    def test_tuning_accuracy_infra_raises_type_error_when_instantiated_directly(self):
        """异常：直接实例化抽象类应抛 TypeError。"""
        with pytest.raises(TypeError):
            TuningAccuracyInfra()  # pylint: disable=abstract-class-instantiated

    def test_subclass_raises_type_error_when_get_accuracy_count_not_implemented(self):
        """异常：子类未实现 get_accuracy_count 应抛 TypeError。"""

        class _Incomplete(TuningAccuracyInfra):
            def get_accuracy(self, practice, evaluation_config):
                return None

            def append_accuracy(self, practice, evaluation_config, evaluation):
                pass

        with pytest.raises(TypeError):
            _Incomplete()  # pylint: disable=abstract-class-instantiated


class TestTuningAccuracyManagerInfra:
    """Test suite for TuningAccuracyManagerInfra — ABC 接口契约。"""

    # ---------- 正常情形 ----------

    def test_load_accuracy_returns_accuracy_and_records_path_when_subclass_implements(self):
        """主路径：load_accuracy 应返回 accuracy 实例并记录调用路径。"""
        acc = _ConcreteAccuracy()
        manager = _ConcreteAccuracyManager(acc)

        result = manager.load_accuracy("/path/to/db")

        assert result is acc
        assert manager.loaded_from == "/path/to/db"

    # ---------- 异常情形 ----------

    def test_tuning_accuracy_manager_infra_raises_type_error_when_instantiated_directly(self):
        """异常：直接实例化抽象类应抛 TypeError。"""
        with pytest.raises(TypeError):
            TuningAccuracyManagerInfra()  # pylint: disable=abstract-class-instantiated
