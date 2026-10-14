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

from msmodelslim.app.auto_tuning.practice_manager_infra import PracticeManagerInfra


class _ConcretePracticeManager(PracticeManagerInfra):
    """Concrete impl that records invocations for assertion."""

    def __init__(self):
        self.save_called_with = None
        self.is_saving_value = True

    def save_practice(self, model_pedigree, practice):
        self.save_called_with = (model_pedigree, practice)

    def is_saving_supported(self):
        return self.is_saving_value


class TestPracticeManagerInfra:
    """Test suite for PracticeManagerInfra — ABC 接口契约。"""

    # ---------- 正常情形 ----------

    def test_save_practice_invokes_subclass_impl_when_pedigree_and_practice_provided(self):
        """主路径：实现类应能正常接收并处理 (pedigree, practice) 元组。"""
        manager = _ConcretePracticeManager()
        manager.save_practice("qwen3", "fake_practice")

        assert manager.save_called_with == ("qwen3", "fake_practice")

    def test_is_saving_supported_returns_subclass_value_when_subclass_overrides(self):
        """主路径：子类应能自定义返回值。"""
        manager = _ConcretePracticeManager()
        manager.is_saving_value = False

        assert manager.is_saving_supported() is False

    # ---------- 边界情形 ----------

    def test_is_saving_supported_returns_true_when_subclass_returns_true(self):
        """边界：默认实现返回 True（初始值）。"""
        manager = _ConcretePracticeManager()

        assert manager.is_saving_supported() is True

    def test_save_practice_accepts_empty_pedigree_string_when_caller_passes_empty(self):
        """边界：允许空字符串 pedigree 透传给子类。"""
        manager = _ConcretePracticeManager()
        manager.save_practice("", None)

        assert manager.save_called_with == ("", None)

    # ---------- 异常情形 ----------

    def test_practice_manager_infra_raises_type_error_when_instantiated_directly(self):
        """异常：直接实例化抽象类应抛 TypeError。"""
        with pytest.raises(TypeError):
            PracticeManagerInfra()  # pylint: disable=abstract-class-instantiated

    def test_subclass_raises_type_error_when_save_practice_not_implemented(self):
        """异常：子类未实现 save_practice 应抛 TypeError。"""

        class _Incomplete(PracticeManagerInfra):
            def is_saving_supported(self):
                return True

        with pytest.raises(TypeError):
            _Incomplete()  # pylint: disable=abstract-class-instantiated

    def test_subclass_raises_type_error_when_is_saving_supported_not_implemented(self):
        """异常：子类未实现 is_saving_supported 应抛 TypeError。"""

        class _Incomplete(PracticeManagerInfra):
            def save_practice(self, model_pedigree, practice):
                pass

        with pytest.raises(TypeError):
            _Incomplete()  # pylint: disable=abstract-class-instantiated
