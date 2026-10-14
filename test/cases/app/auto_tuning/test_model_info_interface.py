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

from msmodelslim.app.auto_tuning.model_info_interface import ModelInfoInterface


class _ConcreteModelInfo(ModelInfoInterface):
    def __init__(self, pedigree: str, model_type: str, model_path_value: str = "/fake/path"):
        self._pedigree = pedigree
        self._type = model_type
        self._model_path_value = model_path_value
        self._trust_remote_code = False

    def get_model_pedigree(self) -> str:
        return self._pedigree

    def get_model_type(self) -> str:
        return self._type

    @property
    def model_type(self) -> str:
        return self._type

    @property
    def model_path(self) -> str:
        return self._model_path_value

    @property
    def trust_remote_code(self) -> bool:
        return self._trust_remote_code


class TestModelInfoInterface:
    """Test suite for ModelInfoInterface — ABC 接口契约。"""

    # ---------- 正常情形 ----------

    def test_get_model_pedigree_returns_subclass_value_when_subclass_implements(self):
        """主路径：get_model_pedigree 应返回子类返回值。"""
        info = _ConcreteModelInfo("qwen3", "Qwen3-32B")

        assert info.get_model_pedigree() == "qwen3"

    def test_get_model_type_returns_subclass_value_when_subclass_implements(self):
        """主路径：get_model_type 应返回子类返回值。"""
        info = _ConcreteModelInfo("qwen3", "Qwen3-32B")

        assert info.get_model_type() == "Qwen3-32B"

    # ---------- 边界情形 ----------

    def test_get_model_pedigree_returns_empty_string_when_subclass_returns_empty(self):
        """边界：子类返回空字符串应原样透传。"""
        info = _ConcreteModelInfo("", "")

        assert info.get_model_pedigree() == ""

    def test_get_model_type_returns_unicode_string_when_subclass_returns_unicode(self):
        """边界：unicode 字符串（如中文）应原样透传。"""
        info = _ConcreteModelInfo("通义千问", "Qwen3-32B")

        assert info.get_model_pedigree() == "通义千问"

    # ---------- 异常情形 ----------

    def test_model_info_interface_raises_type_error_when_instantiated_directly(self):
        """异常：直接实例化抽象类应抛 TypeError。"""
        with pytest.raises(TypeError):
            ModelInfoInterface()  # pylint: disable=abstract-class-instantiated

    def test_subclass_raises_type_error_when_get_model_pedigree_not_implemented(self):
        """异常：子类未实现 get_model_pedigree 应抛 TypeError。"""

        class _Incomplete(ModelInfoInterface):
            def get_model_type(self):
                return "x"

        with pytest.raises(TypeError):
            _Incomplete()  # pylint: disable=abstract-class-instantiated

    def test_subclass_raises_type_error_when_get_model_type_not_implemented(self):
        """异常：子类未实现 get_model_type 应抛 TypeError。"""

        class _Incomplete(ModelInfoInterface):
            def get_model_pedigree(self):
                return "x"

        with pytest.raises(TypeError):
            _Incomplete()  # pylint: disable=abstract-class-instantiated
