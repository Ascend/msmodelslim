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

from msmodelslim.core.context.context_factory import ContextFactory
from msmodelslim.core.context.interface import ContextManager
from msmodelslim.core.quant_service.key_info_persistence_infra import KeyInfoPersistenceInfra


class _StubPersistence(KeyInfoPersistenceInfra):
    def __init__(self):
        self.saved = False

    def save_from_context(self, ctx=None) -> None:
        self.saved = True


class TestKeyInfoPersistenceInfra:
    """Tests for KeyInfoPersistenceInfra."""

    def test_save_from_context_sets_flag_when_stub_called(self):
        """场景：子类实现 save_from_context。预期：标记已保存。"""
        persistence = _StubPersistence()
        persistence.save_from_context()
        assert persistence.saved is True

    def test_save_from_context_accepts_context_when_provided(self):
        """场景：传入显式 context。预期：不抛异常。"""
        persistence = _StubPersistence()
        ctx = ContextFactory().create(is_distributed=False)
        with ContextManager(ctx=ctx):
            persistence.save_from_context(ctx=ctx)
        assert persistence.saved is True

    def test_instantiate_raises_type_error_when_abstract(self):
        """场景：直接实例化抽象类。预期：TypeError。"""
        with pytest.raises(TypeError):
            KeyInfoPersistenceInfra()  # pylint: disable=abstract-class-instantiated
