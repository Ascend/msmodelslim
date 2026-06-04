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
from msmodelslim.core.context.interface import ContextManager, IContextFactory, get_current_context


class TestIContextFactory:
    """Tests for IContextFactory contract."""

    def test_context_factory_satisfies_interface_when_instantiated(self):
        """场景：实例化 ContextFactory。预期：实现 IContextFactory。"""
        factory = ContextFactory()
        assert isinstance(factory, IContextFactory)


class TestContextManager:
    """Tests for ContextManager."""

    def test_enter_sets_current_context_when_ctx_provided(self):
        """场景：with ContextManager 进入。预期：get_current_context 返回同一实例。"""
        factory = ContextFactory()
        ctx = factory.create(is_distributed=False)
        with ContextManager(ctx=ctx) as active:
            assert active is ctx
            assert get_current_context() is ctx

    def test_exit_restores_previous_context_when_nested(self):
        """场景：嵌套 ContextManager。预期：退出后恢复外层 context。"""
        outer = ContextFactory().create(is_distributed=False)
        inner = ContextFactory().create(is_distributed=False)
        with ContextManager(ctx=outer):
            with ContextManager(ctx=inner):
                assert get_current_context() is inner
            assert get_current_context() is outer

    def test_enter_raises_runtime_error_when_ctx_is_none(self):
        """场景：未提供 ctx。预期：RuntimeError。"""
        with pytest.raises(RuntimeError, match="Context is required"):
            with ContextManager(ctx=None):
                pass
