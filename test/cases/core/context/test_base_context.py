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
# pylint: disable=unsubscriptable-object

import pytest
from collections.abc import MutableMapping

from msmodelslim.core.context.context_factory import ContextFactory
from msmodelslim.core.context import (
    ContextManager,
    get_current_context,
    LocalDictContext,
)
from msmodelslim.core.context.interface import IContext
from msmodelslim.utils.exception import SchemaValidateError


class CustomInvalidType:
    """Custom class used to validate non-whitelist type rejection."""

    def __init__(self, name: str = "invalid") -> None:
        self.name = name


class TestContextFactory:
    """Tests for ContextFactory."""

    def test_create_returns_context_with_mutable_namespace_when_is_distributed_false(self):
        """create 在单进程场景下返回的 context 可创建 namespace，且 state/debug 为 MutableMapping 并可读写."""
        factory = ContextFactory(enable_debug=True)
        ctx = factory.create(is_distributed=False)
        with ContextManager(ctx=ctx):
            ctx_obj = get_current_context()
            assert isinstance(ctx_obj["Quarot"].state, MutableMapping)
            assert isinstance(ctx_obj["Quarot"].debug, MutableMapping)
            ctx_obj["Quarot"].state["foo"] = "bar"
            ctx_obj["Quarot"].debug["baz"] = 1
            assert ctx_obj["Quarot"].state["foo"] == "bar"
            assert ctx_obj["Quarot"].debug["baz"] == 1


class TestBaseNamespaceState:
    """Tests for BaseNamespace.state (state 字典的赋值校验)."""

    def test_setitem_raises_schema_validate_error_when_value_is_non_whitelist_custom_type(self):
        """state 赋值非白名单自定义类型时抛出 SchemaValidateError."""
        factory = ContextFactory()
        ctx = factory.create(is_distributed=False)
        with ContextManager(ctx=ctx):
            ns = ctx["test"]
            with pytest.raises(SchemaValidateError, match="Unsupported value type"):
                ns.state["invalid"] = CustomInvalidType("state")


class TestBaseNamespaceDebug:
    """Tests for BaseNamespace.debug (debug 字典的赋值校验)."""

    def test_setitem_raises_schema_validate_error_when_value_is_non_whitelist_custom_type(self):
        """debug 赋值非白名单自定义类型时抛出 SchemaValidateError."""
        factory = ContextFactory(enable_debug=True)
        ctx = factory.create(is_distributed=False)
        with ContextManager(ctx=ctx):
            ns = ctx["test"]
            with pytest.raises(SchemaValidateError, match="Unsupported value type"):
                ns.debug["invalid"] = CustomInvalidType("debug")


class TestContextManager:
    """Tests for ContextManager."""

    def test_enter_returns_context_when_ctx_given(self):
        """传入 ctx 时 enter 返回该 context."""
        ctx = ContextFactory().create(is_distributed=False)
        with ContextManager(ctx=ctx) as entered:
            assert entered is ctx

    def test_enter_raises_runtime_error_when_ctx_is_none(self):
        """ctx 为 None 时 enter 抛出 RuntimeError."""
        with pytest.raises(RuntimeError, match="Context is required"):
            with ContextManager(ctx=None):
                pass

    def test_exit_restores_previous_context_when_nested(self):
        """嵌套 with 时 exit 恢复为上一层 context."""
        ctx1 = ContextFactory().create(is_distributed=False)
        ctx2 = ContextFactory().create(is_distributed=False)
        with ContextManager(ctx=ctx1):
            assert get_current_context() is ctx1
            with ContextManager(ctx=ctx2):
                assert get_current_context() is ctx2
            assert get_current_context() is ctx1
        assert get_current_context() is None

    def test_get_current_returns_none_when_outside_context(self):
        """未进入任何 context 时 get_current_context 返回 None."""
        assert get_current_context() is None


class TestBaseContext:
    """Tests for BaseContext (通过 LocalDictContext 覆盖 get/delete/keys/__getitem__ 等)."""

    def test_get_returns_default_when_key_missing(self):
        """key 不存在时 get 返回 default."""
        ctx = ContextFactory().create(is_distributed=False)
        with ContextManager(ctx=ctx):
            assert ctx.get("missing", "default") == "default"

    def test_get_returns_namespace_when_key_exists(self):
        """key 存在时 get 返回对应 namespace."""
        ctx = ContextFactory().create(is_distributed=False)
        with ContextManager(ctx=ctx):
            var = ctx["a"]  # noqa: F841
            ns = ctx.get("a")
            assert ns is not None
            assert ns.state is not None

    def test_delete_removes_namespace_when_key_exists(self):
        """delete 删除已存在的 namespace."""
        ctx = ContextFactory().create(is_distributed=False)
        with ContextManager(ctx=ctx):
            var = ctx["a"]  # noqa: F841
            assert "a" in ctx
            ctx.delete("a")
            assert "a" not in ctx

    def test_keys_returns_all_namespace_keys_when_has_namespaces(self):
        """有 namespace 时 keys 返回所有 key."""
        ctx = ContextFactory().create(is_distributed=False)
        with ContextManager(ctx=ctx):
            var = ctx["x"]
            var = ctx["y"]  # noqa: F841
            assert set(ctx.keys()) == {"x", "y"}

    def test_getitem_creates_and_returns_namespace_when_key_missing(self):
        """key 不存在时 __getitem__ 会创建并返回 namespace."""
        ctx = ContextFactory().create(is_distributed=False)
        with ContextManager(ctx=ctx):
            ns = ctx["new_ns"]
            assert ns is not None
            assert "new_ns" in ctx

    def test_getitem_returns_existing_namespace_when_key_exists(self):
        """key 已存在时 __getitem__ 返回同一 namespace."""
        ctx = ContextFactory().create(is_distributed=False)
        with ContextManager(ctx=ctx):
            var = ctx["same"]  # noqa: F841
            assert ctx["same"] is ctx["same"]

    def test_delitem_removes_namespace_when_key_exists(self):
        """__delitem__ 删除已存在的 namespace."""
        ctx = ContextFactory().create(is_distributed=False)
        with ContextManager(ctx=ctx):
            var = ctx["d"]  # noqa: F841
            del ctx["d"]
            assert "d" not in ctx

    def test_contains_returns_true_when_key_exists(self):
        """key 存在时 __contains__ 为 True."""
        ctx = ContextFactory().create(is_distributed=False)
        with ContextManager(ctx=ctx):
            var = ctx["k"]  # noqa: F841
            assert "k" in ctx

    def test_contains_returns_false_when_key_missing(self):
        """key 不存在时 __contains__ 为 False."""
        ctx = ContextFactory().create(is_distributed=False)
        with ContextManager(ctx=ctx):
            assert "nonexist" not in ctx

    def test_len_returns_count_of_namespaces(self):
        """__len__ 返回当前 namespace 数量."""
        ctx = ContextFactory().create(is_distributed=False)
        with ContextManager(ctx=ctx):
            assert len(ctx) == 0
            var = ctx["a"]
            var = ctx["b"]  # noqa: F841
            assert len(ctx) == 2

    def test_iter_returns_keys_iterator(self):
        """__iter__ 可迭代出所有 namespace 的 key."""
        ctx = ContextFactory().create(is_distributed=False)
        with ContextManager(ctx=ctx):
            var = ctx["p"]
            var = ctx["q"]  # noqa: F841
            assert set(iter(ctx)) == {"p", "q"}


class TestContextFactoryType:
    """Tests for ContextFactory.create 返回类型."""

    def test_create_returns_local_dict_context_when_is_distributed_false(self):
        """is_distributed=False 时 create 返回 LocalDictContext."""
        factory = ContextFactory()
        ctx = factory.create(is_distributed=False)
        assert isinstance(ctx, LocalDictContext)
        assert isinstance(ctx, IContext)


class TestBaseNamespaceStateWhitelist:
    """Tests for BaseNamespace.state 接受白名单类型."""

    def test_setitem_accepts_value_when_value_is_whitelist_primitive(self):
        """state 接受白名单基本类型（int/str/float/list/dict）."""
        factory = ContextFactory()
        ctx = factory.create(is_distributed=False)
        with ContextManager(ctx=ctx):
            ns = ctx["test"]
            ns.state["int"] = 1
            ns.state["str"] = "ok"
            ns.state["float"] = 1.0
            ns.state["list"] = [1, "a"]
            ns.state["dict"] = {"k": "v"}
            assert ns.state["int"] == 1
            assert ns.state["str"] == "ok"
            assert ns.state["float"] == 1.0
            assert ns.state["list"] == [1, "a"]
            assert ns.state["dict"] == {"k": "v"}
