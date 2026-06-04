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

from collections.abc import MutableMapping

from msmodelslim.core.context.local_dict_context.context import LocalDictContext, Namespace


class TestNamespace:
    """Tests for Namespace."""

    def test_state_is_mutable_mapping_when_created_via_context(self):
        """场景：通过 LocalDictContext 创建 Namespace。预期：state 可读写。"""
        ctx = LocalDictContext(enable_debug=True)
        ns = ctx["test"]
        assert isinstance(ns, Namespace)
        ns.state["alpha"] = 1.0
        assert ns.state["alpha"] == 1.0


class TestLocalDictContext:
    """Tests for LocalDictContext lazy namespace creation."""

    def test_getitem_creates_namespace_lazily_when_key_missing(self):
        """场景：首次访问不存在的 key。预期：自动创建 Namespace 并缓存。"""
        ctx = LocalDictContext()
        assert len(ctx) == 0
        ns = ctx["Quarot"]
        assert isinstance(ns, Namespace)
        assert "Quarot" in ctx
        assert ctx["Quarot"] is ns

    def test_namespace_state_is_mutable_mapping_when_accessed(self):
        """场景：访问 namespace.state。预期：为 MutableMapping 且可赋值。"""
        ctx = LocalDictContext(enable_debug=True)
        ns = ctx["test"]
        assert isinstance(ns.state, MutableMapping)
        ns.state["alpha"] = 1.0
        assert ns.state["alpha"] == 1.0
