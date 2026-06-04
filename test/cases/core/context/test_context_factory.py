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

import sys

import pytest

from msmodelslim.core.context.context_factory import ContextFactory
from msmodelslim.core.context.local_dict_context.context import LocalDictContext
from msmodelslim.core.context.shared_dict_context.context import SharedDictContext


class TestContextFactory:
    """Tests for ContextFactory."""

    def test_create_returns_local_dict_context_when_not_distributed(self):
        """场景：单进程创建 context。预期：返回 LocalDictContext。"""
        factory = ContextFactory()
        ctx = factory.create(is_distributed=False)
        assert isinstance(ctx, LocalDictContext)

    @pytest.mark.skipif(sys.platform != "linux", reason="SharedDictContext requires Linux peercred")
    def test_create_returns_shared_dict_context_when_distributed(self):
        """场景：分布式创建 context。预期：返回 SharedDictContext。"""
        factory = ContextFactory(enable_debug=True)
        ctx = factory.create(is_distributed=True)
        assert isinstance(ctx, SharedDictContext)
        assert ctx.is_enable_debug() is True
