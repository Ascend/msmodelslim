#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

from msmodelslim.core.context.interface import IContextFactory, IContext


class ContextFactory(IContextFactory):
    def __init__(self, enable_debug: bool = False):
        """Initialize ContextFactory with debug setting.

        Args:
            enable_debug: Whether to enable debug recording in created contexts.
        """
        self.enable_debug = enable_debug

    def create(self, is_distributed: bool = False) -> IContext:
        """Create a context instance.

        Args:
            is_distributed: Whether to create a distributed context for multi-process scenarios.

        Returns:
            IContext: A context instance (LocalDictContext or SharedDictContext).
        """

        if is_distributed:
            from msmodelslim.core.context.shared_dict_context.context import SharedDictContext

            return SharedDictContext(enable_debug=self.enable_debug)
        from msmodelslim.core.context.local_dict_context.context import LocalDictContext

        return LocalDictContext(enable_debug=self.enable_debug)
