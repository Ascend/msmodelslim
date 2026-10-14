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

from abc import ABC, abstractmethod
from pathlib import Path  # noqa: F401
from typing import Optional

from msmodelslim.core.context import IContext


class KeyInfoPersistenceInfra(ABC):
    """
    Abstract interface for context persistence.

    This interface defines the contract for saving context data during quantization.
    Implementations should handle the serialization and storage of context information.
    """

    @abstractmethod
    def save_from_context(self, ctx: Optional[IContext] = None) -> None:
        """
        Save context data to persistent storage.

        Args:
            ctx: Context to save. If None, uses current context from get_current_context().
            include_state: Whether to include state data (default: True)
            include_debug: Whether to include debug data (default: False)

        Raises:
            SchemaValidateError: If context is None or invalid
        """
