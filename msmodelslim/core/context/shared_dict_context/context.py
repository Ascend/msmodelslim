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

from typing import Any, Dict
from msmodelslim.core.context.base import BaseContext, BaseNamespace
from .peercred_manager import PeercredManager


class SharedNamespace(BaseNamespace):
    """Shared namespace for multi-process scenarios."""

    def __init__(self, enable_debug: bool = False, manager: PeercredManager = None) -> None:
        super().__init__(enable_debug, dict_factory=manager.validated_dict)

    def __repr__(self) -> str:
        return f"SharedNamespace(state={self._state}, debug={self._debug})"


class SharedDictContext(BaseContext):
    def __init__(self, enable_debug: bool = False) -> None:
        super().__init__(enable_debug)
        self._manager = PeercredManager()
        self._manager.start()
        self._address = self._manager.address
        self._namespaces = self._manager.dict()

    def _ensure_manager(self) -> PeercredManager:
        """Lazy reconnect to manager when unpickled."""
        if self._manager is None:
            self._manager = PeercredManager(address=self._address)
        return self._manager

    def get_namespaces(self) -> Dict[str, SharedNamespace]:
        return self._namespaces

    def __getitem__(self, key: str) -> SharedNamespace:
        if key not in self._namespaces:
            manager = self._ensure_manager()
            self._namespaces[key] = SharedNamespace(self._enable_debug, manager)
        return self._namespaces[key]

    def __repr__(self) -> str:
        return f"SharedContext(namespaces={list(self._namespaces.keys())})"

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state['_manager'] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._manager = None
        # Ensure _enable_debug exists after unpickling
        if "_enable_debug" not in self.__dict__:
            self._enable_debug = False
