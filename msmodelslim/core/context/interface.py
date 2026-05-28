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
from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional
from collections.abc import MutableMapping


NAMESPACE_VALUE_WHITELIST = (
    type(None),
    bool,
    int,
    float,
    str,
    bytes,
    list,
    tuple,
    dict,
)


class IValidatedState(MutableMapping):
    """Abstract interface for validated context state.
    
    validated context state is a context state that validates the values of the keys before setting them.
    Only python basic types and cpu torch.Tensor are allowed.
    Max depth is 10. If the depth is greater than 10, a SchemaValidateError will be raised.
    If the value is a torch.Tensor, it must be on cpu. If not, a SchemaValidateError will be raised.
    """

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def __setitem__(self, key, value) -> None:
        pass

    @abstractmethod
    def __delitem__(self, key) -> None:
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class INamespace(ABC):
    """Abstract interface for namespace.
    
    Namespace provides organized storage with two predefined sections:
    - state: For storing configuration and state data
    - debug: For storing debug and diagnostic information
    """

    @property
    @abstractmethod
    def state(self) -> IValidatedState:
        """Get the state dictionary.
        
        Returns:
            Dict-like object for storing state data.
        """

    @property
    @abstractmethod
    def debug(self) -> IValidatedState:
        """Get the debug dictionary.
        
        Returns:
            Dict-like object for storing debug/diagnostic data.
        """

    @abstractmethod
    def __repr__(self) -> str:
        """Return string representation."""


class IContext(ABC):
    """Abstract interface for context (manages multiple namespaces)."""

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get a namespace by key."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a namespace by key."""

    @abstractmethod
    def keys(self) -> Iterable[str]:
        """List all namespace keys."""

    @abstractmethod
    def is_enable_debug(self) -> bool:
        """Check if debug recording is enabled."""

    @abstractmethod
    def __getitem__(self, key: str) -> INamespace:
        """Get a namespace by key."""

    @abstractmethod
    def __delitem__(self, key: str) -> None:
        """Delete a namespace by key."""

    @abstractmethod
    def __contains__(self, key: str) -> bool:
        """Check if a namespace exists."""

    @abstractmethod
    def __iter__(self):
        """Iterate over namespace keys."""

    @abstractmethod
    def __len__(self) -> int:
        """Get the number of namespaces."""


class IContextFactory(ABC):
    """Abstract interface for context factory."""

    @abstractmethod
    def create(self, is_distributed: bool = False) -> IContext:
        """Create an IContext instance."""


class ContextManager:
    """Context lifecycle manager (context manager pattern)."""

    _current: Optional[IContext] = None

    def __init__(self, ctx: Optional[IContext] = None) -> None:
        self._context = ctx
        self._previous: Optional[IContext] = None

    def __enter__(self) -> IContext:
        """Enter the context and return the Context instance."""
        if self._context is None:
            raise RuntimeError("Context is required for ContextManager.")
        self._previous = ContextManager._current
        ContextManager._current = self._context
        return self._context

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context."""
        ContextManager._current = self._previous
        self._context = None
        self._previous = None

    @classmethod
    def get_current(cls) -> Optional[IContext]:
        """Get the current context."""
        return cls._current


def get_current_context() -> Optional[IContext]:
    """Get the current context."""
    return ContextManager.get_current()
