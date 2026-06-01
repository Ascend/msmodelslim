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

from abc import ABC
from typing import Any, Callable, Iterable, Optional
from collections.abc import Sequence
from collections.abc import MutableMapping
import torch
from msmodelslim.core.context.interface import IContext, INamespace, IValidatedState
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.core.context.interface import NAMESPACE_VALUE_WHITELIST
from msmodelslim.utils.distributed import is_rank_zero


def _is_torch_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor)


def _validate_namespace_value(
    value: Any,
    whitelist: Sequence[type] = NAMESPACE_VALUE_WHITELIST,
    path: str = "value",
    depth: int = 1,
    max_depth: int = 10,
) -> None:
    if depth > max_depth:
        raise SchemaValidateError(
            f"Unsupported nesting depth at {path}: {depth}. Maximum allowed depth is {max_depth}."
        )
    if _is_torch_tensor(value):
        if value.device.type != "cpu":
            raise SchemaValidateError(
                f"Unsupported torch.Tensor device at {path}: {value.device.type}. Only cpu tensors are allowed."
            )
        return

    if isinstance(value, list):
        for idx, item in enumerate(value):
            _validate_namespace_value(item, whitelist, f"{path}[{idx}]", depth + 1, max_depth)
        return

    if isinstance(value, tuple):
        for idx, item in enumerate(value):
            _validate_namespace_value(item, whitelist, f"{path}[{idx}]", depth + 1, max_depth)
        return

    if isinstance(value, dict):
        for key, item in value.items():
            _validate_namespace_value(key, whitelist, f"{path}.key", depth + 1, max_depth)
            _validate_namespace_value(item, whitelist, f"{path}[{repr(key)}]", depth + 1, max_depth)
        return

    primitive_types = tuple(t for t in whitelist if t not in (list, tuple, dict))
    if isinstance(value, primitive_types):
        return

    raise SchemaValidateError(
        f"Unsupported value type at {path}: {type(value)}. Only Python basic types and cpu torch.Tensor are allowed."
    )


class ValidatedDict(IValidatedState):
    def __init__(self, *args, **kwargs) -> None:
        self._dict = {}
        if args or kwargs:
            self.update(*args, **kwargs)
        self.whitelist = NAMESPACE_VALUE_WHITELIST

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value) -> None:
        _validate_namespace_value(value, whitelist=self.whitelist)
        self._dict[key] = value

    def __delitem__(self, key) -> None:
        del self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self) -> str:
        return repr(self._dict)


class DebugDict(ValidatedDict):
    """Debug dictionary that only records when debug is enabled and on rank 0."""

    def __init__(self, enable_debug: bool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._enable_debug = enable_debug

    def __setitem__(self, key, value) -> None:
        # Only record debug info when debug is enabled and on rank 0
        if self._enable_debug and is_rank_zero():
            super().__setitem__(key, value)


class BaseNamespace(INamespace):
    def __init__(
        self,
        enable_debug: bool = False,
        dict_factory: Optional[Callable[[], MutableMapping]] = None,
    ) -> None:
        if dict_factory is None:
            dict_factory = ValidatedDict
        self._state = dict_factory()
        self._debug = DebugDict(enable_debug)

    @property
    def state(self) -> MutableMapping:
        return self._state

    @property
    def debug(self) -> MutableMapping:
        return self._debug

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(state={self._state}, debug={self._debug})"


class BaseContext(IContext, ABC):
    """Base context implementation with common dict-based namespace operations."""

    def __init__(self, enable_debug: bool = False) -> None:
        self._namespaces = {}
        self._enable_debug = enable_debug

    def get(self, key: str, default: Any = None) -> Any:
        return self._namespaces.get(key, default)

    def delete(self, key: str) -> None:
        self._namespaces.pop(key, None)

    def keys(self) -> Iterable[str]:
        return self._namespaces.keys()

    def is_enable_debug(self) -> bool:
        """Check if debug recording is enabled.

        Returns:
            bool: True if debug recording is enabled, False otherwise.
        """
        return self._enable_debug

    def __getitem__(self, key: str) -> INamespace:
        if key not in self._namespaces:
            self._namespaces[key] = BaseNamespace(self._enable_debug)
        return self._namespaces[key]

    def __delitem__(self, key: str) -> None:
        del self._namespaces[key]

    def __contains__(self, key: str) -> bool:
        return key in self._namespaces

    def __iter__(self):
        return iter(self._namespaces)

    def __len__(self) -> int:
        return len(self._namespaces)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(namespaces={list(self._namespaces.keys())})"
