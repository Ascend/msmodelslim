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

# TLQ Kernel 注册表：按 ``(q_dtype, storage_dtype, scope, symmetric)`` 分发。

from __future__ import annotations

from typing import Callable, Dict, Literal, Tuple, TypeVar

from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.utils.exception import UnsupportedError

from .base import TLQKernel

__all__ = [
    "TLQKernelDispatchKey",
    "TLQKernelFactory",
    "register_tlq_kernel",
    "create_tlq_kernel",
    "tlq_kernel_dispatch_key",
    "registered_tlq_dispatch_keys",
    "ensure_tlq_kernel_registered",
]

TLQKernelDispatchKey = Tuple[QDType, QDType, QScope, bool]
TLQKernelFactory = Callable[[QConfig], TLQKernel]

_KERNEL_REGISTRY: Dict[TLQKernelDispatchKey, TLQKernelFactory] = {}

F = TypeVar("F", bound=TLQKernelFactory)


def _coerce_dispatch_key(key: Tuple) -> TLQKernelDispatchKey:
    if len(key) != 4:
        raise ValueError(f"TLQ kernel dispatch_key must be (q_dtype, storage_dtype, scope, symmetric), got {key!r}")
    q_dtype, storage_dtype, scope, symmetric = key
    return (
        QDType(q_dtype),
        QDType(storage_dtype),
        QScope(scope),
        bool(symmetric),
    )


def tlq_kernel_dispatch_key(config: QConfig) -> TLQKernelDispatchKey:
    q_dtype = QDType(config.dtype)
    return (q_dtype, q_dtype, QScope(config.scope), bool(config.symmetric))


def register_tlq_kernel(*dispatch_keys: Tuple) -> Callable[[F], F]:
    if not dispatch_keys:
        raise ValueError("register_tlq_kernel requires at least one dispatch_key")

    def decorator(factory: F) -> F:
        for raw_key in dispatch_keys:
            key = _coerce_dispatch_key(raw_key)
            if key in _KERNEL_REGISTRY and _KERNEL_REGISTRY[key] is not factory:
                raise ValueError(
                    f"TLQ kernel dispatch key {key!r} already registered to "
                    f"{_KERNEL_REGISTRY[key]!r}, cannot register {factory!r}"
                )
            _KERNEL_REGISTRY[key] = factory
        return factory

    return decorator


def registered_tlq_dispatch_keys() -> list[TLQKernelDispatchKey]:
    return list(_KERNEL_REGISTRY.keys())


def create_tlq_kernel(config: QConfig) -> TLQKernel:
    key = tlq_kernel_dispatch_key(config)
    if key not in _KERNEL_REGISTRY:
        raise UnsupportedError(
            f"No TLQ kernel for dispatch key {key!r}",
            action=f"Register a kernel for this scheme; registered: {registered_tlq_dispatch_keys()}",
        )
    return _KERNEL_REGISTRY[key](config)


def ensure_tlq_kernel_registered(config: QConfig, role: Literal["weight", "act"]) -> None:
    _ = role
    key = tlq_kernel_dispatch_key(config)
    if key not in _KERNEL_REGISTRY:
        raise UnsupportedError(
            f"No TLQ kernel for {role} config dispatch key {key!r}",
            action=f"Registered keys: {registered_tlq_dispatch_keys()}",
        )
