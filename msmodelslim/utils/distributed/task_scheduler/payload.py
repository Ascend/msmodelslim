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

Submit 载荷校验与跨 rank 语义哈希（与调度 backend 无关）。
"""

import functools
import hashlib
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn

from msmodelslim.utils.exception import SchemaValidateError

from msmodelslim.utils.distributed.task_scheduler.types import TaskExecutionRecord, TaskSyncContext


def validate_dependency_paths(model: nn.Module, dependencies: Optional[List[str]]) -> None:
    """在 ``submit`` 阶段校验 ``dependencies`` 均可被 ``model.get_submodule`` 解析，避免拖到 ``run`` 同步阶段才失败。"""
    for raw in dependencies or []:
        if raw is None:
            raise SchemaValidateError(
                "DTS submit: dependency path must not be None",
                action=(
                    "Set dependencies to a list of valid module paths, e.g. "
                    "['model.layers.0.self_attn.q_proj']; use [] when no module is targeted."
                ),
            )
        path = str(raw).strip().strip(".")
        if not path:
            raise SchemaValidateError(
                "DTS submit: each dependency must be a non-empty module path (after strip).",
                action=(
                    "Remove empty items in dependencies and provide canonical submodule paths; "
                    "use dependencies=[] when no module is targeted."
                ),
            )
        try:
            model.get_submodule(path)
        except AttributeError as e:
            raise SchemaValidateError(
                f"DTS submit: invalid dependency path {raw!r} (resolved as {path!r}): not found under model.",
                action=(
                    "Check model.named_modules() to confirm full path names, and ensure the path "
                    "is valid on every rank when using distributed execution."
                ),
            ) from e


def normalize_hash_value(value: Any, field_name: str) -> Any:
    """将 submit 参数规约为稳定 JSON 结构；不可稳定规约时 fail-fast。"""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        return {
            "__tensor__": True,
            "shape": tuple(int(i) for i in value.shape),
            "dtype": str(value.dtype),
            "device_type": str(value.device.type),
            "requires_grad": bool(value.requires_grad),
        }
    if isinstance(value, (tuple, list)):
        return [normalize_hash_value(v, field_name) for v in value]
    if isinstance(value, set):
        norm_vals = [normalize_hash_value(v, field_name) for v in value]
        return sorted(norm_vals, key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False))
    if isinstance(value, dict):
        normalized_items = []
        for k, v in value.items():
            nk = normalize_hash_value(k, field_name)
            nv = normalize_hash_value(v, field_name)
            normalized_items.append((nk, nv))
        normalized_items.sort(key=lambda kv: json.dumps(kv[0], sort_keys=True, ensure_ascii=False))
        return {json.dumps(k, sort_keys=True, ensure_ascii=False): v for k, v in normalized_items}
    raise SchemaValidateError(
        f"DTS submit: unsupported value type {type(value)!r} in {field_name!r} for semantic hash.",
        action=(
            "Restrict args/kwargs/dependencies to stable JSON-like values (or Tensor), "
            "and avoid runtime-only objects in submit payload."
        ),
    )


def stable_callable_identifier(cb: Optional[Callable[..., Any]], field_name: str) -> Optional[Any]:
    """将 callable 规约为跨 rank 可比较的稳定标识。"""
    if cb is None:
        return None
    if isinstance(cb, functools.partial):
        return {
            "__partial__": True,
            "func": stable_callable_identifier(cb.func, field_name),
            "args": normalize_hash_value(tuple(cb.args), f"{field_name}.partial_args"),
            "kwargs": normalize_hash_value(dict(cb.keywords or {}), f"{field_name}.partial_kwargs"),
        }
    module = getattr(cb, "__module__", None)
    qualname = getattr(cb, "__qualname__", None)
    name = getattr(cb, "__name__", None)
    if isinstance(module, str) and isinstance(qualname, str):
        return f"{module}:{qualname}"
    if isinstance(module, str) and isinstance(name, str):
        return f"{module}:{name}"
    raise SchemaValidateError(
        f"DTS submit: unsupported callable in {field_name!r}; cannot derive stable identifier.",
        action=(
            "Use top-level function / bound method / functools.partial with stable __module__/__qualname__, "
            "or pass only serializable metadata in args/kwargs."
        ),
    )


def task_semantic_hash(
    fn: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]],
    dependencies: List[str],
    parallel: bool,
    sync_fn: Optional[Callable[[TaskExecutionRecord, TaskSyncContext], Any]],
) -> str:
    """为 submit 语义生成跨 rank 可比较哈希。"""
    payload = {
        "fn": stable_callable_identifier(fn, "fn"),
        "sync_fn": stable_callable_identifier(sync_fn, "sync_fn"),
        "args": normalize_hash_value(tuple(args), "args"),
        "kwargs": normalize_hash_value(dict(kwargs or {}), "kwargs"),
        "dependencies": normalize_hash_value(list(dependencies), "dependencies"),
        "parallel": bool(parallel),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def wave_semantic_hash(task_hashes: List[str]) -> str:
    """对单 wave 的任务语义哈希序列再聚合一次，降低跨 rank 对比开销。"""
    raw = json.dumps(list(task_hashes), sort_keys=False, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
