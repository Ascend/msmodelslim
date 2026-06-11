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

from typing import Any, List, Union

import pydantic

from msmodelslim.utils.exception import SchemaValidateError

PathSegment = Union[str, int]


def _parse_path(path: str) -> List[PathSegment]:
    if not path or not path.strip():
        raise SchemaValidateError("pydantic model path must be a non-empty dot-separated string")
    segments: List[PathSegment] = []
    for raw in path.split("."):
        if not raw:
            raise SchemaValidateError(f"invalid pydantic model path {path!r}: empty segment")
        if raw.isdigit():
            segments.append(int(raw))
        else:
            segments.append(raw)
    return segments


def _resolve_parent_and_key(model: pydantic.BaseModel, path: str) -> tuple[Any, PathSegment, List[PathSegment]]:
    segments = _parse_path(path)
    current: Any = model
    for segment in segments[:-1]:
        current = _get_child(current, segment, path)
    return current, segments[-1], segments


def _is_model_node(obj: Any) -> bool:
    return callable(getattr(obj, "model_copy", None))


def _get_child(current: Any, segment: PathSegment, path: str) -> Any:
    if isinstance(segment, int):
        if not isinstance(current, list):
            raise SchemaValidateError(
                f"invalid pydantic model path {path!r}: segment {segment!r} expects list parent, "
                f"got {type(current).__name__}"
            )
        try:
            return current[segment]
        except IndexError as exc:
            raise SchemaValidateError(
                f"invalid pydantic model path {path!r}: list index {segment} out of range"
            ) from exc
    if _is_model_node(current):
        try:
            return getattr(current, str(segment))
        except AttributeError as exc:
            raise SchemaValidateError(
                f"invalid pydantic model path {path!r}: attribute {segment!r} not found on {type(current).__name__}"
            ) from exc
    if isinstance(current, dict):
        if segment not in current:
            raise SchemaValidateError(f"invalid pydantic model path {path!r}: key {segment!r} not found")
        return current[segment]
    raise SchemaValidateError(
        f"invalid pydantic model path {path!r}: cannot traverse segment {segment!r} on {type(current).__name__}"
    )


def _set_child(parent: Any, segment: PathSegment, value: Any, path: str) -> None:
    if isinstance(segment, int):
        if not isinstance(parent, list):
            raise SchemaValidateError(
                f"invalid pydantic model path {path!r}: segment {segment!r} expects list parent, "
                f"got {type(parent).__name__}"
            )
        try:
            parent[segment] = value
        except IndexError as exc:
            raise SchemaValidateError(
                f"invalid pydantic model path {path!r}: list index {segment} out of range"
            ) from exc
        return
    if _is_model_node(parent):
        try:
            setattr(parent, str(segment), value)
        except (AttributeError, TypeError) as exc:
            raise SchemaValidateError(
                f"invalid pydantic model path {path!r}: attribute {segment!r} not found on {type(parent).__name__}"
            ) from exc
        return
    if isinstance(parent, dict):
        parent[segment] = value
        return
    raise SchemaValidateError(
        f"invalid pydantic model path {path!r}: cannot assign segment {segment!r} on {type(parent).__name__}"
    )


def resolve_pydantic_model_path(model: pydantic.BaseModel, path: str) -> Any:
    """Read a nested field from a Pydantic model using a dot-separated path."""
    parent, key, _ = _resolve_parent_and_key(model, path)
    if isinstance(key, int):
        if not isinstance(parent, list):
            raise SchemaValidateError(f"invalid pydantic model path {path!r}: segment {key!r} expects list parent")
        try:
            return parent[key]
        except IndexError as exc:
            raise SchemaValidateError(f"invalid pydantic model path {path!r}: list index {key} out of range") from exc
    if _is_model_node(parent):
        try:
            return getattr(parent, str(key))
        except AttributeError as exc:
            raise SchemaValidateError(f"invalid pydantic model path {path!r}: cannot read segment {key!r}") from exc
    if isinstance(parent, dict):
        return parent[str(key)]
    raise SchemaValidateError(f"invalid pydantic model path {path!r}: cannot read segment {key!r}")


def set_pydantic_model_path(model: pydantic.BaseModel, path: str, value: Any) -> pydantic.BaseModel:
    """Return a deep copy of model with the nested field at path set to value."""
    copied = model.model_copy(deep=True)
    parent, key, _ = _resolve_parent_and_key(copied, path)
    _set_child(parent, key, value, path)
    return copied


def validate_pydantic_model_list_path(model: pydantic.BaseModel, path: str) -> None:
    """Ensure path exists and resolves to a list."""
    value = resolve_pydantic_model_path(model, path)
    if not isinstance(value, list):
        raise SchemaValidateError(f"pydantic model path {path!r} must resolve to a list, got {type(value).__name__}")
