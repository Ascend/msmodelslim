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

从 Pydantic 配置模型抽取字段、约束、类型与示例事实。
"""

# pylint: disable=too-many-lines
from __future__ import annotations

import enum
import inspect
import re
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic.functional_validators import AfterValidator, BeforeValidator
from pydantic.json_schema import GenerateJsonSchema
from pydantic_core import PydanticUndefined, to_jsonable_python

from tools.config_docgen.records import ClassDocSections, FieldRecord


INTERNAL_TYPE_PREFIX = "_"


DECIMAL_JSON_SCHEMA_PATTERN = r"^(?!^[-+.]*$)[+-]?0*\d*\.?\d*$"


SIMPLE_TYPE_NAMES = {
    str: "string",
    int: "int",
    float: "float",
    bool: "bool",
    dict: "object",
    list: "list",
    type(None): "null",
}


def is_pydantic_model(obj: Any) -> bool:
    return inspect.isclass(obj) and issubclass(obj, BaseModel) and obj is not BaseModel


class DocJsonSchemaGenerator(GenerateJsonSchema):
    """生成文档用 JSON Schema。

    丢掉不可 JSON 化的 ``json_schema_extra``（如 ``exclude_if`` 闭包）；
    无法表示的类型（如 ``torch.dtype``）回退为 string，避免整份 Schema 生成失败。
    """

    def get_decimal_pattern(self, schema):  # type: ignore[override]
        """固定 Decimal 的序列化正则，避免 Pydantic 版本变化导致文档漂移。"""
        return DECIMAL_JSON_SCHEMA_PATTERN

    def generate_inner(self, schema):  # type: ignore[override]
        if isinstance(schema, dict):
            meta = schema.get("metadata")
            extra = meta.get("pydantic_js_extra") if isinstance(meta, dict) else None
            if isinstance(extra, dict):
                cleaned = {}
                for key, value in extra.items():
                    if callable(value):
                        continue
                    try:
                        to_jsonable_python(value)
                    except Exception:  # nosec B112
                        continue
                    cleaned[key] = value
                schema = {**schema, "metadata": {**meta, "pydantic_js_extra": cleaned}}
        return super().generate_inner(schema)

    def handle_invalid_for_json_schema(self, schema, error_info: str):  # type: ignore[override]
        return {"type": "string"}


def model_json_schema_for_docs(model_cls: type) -> Dict[str, Any]:
    return model_cls.model_json_schema(schema_generator=DocJsonSchemaGenerator, mode="serialization")


_MISSING = object()


_PREFERRED_ENUM_VALUES = ("int8", "per_channel", "per_token", "per_tensor", "minmax")


class _Flattened(dict):
    """内部 Union 载体：字段展平到父配置，不在 YAML 里保留该嵌套键。"""


def _dump_extractable_value(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, BaseModel):
        try:
            return value.model_dump(mode="python", exclude_unset=False)
        except Exception:
            return None
    return value


def _min_items(info: FieldInfo) -> Optional[int]:
    if getattr(info, "min_length", None) is not None:
        return int(info.min_length)
    for meta in info.metadata or []:
        if getattr(meta, "min_length", None) is not None:
            return int(meta.min_length)
    return None


def _numeric_from_constraints(info: FieldInfo) -> Any:
    for meta in list(info.metadata or []):
        if getattr(meta, "ge", None) is not None:
            return meta.ge
        if getattr(meta, "gt", None) is not None:
            value = meta.gt
            return value + 1 if isinstance(value, int) and not isinstance(value, bool) else value
        if getattr(meta, "le", None) is not None:
            return meta.le
        if getattr(meta, "lt", None) is not None:
            value = meta.lt
            return value - 1 if isinstance(value, int) and not isinstance(value, bool) else value
    return _MISSING


def _field_info_constraints(info: FieldInfo) -> List[str]:
    """从 FieldInfo 元数据提取数值边界，兼容不输出约束的 Pydantic 版本。"""
    constraints: List[str] = []
    for meta in info.metadata or []:
        for attr, operator in (("gt", ">"), ("ge", "≥"), ("lt", "<"), ("le", "≤")):
            value = getattr(meta, attr, None)
            if value is not None:
                constraints.append(f"{operator}{value}")
    return constraints


def _list_too_short(dumped: Any, info: FieldInfo) -> bool:
    min_items = _min_items(info)
    return bool(isinstance(dumped, list) and min_items and len(dumped) < min_items)


def _pick_enum_or_literal(values: Optional[List[Any]], info: FieldInfo) -> Any:
    if not values:
        return _MISSING
    if not info.is_required() and len(values) > 1:
        return _MISSING
    for preferred in _PREFERRED_ENUM_VALUES:
        if preferred in values:
            return preferred
    return values[0]


def _inner_list_model(core: Any) -> Optional[type]:
    args = get_args(core)
    inner = args[0] if args else None
    if inner is None:
        return None
    inner_core, _ = unwrap_annotation(inner)
    inner_core, _ = split_optional(inner_core)
    return inner_core if inner_core is not None and is_pydantic_model(inner_core) else None


def _extractable_union_value(core: Any, depth: int) -> Any:
    models = [arg for arg in get_args(core) if is_pydantic_model(arg)]
    if not models:
        return _MISSING
    nested = extractable_example_payload(models[0], depth=depth + 1)
    if nested is None:
        return _MISSING
    tag = raw_type_tag(models[0])
    if tag and is_internal_type_tag(tag):
        flattened = {key: value for key, value in nested.items() if not (key == "type" and is_internal_type_tag(value))}
        return _Flattened(flattened)
    return nested


def _extractable_field_value(info: FieldInfo, annotation: Any, depth: int, field_name: str = "") -> Any:
    if info.exclude is True:
        return _MISSING
    if info.default is not PydanticUndefined:
        dumped = _dump_extractable_value(info.default)
        if dumped is None:
            return _MISSING
        if not _list_too_short(dumped, info):
            return dumped
    if info.default_factory is not None:
        try:
            dumped = _dump_extractable_value(info.default_factory())
        except Exception:
            dumped = _MISSING
        else:
            if dumped is not None and dumped is not _MISSING and not _list_too_short(dumped, info):
                return dumped
    literals = _literal_values(annotation)
    picked = _pick_enum_or_literal(literals, info) if literals else _MISSING
    if picked is not _MISSING:
        return picked
    enums = _enum_values(annotation)
    picked = _pick_enum_or_literal(enums, info) if enums else _MISSING
    if picked is not _MISSING:
        return picked
    core, _ = unwrap_annotation(annotation)
    core, opt = split_optional(core)
    origin = get_origin(core)
    if is_pydantic_model(core):
        nested = extractable_example_payload(core, depth=depth + 1)
        if nested is None:
            return _MISSING
        return nested
    if origin is Union:
        value = _extractable_union_value(core, depth)
        if value is _MISSING and (not info.is_required() or opt):
            return _MISSING
        return value
    if origin in (list, List):
        min_items = _min_items(info) or 0
        inner_core = _inner_list_model(core)
        if min_items <= 0 and not info.is_required():
            return _MISSING
        if min_items <= 0:
            return []
        if inner_core is not None:
            item = extractable_example_payload(inner_core, depth=depth + 1)
            if item is None:
                return _MISSING
            return [item]
        return _MISSING
    if not info.is_required() or opt:
        return _MISSING
    bound = _numeric_from_constraints(info)
    if bound is not _MISSING:
        return bound
    if core is bool:
        return True
    if core is str and field_name == "method":
        return "minmax"
    return _MISSING


def extractable_example_payload(model_cls: type, depth: int = 0) -> Optional[Dict[str, Any]]:
    """从默认值、唯一 Literal/Enum、取值范围抽出示例对象。缺必选且无法抽出时返回 None。"""
    if depth > 6:
        return None
    data: Dict[str, Any] = {}
    for name, info in getattr(model_cls, "model_fields", {}).items():
        if info.exclude is True:
            continue
        value = _extractable_field_value(info, info.annotation, depth, field_name=name)
        if isinstance(value, _Flattened):
            for key, item in value.items():
                data.setdefault(key, item)
            continue
        if value is _MISSING:
            if info.is_required():
                return None
            continue
        alias = info.serialization_alias or info.alias or name
        data[alias] = value
    return data


def _assign_parts(cursor: Dict[str, Any], parts: Sequence[str], value: Any) -> None:
    if not parts:
        if isinstance(value, dict):
            cursor.update(value)
        return
    for index, part in enumerate(parts):
        is_list = part.endswith("[]")
        key = part[:-2] if is_list else part
        last = index == len(parts) - 1
        if last:
            cursor[key] = [value] if is_list else value
            return
        if is_list:
            items = cursor.setdefault(key, [{}])
            if not items or not isinstance(items[0], dict):
                cursor[key] = [{}]
                items = cursor[key]
            cursor = items[0]
            continue
        nxt = cursor.setdefault(key, {})
        if not isinstance(nxt, dict):
            cursor[key] = {}
            nxt = cursor[key]
        cursor = nxt


def _assign_path(target: Dict[str, Any], path: str, value: Any) -> None:
    parts = [part for part in path.split(".") if part]
    _assign_parts(target, parts, value)


def wrap_example_at_path(
    payload: Dict[str, Any],
    yaml_path: str,
    *,
    apiversion: str,
    host_item: Optional[Dict[str, Any]] = None,
    category: str = "",
) -> Dict[str, Any]:
    """把当前配置放到模板要求的完整字段路径中。带 [] 的段写成单元素列表。"""
    path = "" if yaml_path in {"", "(根)"} else yaml_path
    if not path:
        data = dict(payload)
        data["apiversion"] = apiversion
        return data
    if path == "spec" or category == "服务规格":
        return {"apiversion": apiversion, "spec": payload}

    rest = path[5:] if path.startswith("spec.") else path
    parts = [part for part in rest.split(".") if part]
    spec: Dict[str, Any] = {}
    if parts and parts[0].endswith("[]"):
        key = parts[0][:-2]
        remainder = parts[1:]
        if remainder:
            item = dict(host_item or {})
            _assign_parts(item, remainder, payload)
        else:
            item = dict(payload)
            if host_item:
                item = {**host_item, **item}
        spec[key] = [item]
        return {"apiversion": apiversion, "spec": spec}
    _assign_parts(spec, parts, payload)
    return {"apiversion": apiversion, "spec": spec}


def _ref_name(ref: str) -> str:
    return ref.rsplit("/", 1)[-1]


def _resolve_schema(node: Any, defs: Dict[str, Any], seen: Optional[Set[str]] = None) -> Dict[str, Any]:
    if not isinstance(node, dict):
        return {}
    seen = seen or set()
    if "$ref" in node:
        name = _ref_name(node["$ref"])
        if name in seen:
            return {k: v for k, v in node.items() if k != "$ref"}
        seen.add(name)
        target = dict(defs.get(name) or {})
        merged = {**target, **{k: v for k, v in node.items() if k != "$ref"}}
        return _resolve_schema(merged, defs, seen)
    if "allOf" in node and node["allOf"]:
        merged: Dict[str, Any] = {}
        for item in node["allOf"]:
            merged.update(_resolve_schema(item, defs, seen))
        merged.update({k: v for k, v in node.items() if k != "allOf"})
        return merged
    return node


def _schema_model_refs(node: Any, defs: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    if not isinstance(node, dict):
        return names
    if "$ref" in node:
        name = _ref_name(node["$ref"])
        target = defs.get(name) or {}
        if "enum" not in target:
            names.append(name)
        for child in (node.get("anyOf"), node.get("oneOf"), node.get("allOf")):
            if isinstance(child, list):
                for item in child:
                    names.extend(_schema_model_refs(item, defs))
        return names
    for key in ("anyOf", "oneOf", "allOf"):
        for item in node.get(key) or []:
            names.extend(_schema_model_refs(item, defs))
    if node.get("type") == "array":
        names.extend(_schema_model_refs(node.get("items"), defs))
    return list(dict.fromkeys(names))


def _schema_type_name(node: Dict[str, Any], defs: Dict[str, Any]) -> str:
    resolved = _resolve_schema(node, defs)
    if "anyOf" in node or "oneOf" in node:
        variants = node.get("anyOf") or node.get("oneOf") or []
        parts = []
        for item in variants:
            part = _schema_type_name(item, defs)
            if part not in parts:
                parts.append(part)
        return " / ".join(parts) if parts else "any"
    if resolved.get("enum") and resolved.get("type") in {None, "string"}:
        return "string"
    if "const" in resolved:
        const = resolved["const"]
        if isinstance(const, bool):
            return "bool"
        if isinstance(const, int) and not isinstance(const, bool):
            return "int"
        if isinstance(const, float):
            return "float"
        return "string"
    json_type = resolved.get("type")
    if isinstance(json_type, list):
        mapped = [_json_type_name(item, resolved, defs) for item in json_type]
        uniq = list(dict.fromkeys(mapped))
        return " / ".join(uniq)
    if json_type:
        return _json_type_name(json_type, resolved, defs)
    if "$ref" in node:
        return "object"
    if "properties" in resolved:
        return "object"
    return "any"


def _json_type_name(json_type: str, node: Dict[str, Any], defs: Dict[str, Any]) -> str:
    if json_type == "integer":
        return "int"
    if json_type == "number":
        return "float"
    if json_type == "boolean":
        return "bool"
    if json_type == "array":
        items = node.get("items") or {}
        inner = _schema_type_name(items, defs) if items else "any"
        return f"list[{inner}]"
    if json_type == "object":
        return "object"
    if json_type == "null":
        return "null"
    return "string"


def _list_element_constraint(text: str) -> str:
    """为数组元素约束补充明确的列表上下文。"""
    return text if text.startswith("列表元素") else f"列表元素：{text}"


def _schema_constraints(node: Dict[str, Any], defs: Dict[str, Any]) -> str:
    resolved = _resolve_schema(node, defs)
    parts: List[str] = []
    const = resolved.get("const")
    if const is not None:
        parts.append(f"`{const}`")
    enums = resolved.get("enum")
    if enums:
        values = [item for item in enums if str(item).lower() != "placeholder"]
        if values:
            parts.append("、".join(f"`{item}`" for item in values))
    if resolved.get("ge") is not None:
        parts.append(f"≥{resolved['ge']}")
    minimum = resolved.get("minimum")
    exclusive_minimum = resolved.get("exclusiveMinimum")
    if minimum is not None:
        if exclusive_minimum is True:
            parts.append(f">{minimum}")
        else:
            parts.append(f"≥{minimum}")
    if resolved.get("gt") is not None:
        parts.append(f">{resolved['gt']}")
    if exclusive_minimum is not None and not isinstance(exclusive_minimum, bool):
        parts.append(f">{exclusive_minimum}")
    if resolved.get("le") is not None:
        parts.append(f"≤{resolved['le']}")
    maximum = resolved.get("maximum")
    exclusive_maximum = resolved.get("exclusiveMaximum")
    if maximum is not None:
        if exclusive_maximum is True:
            parts.append(f"<{maximum}")
        else:
            parts.append(f"≤{maximum}")
    if resolved.get("lt") is not None:
        parts.append(f"<{resolved['lt']}")
    if exclusive_maximum is not None and not isinstance(exclusive_maximum, bool):
        parts.append(f"<{exclusive_maximum}")
    if resolved.get("minLength") is not None:
        parts.append(f"字符串长度 ≥ {resolved['minLength']}")
    if resolved.get("maxLength") is not None:
        parts.append(f"字符串长度 ≤ {resolved['maxLength']}")
    if resolved.get("pattern"):
        parts.append(f"`{resolved['pattern']}`")
    if resolved.get("minItems") is not None:
        parts.append(f"最少{resolved['minItems']}项")
    if resolved.get("maxItems") is not None:
        parts.append(f"最多{resolved['maxItems']}项")
    items = resolved.get("items")
    if isinstance(items, dict):
        inner = _schema_constraints(items, defs)
        if inner and inner != "—":
            item_constraint = _list_element_constraint(inner)
            if item_constraint not in parts:
                parts.append(item_constraint)
    variants: List[Dict[str, Any]] = []
    for key in ("anyOf", "oneOf"):
        variants.extend(item for item in resolved.get(key) or [] if isinstance(item, dict))
        if resolved is not node:
            variants.extend(item for item in node.get(key) or [] if isinstance(item, dict))
    for item in variants:
        inner = _schema_constraints(item, defs)
        if inner and inner != "—":
            parts.append(inner)
    return "；".join(dict.fromkeys(parts)) if parts else "—"


def _schema_default(node: Dict[str, Any], defs: Dict[str, Any]) -> Optional[str]:
    resolved = _resolve_schema(node, defs)
    if "default" not in node and "default" not in resolved:
        return None
    value = node["default"] if "default" in node else resolved["default"]
    return _scalar_text(value)


def fields_from_json_schema(schema: Dict[str, Any], name_of: Any = None) -> List[FieldRecord]:
    name_of = name_of or (lambda item: getattr(item, "__name__", str(item)))
    defs = schema.get("$defs") or schema.get("definitions") or {}
    required = set(schema.get("required") or [])
    records: List[FieldRecord] = []
    for name, spec in (schema.get("properties") or {}).items():
        if not isinstance(spec, dict):
            continue
        nested = []
        for ref_name in _schema_model_refs(spec, defs):
            try:
                nested.append(name_of(ref_name))
            except Exception:
                nested.append(ref_name)
        default = _schema_default(spec, defs)
        records.append(
            FieldRecord(
                name=name,
                type_name=_schema_type_name(spec, defs),
                required=name in required,
                default="无" if default is None and name in required else (default or "无"),
                constraint=_schema_constraints(spec, defs),
                description=(spec.get("description") or _resolve_schema(spec, defs).get("description") or "").strip()
                or "—",
                nested_models=nested,
                is_union_type="anyOf" in spec or "oneOf" in spec,
                discriminator=(spec.get("discriminator") or {}).get("propertyName")
                if isinstance(spec.get("discriminator"), dict)
                else None,
            )
        )
    return records


def unwrap_annotation(annotation: Any) -> Tuple[Any, List[Any]]:
    metadata: List[Any] = []
    current = annotation
    while True:
        origin = get_origin(current)
        if origin is None:
            return current, metadata
        name = getattr(origin, "__name__", "") or str(origin)
        origin_str = str(origin)
        if origin is AfterValidator or origin is BeforeValidator or name in {"AfterValidator", "BeforeValidator"}:
            metadata.append(current)
            return current, metadata
        args = get_args(current)
        if name == "SerializeAsAny" or origin_str.endswith("SerializeAsAny"):
            current = args[0] if args else current
            continue
        if name == "Annotated" or origin_str.endswith("Annotated"):
            if not args:
                return current, metadata
            metadata.extend(args[1:])
            current = args[0]
            continue
        return current, metadata


def split_optional(annotation: Any) -> Tuple[Any, bool]:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Union and type(None) in args:  # pylint: disable=unidiomatic-typecheck
        rest = tuple(a for a in args if a is not type(None))  # pylint: disable=unidiomatic-typecheck
        if len(rest) == 1:
            return rest[0], True
        return Union[rest], True
    return annotation, False


def format_type_name(annotation: Any) -> str:
    annotation, _ = unwrap_annotation(annotation)
    annotation, optional = split_optional(annotation)
    origin = get_origin(annotation)
    args = get_args(annotation)
    if optional and origin is None:
        inner = format_type_name(annotation)
        return inner if inner.endswith(" / null") else f"{inner} / null"
    if annotation in SIMPLE_TYPE_NAMES:
        return SIMPLE_TYPE_NAMES[annotation]
    if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
        return "string"
    if is_pydantic_model(annotation):
        return "object"
    if origin in (list, List) or origin is list:
        item = format_type_name(args[0]) if args else "any"
        return f"list[{item}]"
    if origin in (dict, Dict) or origin is dict:
        return "object"
    if origin is Union:
        parts = [format_type_name(a) for a in args if a is not type(None)]  # pylint: disable=unidiomatic-typecheck
        uniq = []
        for part in parts:
            if part not in uniq:
                uniq.append(part)
        if optional or type(None) in args:  # pylint: disable=unidiomatic-typecheck
            uniq.append("null")
        return " / ".join(uniq)
    if origin is not None and str(origin).endswith("Literal"):
        return "string" if all(isinstance(a, str) for a in args) else "any"
    if inspect.isclass(annotation):
        return annotation.__name__
    return "any"


def _literal_values(annotation: Any) -> Optional[List[Any]]:
    annotation, _ = unwrap_annotation(annotation)
    origin = get_origin(annotation)
    if origin is not None and str(origin).endswith("Literal"):
        return list(get_args(annotation))
    return None


def _enum_values(annotation: Any) -> Optional[List[str]]:
    annotation, _ = unwrap_annotation(annotation)
    annotation, _ = split_optional(annotation)
    if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
        values = []
        for item in annotation:
            value = item.value if hasattr(item, "value") else item
            if str(value).lower() == "placeholder":
                continue
            values.append(str(value))
        return values
    return None


def collect_nested_models(annotation: Any) -> List[type]:
    return _nested_models(annotation)


def annotation_refers_to(annotation: Any, base_cls: type) -> bool:
    annotation, _ = unwrap_annotation(annotation)
    annotation, _ = split_optional(annotation)
    if is_pydantic_model(annotation) and issubclass(annotation, base_cls):
        return True
    origin = get_origin(annotation)
    if origin is None:
        return False
    return any(annotation_refers_to(arg, base_cls) for arg in get_args(annotation))


def _nested_models(annotation: Any) -> List[type]:
    annotation, _ = unwrap_annotation(annotation)
    annotation, _ = split_optional(annotation)
    origin = get_origin(annotation)
    found: List[type] = []
    if is_pydantic_model(annotation):
        found.append(annotation)
    elif origin in (list, List, dict, Dict, Union) or origin in (list, dict):
        for arg in get_args(annotation):
            found.extend(_nested_models(arg))
    return found


def _default_text(info: FieldInfo) -> str:
    if info.default_factory is not None:
        try:
            value = info.default_factory()
            return _scalar_text(value)
        except Exception:
            return "由工厂函数生成"
    if info.default is PydanticUndefined:
        return "无"
    return _scalar_text(info.default)


def _scalar_text(value: Any) -> str:
    if value is None:
        return "`null`"
    if isinstance(value, enum.Enum):
        return f"`{value.value}`"
    if isinstance(value, BaseModel):
        return "见嵌套配置默认值"
    if isinstance(value, (list, dict)):
        return f"`{value!r}`"
    if isinstance(value, bool):
        return "`true`" if value else "`false`"
    if isinstance(value, str):
        return f"`{value}`"
    return f"`{value}`"


def _validator_belongs_to_class(func: Any, cls_name: str) -> bool:
    """validator 是否直接定义在模型类上（排除从基类继承的内部 validator）。"""
    qualname = getattr(func, "__qualname__", "") or ""
    return qualname.split(".")[0] == cls_name


def is_internal_validator(func: Any) -> bool:
    """validator 是否显式标记为仅用于内部实现。"""
    candidates = (func, getattr(func, "__func__", None), getattr(func, "func", None))
    return any(getattr(candidate, "config_internal", False) for candidate in candidates if candidate is not None)


def collect_validator_docs(model_cls: type) -> List[str]:
    docs: List[str] = []
    decorators = getattr(model_cls, "__pydantic_decorators__", None)
    if decorators is None:
        return docs
    cls_name = getattr(model_cls, "__name__", "")
    for group_name in ("field_validators", "model_validators"):
        group = getattr(decorators, group_name, {}) or {}
        for wrapped in group.values():
            func = getattr(wrapped, "func", wrapped)
            if is_internal_validator(func):
                continue
            if not _validator_belongs_to_class(func, cls_name):
                continue
            text = inspect.getdoc(func) or ""
            text = " ".join(text.split())
            if text:
                docs.append(text)
    return docs


def is_internal_type_tag(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(INTERNAL_TYPE_PREFIX)


def raw_type_tag(model_cls: type) -> Optional[str]:
    info = model_cls.model_fields.get("type")
    if info is None:
        return None
    literals = _literal_values(info.annotation)
    if not literals or len(literals) != 1:
        return None
    tag = literals[0]
    return str(tag) if isinstance(tag, str) else None


def public_type_tag(model_cls: type) -> Optional[str]:
    tag = raw_type_tag(model_cls)
    if tag is None or is_internal_type_tag(tag):
        return None
    return tag


def rewrite_example_internal_types(
    data: Any,
    *,
    public_format_type: str = "ascendv1_saver",
    internal_format_tags: Optional[Set[str]] = None,
) -> Any:
    """用户可复制的示例不得出现内部分派 type。

    `_auto_save` 等保存器基类默认值替换为公开 format；其余内部 type 从示例中删除
    （由父配置写入，用户不必填写）。
    """
    format_tags = internal_format_tags or {"_auto_save"}
    if isinstance(data, dict):
        rewritten = {
            key: rewrite_example_internal_types(
                value, public_format_type=public_format_type, internal_format_tags=format_tags
            )
            for key, value in data.items()
        }
        tag = rewritten.get("type")
        if is_internal_type_tag(tag):
            if tag in format_tags:
                rewritten["type"] = public_format_type
            else:
                rewritten.pop("type", None)
        return rewritten
    if isinstance(data, list):
        return [
            rewrite_example_internal_types(
                item, public_format_type=public_format_type, internal_format_tags=format_tags
            )
            for item in data
        ]
    return data


def snake_name(name: str) -> str:
    for src, dst in (
        ("FA3", "Fa3"),
        ("TLQ", "Tlq"),
        ("QConfig", "Qconfig"),
        ("SDConfig", "SdConfig"),
        ("SDModelslim", "SdModelslim"),
        ("SDService", "SdService"),
    ):
        name = name.replace(src, dst)
    slug = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return re.sub(r"_+", "_", slug).strip("_")


def slugify_class(model_cls: type) -> str:
    tag = public_type_tag(model_cls)
    if tag:
        return re.sub(r"[^a-z0-9_]+", "_", tag.lower()).strip("_")
    return snake_name(model_cls.__name__) or model_cls.__name__.lower()


def _field_info_by_schema_name(model_cls: type) -> Dict[str, FieldInfo]:
    mapping: Dict[str, FieldInfo] = {}
    for info in getattr(model_cls, "model_fields", {}).values():
        names = [info.serialization_alias, info.alias]
        for name in names:
            if name:
                mapping[name] = info
    for name, info in getattr(model_cls, "model_fields", {}).items():
        mapping.setdefault(name, info)
    return mapping


def _validator_closure_value(func: Any, name: str) -> Any:
    """读取 validator 闭包中的文档参数，不执行 validator。"""
    try:
        values = inspect.getclosurevars(func)
    except (TypeError, ValueError):
        return None
    return values.nonlocals.get(name, values.globals.get(name))


def _iter_after_validators(value: Any, *, list_item: bool = False):
    """递归查找 AfterValidator，并标记其是否作用于列表元素。"""
    if isinstance(value, AfterValidator):
        yield value, list_item
        return
    child_list_item = list_item or get_origin(value) is list
    for arg in get_args(value):
        yield from _iter_after_validators(arg, list_item=child_list_item)


def _legacy_validator_constraint(func: Any) -> str:
    """兼容尚未迁移到 ``config_constraint`` 的项目通用 validator。"""
    name = getattr(func, "__name__", "")
    qualname = getattr(func, "__qualname__", name)
    if name == "at_least_one_element":
        return "列表至少包含 1 项"
    if name in {"greater_than_zero", "int_greater_than_zero"}:
        return ">0"
    if name == "validate_normalized_value":
        return "0 < 值 < 1"
    if name == "is_string_list":
        return "列表元素必须全部为字符串"
    if name == "non_empty_string":
        return "非空字符串"
    if name == "is_port":
        return "1–65535"
    if name == "is_safe_host":
        return "合法主机名/IP，仅允许本机、回环或私有/链路本地地址"
    if name == "is_safe_endpoint":
        return "以 `/` 开头的安全路径，禁止 `..`、`//` 和非安全字符"
    if name == "allow_empty_list":
        return "列表（允许为空列表）"
    if name == "allow_empty_dict":
        return "对象（允许为空对象）"
    if name == "at_least_one_key":
        return "对象至少包含 1 个键值对"
    if "validate_str_length" in qualname:
        max_len = _validator_closure_value(func, "max_len")
        if isinstance(max_len, int) and max_len > 0:
            return f"字符串长度 ≤ {max_len}"
    if "in_range" in qualname:
        min_val = _validator_closure_value(func, "min_val")
        max_val = _validator_closure_value(func, "max_val")
        parts: List[str] = []
        if min_val is not None:
            parts.append(f"≥{min_val}")
        if max_val is not None:
            parts.append(f"≤{max_val}")
        return "；".join(parts)
    return ""


def iter_field_after_validators(info: FieldInfo):
    """Yield unique AfterValidator instances and whether they validate list items."""
    seen: Set[Tuple[int, bool]] = set()
    for value in [*info.metadata, info.annotation]:
        for metadata, list_item in _iter_after_validators(value):
            key = (id(metadata), list_item)
            if key in seen:
                continue
            seen.add(key)
            yield metadata, list_item


def validator_constraint_text(func: Any) -> str:
    """Return the source-declared documentation constraint for a reusable validator."""
    text = getattr(func, "config_constraint", None)
    if not isinstance(text, str) or not text.strip():
        text = _legacy_validator_constraint(func)
    return text.strip() if isinstance(text, str) else ""


def _field_validator_constraints(info: FieldInfo) -> List[str]:
    """把 AfterValidator 的运行期约束补充到文档取值范围列。

    JSON Schema 无法表达自定义 ``AfterValidator``。生成器优先读取 validator
    函数上的 ``config_constraint`` 属性，未迁移的通用 validator 再回退到函数名
    与闭包参数映射。字段注解会递归扫描，以覆盖列表元素中的 ``Annotated``。
    """
    constraints: List[str] = []
    for metadata, list_item in iter_field_after_validators(info):
        text = validator_constraint_text(metadata.func)
        if list_item and text:
            text = _list_element_constraint(text)
        if text and text not in constraints:
            constraints.append(text)
    return constraints


def _merge_constraints(base: str, additions: Sequence[str]) -> str:
    parts = [] if base == "—" else [part for part in base.split("；") if part]
    for addition in additions:
        for part in addition.split("；"):
            if part and part not in parts:
                parts.append(part)
    return "；".join(parts) if parts else "—"


def extract_fields(model_cls: type, name_of: Any = None) -> List[FieldRecord]:
    """从 JSON Schema 抽取字段表；Schema 未给出 default 时回退 Field / default_factory。"""
    name_of = name_of or (lambda item: getattr(item, "__name__", str(item)))
    schema = model_json_schema_for_docs(model_cls)
    records = fields_from_json_schema(schema, name_of=name_of)
    infos = _field_info_by_schema_name(model_cls)
    computed = set(getattr(model_cls, "model_computed_fields", {}) or {})
    kept: List[FieldRecord] = []
    for record in records:
        if record.name in computed or record.name not in infos:
            continue
        info = infos[record.name]
        # 用注解里的真实嵌套模型解析 slug，避免 JSON Schema $ref 同名（如三个 QuantStrategyConfig）时
        # 按类名取到第一个定义、导致指向错误的配置文档。
        model_nested = collect_nested_models(info.annotation)
        if model_nested:
            record.nested_models = [name_of(nested) for nested in model_nested]
        if record.default == "无" and not record.required:
            record.default = _default_text(info)
        # Decimal 等字段的数值边界在部分 Pydantic 版本中不进入序列化 Schema；
        # AfterValidator 也不进入 JSON Schema。两者都从源码注解补入取值范围列。
        record.constraint = _merge_constraints(
            record.constraint,
            [*_field_info_constraints(info), *_field_validator_constraints(info)],
        )
        # 取值范围列补足默认值语义：默认 null 的可空字段补充「或 null」；
        # 默认空列表且非必填的 list 字段把 minItems 表达为「0项或≥N项」。
        if record.constraint and record.constraint != "—":
            if record.default == "`null`" and "null" not in record.constraint:
                record.constraint = f"{record.constraint}；或 null"
            if not record.required and "最少" in record.constraint:
                try:
                    default_value = (
                        info.default_factory()
                        if info.default_factory is not None
                        else (info.default if info.default is not PydanticUndefined else None)
                    )
                except Exception:
                    default_value = None
                if default_value in ([], {}, ""):
                    record.constraint = re.sub(r"最少(\d+)项", r"0项或≥\1项", record.constraint)
        if record.description == "—" and info.description:
            record.description = info.description.strip() or "—"
        kept.append(record)
    return kept


def _join_docstring_lines(paragraph: str) -> str:
    """合并段内硬换行；中文换行不额外插入空格，英文和行内代码边界保留间隔。"""
    cjk = re.compile(r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]")
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if not lines:
        return ""
    merged = lines[0]
    for line in lines[1:]:
        if line.startswith("`") and merged[-1] not in "（《“‘【、，。；：！？）》】":
            separator = " "
        else:
            separator = "" if cjk.search(merged[-1]) or cjk.search(line[0]) else " "
        merged += separator + line
    return merged


def first_paragraph(text: str) -> str:
    """返回 docstring 首段并合并段内硬换行，忽略后续段落。"""
    text = textwrap.dedent(text or "").strip()
    if not text:
        return ""
    paragraph = re.split(r"\n\s*\n", text, maxsplit=1)[0]
    return _join_docstring_lines(paragraph)


def class_doc_sections(model_cls: type) -> ClassDocSections:
    """解析配置类自身 docstring 的概述和固定说明小节。

    首段作为配置概述；只有 ``说明``、``约束`` 和 ``注意事项`` 固定标题下的
    内容进入生成文档，其他段落保留在源码但不参与抽取。每个列表项可以跨多行。
    """
    raw = model_cls.__dict__.get("__doc__") or ""
    cleaned = inspect.cleandoc(raw).strip()
    if not cleaned:
        return ClassDocSections(summary="")

    parts = re.split(r"\n\s*\n", cleaned, maxsplit=1)
    remaining = parts[1] if len(parts) > 1 else ""
    items: Dict[str, List[str]] = {
        "description": [],
        "constraints": [],
        "notes": [],
    }
    headings = {
        "说明": "description",
        "约束": "constraints",
        "注意事项": "notes",
    }
    current_section: Optional[str] = None
    current_item: Optional[str] = None

    def flush_item() -> None:
        nonlocal current_item
        if current_section and current_item:
            merged = _join_docstring_lines(current_item)
            if merged:
                items[current_section].append(merged)
        current_item = None

    for raw_line in remaining.splitlines():
        line = raw_line.strip()
        if not line:
            flush_item()
            continue

        heading_match = re.fullmatch(r"([^：:\s]{1,16})[：:]", line)
        if heading_match:
            flush_item()
            current_section = headings.get(heading_match.group(1).strip())
            continue

        bullet_match = re.match(r"^-\s+(.+)$", line)
        if current_section is None:
            continue

        if bullet_match:
            flush_item()
            current_item = bullet_match.group(1)
        elif current_item is None:
            current_item = line
        else:
            current_item += "\n" + line

    flush_item()
    return ClassDocSections(
        summary=first_paragraph(cleaned),
        description=tuple(items["description"]),
        constraints=tuple(items["constraints"]),
        notes=tuple(items["notes"]),
    )


def class_summary(model_cls: type) -> str:
    """返回配置类自身 docstring 的首段。"""
    return class_doc_sections(model_cls).summary


def example_value(info: FieldInfo, annotation: Any, depth: int = 0, field_name: str = "") -> Any:
    if info.exclude is True:
        return None
    if info.default is not PydanticUndefined:
        value = info.default
        if isinstance(value, enum.Enum):
            return value.value
        if isinstance(value, BaseModel):
            try:
                return value.model_dump(mode="python")
            except Exception:
                return example_model(type(value), depth + 1)
        return value
    if info.default_factory is not None:
        try:
            value = info.default_factory()
            if isinstance(value, BaseModel):
                return example_model(type(value), depth + 1)
            return value
        except Exception:  # nosec B110
            pass
    literals = _literal_values(annotation)
    if literals:
        return literals[0]
    enums = _enum_values(annotation)
    if enums:
        for preferred in ("int8", "per_channel", "per_token", "minmax"):
            if preferred in enums:
                return preferred
        return enums[0]
    core, _ = unwrap_annotation(annotation)
    core, _ = split_optional(core)
    origin = get_origin(core)
    if core is bool:
        return True
    if core is int:
        return 0
    if core is float:
        return 0.0
    if core is str:
        if field_name == "method":
            return "minmax"
        return "example"
    if origin in (list, List):
        return []
    if origin in (dict, Dict):
        return {}
    if is_pydantic_model(core) and depth < 2:
        return example_model(core, depth + 1)
    return None


def example_model(model_cls: type, depth: int = 0) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for name, info in model_cls.model_fields.items():
        if info.exclude is True:
            continue
        value = example_value(info, info.annotation, depth, name)
        if value is None and not info.is_required():
            continue
        alias = info.serialization_alias or info.alias or name
        data[alias] = value
    return data
