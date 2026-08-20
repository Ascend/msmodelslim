#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Pydantic 配置模型 → 量化配置文档（模板 04）渲染。

本模块不导入 msmodelslim，便于用最小 pydantic 模型做单测。
"""

# pylint: disable=too-many-lines

from __future__ import annotations

import enum
import inspect
import posixpath
import re
import textwrap
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic.functional_validators import AfterValidator, BeforeValidator
from pydantic.json_schema import GenerateJsonSchema
from pydantic_core import PydanticUndefined, to_jsonable_python

GENERATED_MARKER = "generated-by: skills/docs-management/scripts/gen_quant_config_docs.py"

# YAML `type` 以该前缀开头的配置视为内部类：不进入公开分派索引，示例不得当作用户可写 type。
INTERNAL_TYPE_PREFIX = "_"

SIMPLE_TYPE_NAMES = {
    str: "string",
    int: "int",
    float: "float",
    bool: "bool",
    dict: "object",
    list: "list",
    type(None): "null",
}


@dataclass
class FieldRecord:
    name: str
    type_name: str
    required: bool
    default: str
    constraint: str
    description: str
    nested_models: List[str] = field(default_factory=list)
    is_union_type: bool = False
    discriminator: Optional[str] = None


@dataclass
class ModelRecord:
    qualname: str
    class_name: str
    slug: str
    title: str
    summary: str
    source_rel: str
    yaml_path: str
    category: str
    type_tag: Optional[str]
    fields: List[FieldRecord]
    constraints: List[str]
    nested_refs: List[Tuple[str, str, str]]
    parents: List[Tuple[str, str, str]] = field(default_factory=list)
    # field_path → type 分派基础类显示名（如 select_best → SelectBestConfig）
    dispatch_bases: Dict[str, str] = field(default_factory=dict)
    example_yaml: str = ""
    # 类 docstring 首段（无 docstring 时为空字符串）；用于嵌套配置块的类级描述
    class_summary: str = ""

    @property
    def filename(self) -> str:
        return f"{self.slug}.md"


def is_pydantic_model(obj: Any) -> bool:
    return inspect.isclass(obj) and issubclass(obj, BaseModel) and obj is not BaseModel


class DocJsonSchemaGenerator(GenerateJsonSchema):
    """生成文档用 JSON Schema。

    丢掉不可 JSON 化的 ``json_schema_extra``（如 ``exclude_if`` 闭包）；
    无法表示的类型（如 ``torch.dtype``）回退为 string，避免整份 Schema 生成失败。
    """

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


def extract_declared_example(model_cls: type) -> Optional[Any]:
    """只取模型显式声明的完整示例，不根据默认值拼装。"""
    extra = getattr(model_cls, "model_config", None) or {}
    js_extra = extra.get("json_schema_extra") if hasattr(extra, "get") else None
    schema = model_json_schema_for_docs(model_cls)
    chunks: List[Any] = []
    for source in (js_extra if isinstance(js_extra, dict) else {}, schema):
        if "examples" in source:
            chunks.append(source["examples"])
        if "example" in source:
            chunks.append(source["example"])
    for chunk in chunks:
        if chunk in (None, "", {}, []):
            continue
        if isinstance(chunk, list):
            for item in chunk:
                if item not in (None, "", {}, []):
                    return item
            continue
        return chunk
    return None


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
    if resolved.get("minimum") is not None:
        parts.append(f"≥{resolved['minimum']}")
    if resolved.get("exclusiveMinimum") is not None and not isinstance(resolved.get("exclusiveMinimum"), bool):
        parts.append(f">{resolved['exclusiveMinimum']}")
    if resolved.get("maximum") is not None:
        parts.append(f"≤{resolved['maximum']}")
    if resolved.get("exclusiveMaximum") is not None and not isinstance(resolved.get("exclusiveMaximum"), bool):
        parts.append(f"<{resolved['exclusiveMaximum']}")
    if resolved.get("minLength") is not None:
        parts.append(f"长度 ≥ {resolved['minLength']}")
    if resolved.get("maxLength") is not None:
        parts.append(f"长度 ≤ {resolved['maxLength']}")
    if resolved.get("pattern"):
        parts.append(f"`{resolved['pattern']}`")
    if resolved.get("minItems") is not None:
        parts.append(f"最少{resolved['minItems']}项")
    if resolved.get("maxItems") is not None:
        parts.append(f"最多{resolved['maxItems']}项")
    items = resolved.get("items")
    if isinstance(items, dict):
        inner = _schema_constraints(items, defs)
        if inner and inner != "—" and inner not in parts:
            parts.append(inner)
    if "anyOf" in node:
        for item in node.get("anyOf") or []:
            inner = _schema_constraints(item, defs)
            if inner and inner != "—" and inner not in parts:
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
        if record.description == "—" and info.description:
            record.description = info.description.strip() or "—"
        kept.append(record)
    return kept


def first_paragraph(text: str) -> str:
    text = textwrap.dedent(text or "").strip()
    if not text:
        return ""
    parts = re.split(r"\n\s*\n", text, maxsplit=1)
    return " ".join(parts[0].split())


def class_summary(model_cls: type) -> str:
    raw = model_cls.__dict__.get("__doc__") or ""
    return first_paragraph(raw)


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


def _md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


# 配置分类 → 输出子目录（按类型组织 API 文档）
CATEGORY_DIRS = {
    "任务配置": "task",
    "服务规格": "spec",
    "处理器": "processor",
    "保存格式": "format",
    "自动调优": "tuning",
    "嵌套配置": "nested",
}


def _doc_path(record: "ModelRecord") -> str:
    """记录对应的输出相对路径（按分类子目录组织）。"""
    directory = CATEGORY_DIRS.get(record.category, "other")
    return f"{directory}/{record.filename}"


def _rel_link(source: "ModelRecord", target: "ModelRecord") -> str:
    """从 source 页面所在目录到 target 文档页面的相对链接。"""
    return posixpath.relpath(_doc_path(target), start=CATEGORY_DIRS.get(source.category, "other"))


def _rel_link_to(source: "ModelRecord", doc_rel: str) -> str:
    """从 source 页面所在目录到指定输出相对路径（如 task/modelslim_v1.md）的相对链接。"""
    return posixpath.relpath(doc_rel, start=CATEGORY_DIRS.get(source.category, "other"))


def field_yaml_path(yaml_path: str, name: str) -> str:
    """字段路径：只显示相对当前配置的字段名，不叠加整条 YAML 路径。

    配置自身在 YAML 中的位置由「配置概述」或展开小节标题给出。
    """
    return name


def _full_field_path(record: "ModelRecord", name: str) -> str:
    """字段的完整 YAML 路径（用于分派判断，不直接展示）。"""
    if record.yaml_path in {"", "(根)"}:
        return name
    return f"{record.yaml_path}.{name}"


def _anchor(text: str) -> str:
    slug = text.lower()
    slug = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", slug).strip("-")
    return slug


def _jump_link(label: str, anchor: str) -> str:
    """页内跳转用 HTML 锚点：``<a href=\"#anchor\">§label</a>``。"""
    return f'<a href="#{anchor}">§{label}</a>'


def is_expandable_nested(record: "ModelRecord") -> bool:
    """嵌套配置类别（无独立页面、展开进上级文档的内部配置）是否应展开。"""
    return record.category == "嵌套配置"


def _collect_expanded(
    record: "ModelRecord",
    catalog: Dict[str, "ModelRecord"],
    is_expandable: Any,
) -> List[Tuple[str, str]]:
    """收集需要展开进本页的嵌套配置 (slug, 上下文 YAML 路径)。

    BFS 含多级嵌套，按出现顺序去重；路径取自引用方 nested_refs，保证展开内容
    使用该页上下文中的真实字段路径（例如 Metadata 在 binary_fallback 下是
    ``strategy.template.metadata``）。
    """
    ordered: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    queue: List[Tuple[str, str]] = [(slug, path) for path, slug, _ in record.nested_refs]
    while queue:
        slug, path = queue.pop(0)
        if slug in seen:
            continue
        target = catalog.get(slug)
        if target is None or not is_expandable(target):
            continue
        seen.add(slug)
        ordered.append((slug, path))
        queue.extend((s, p) for p, s, _ in target.nested_refs)
    return ordered


def _dispatch_slugs(record: "ModelRecord", field_path: str) -> List[str]:
    """字段在 type/mode 分派下分派到的配置 slug 列表。

    优先用 ``dispatch_bases``（含 Union 分派如 select_best），否则退到
    ``type 分派`` 关系（处理器 / 保存格式 / 策略 / 评估服务）。
    """
    targets: List[str] = []
    if field_path in record.dispatch_bases or f"{field_path}[]" in record.dispatch_bases:
        for fp, slug, relation in record.nested_refs:
            if fp in (field_path, f"{field_path}[]"):
                targets.append(slug)
        return targets
    for fp, slug, relation in record.nested_refs:
        if relation != "type 分派":
            continue
        if fp in (field_path, f"{field_path}[]"):
            targets.append(slug)
    return targets


def _field_ref_text(
    page_record: "ModelRecord",
    item: FieldRecord,
    catalog: Dict[str, "ModelRecord"],
    anchors: Dict[str, Tuple[str, str]],
    dispatch_slugs: Sequence[str],
    link_overrides: Optional[Dict[str, Tuple[str, str]]] = None,
    field_path: Optional[str] = None,
) -> str:
    """参数列表「引用配置」列文案：展开进本页的走页内锚点，公开页面走相对链接。

    ``page_record`` 是当前渲染页面（展开小节时为上级页面），用于计算跨目录相对路径。
    ``link_overrides`` 把未独立成页的配置（如已合并进 task 页的 spec）重定向到
    合并页的锚点，映射为 slug → (输出相对路径, #锚点)。
    ``field_path`` 是当前字段的完整 YAML 路径：type 分派字段优先指向页内的
    基础类块锚点（``anchors[field_path]``）。
    """
    if dispatch_slugs:
        if field_path:
            for key in (field_path, f"{field_path}[]"):
                if key in anchors:
                    label, anchor = anchors[key]
                    return f"本页 {_jump_link(label, anchor)}"
        in_page = [s for s in dispatch_slugs if s in anchors]
        external = [s for s in dispatch_slugs if s not in anchors]
        cells = [f"本页 {_jump_link(*anchors[s])}" for s in in_page]
        if external and len(external) <= 4:
            for s in external:
                target = catalog.get(s)
                if target:
                    if link_overrides and s in link_overrides:
                        doc_rel, anchor = link_overrides[s]
                        cells.append(f"《[{target.title}]({_rel_link_to(page_record, doc_rel)}{anchor})》")
                    else:
                        cells.append(f"《[{target.title}]({_rel_link(page_record, target)})》")
                else:
                    cells.append(s)
        elif external:
            cells.append("按 `type` 分派，见对应配置文档")
        return "、".join(cells) if cells else "无"
    if item.nested_models:
        links = []
        for nested_key in item.nested_models:
            if nested_key in anchors:
                label, anchor = anchors[nested_key]
                links.append(f"本页 {_jump_link(label, anchor)}")
                continue
            target = catalog.get(nested_key)
            if target:
                if link_overrides and nested_key in link_overrides:
                    doc_rel, anchor = link_overrides[nested_key]
                    links.append(f"《[{target.title}]({_rel_link_to(page_record, doc_rel)}{anchor})》")
                else:
                    links.append(f"《[{target.title}]({_rel_link(page_record, target)})》")
            else:
                links.append(nested_key)
        return "、".join(links)
    return "无"


def _param_rows(
    record: "ModelRecord",
    catalog: Dict[str, "ModelRecord"],
    anchors: Dict[str, Tuple[int, str]],
    page_record: Optional["ModelRecord"] = None,
    link_overrides: Optional[Dict[str, Tuple[str, str]]] = None,
) -> List[str]:
    """渲染一个配置类的参数表；``page_record`` 用于计算跨目录链接（缺省为 record）。"""
    if page_record is None:
        page_record = record
    dispatch_paths = set(record.dispatch_bases)
    dispatch_paths.update(field_path for field_path, _, relation in record.nested_refs if relation == "type 分派")
    rows: List[str] = []
    for item in record.fields:
        req = "必选" if item.required else "可选"
        default = "无" if item.required and item.default == "无" else item.default
        item_path = field_yaml_path(record.yaml_path, item.name)
        full_item_path = _full_field_path(record, item.name)
        dispatch_slugs: List[str] = []
        if full_item_path in dispatch_paths or f"{full_item_path}[]" in dispatch_paths:
            dispatch_slugs = _dispatch_slugs(record, full_item_path)
        ref = _field_ref_text(
            page_record,
            item,
            catalog,
            anchors,
            dispatch_slugs,
            link_overrides=link_overrides,
            field_path=full_item_path,
        )
        path = f"`{item_path}`"
        # apiversion 的默认值是代码占位，取值是固定协议名；文档给出真实可取值避免误导。
        if item.name == "apiversion":
            if "Unknown" in default:
                default = "Unknown（代码占位；YAML 中须按任务类型显式指定）"
            if item.constraint in ("—", ""):
                item.constraint = (
                    "`modelslim_v1`、`multimodal_vlm_modelslim_v1`、`multimodal_sd_modelslim_v1`、`modelslim_convert`"
                )
        rows.append(
            f"| {path} | `{item.type_name}` | {req} | {_md_cell(default)} | "
            f"{_md_cell(item.constraint)} | {_md_cell(item.description)} | {ref} |"
        )
    return rows


def _render_param_table(rows: Sequence[str]) -> str:
    if not rows:
        return "无对外字段。"
    header = (
        "| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |\n"
        "|----------|------|-----------|--------|----------------|------|----------|\n"
    )
    return header + "\n".join(rows)


def _constraint_text(constraints: Sequence[str]) -> str:
    return "\n".join(f"- {c}" for c in constraints) if constraints else "- 无。"


def _field_name_from_path(field_path: str) -> str:
    """从完整字段路径（如 spec.process[].train_config.select_best）取字段名。"""
    name = field_path.rsplit(".", 1)[-1]
    return name.rstrip("[]")


def _dispatch_value(target: "ModelRecord", discriminator: Optional[str]) -> str:
    """分派字段下派生配置的判别值（type/mode 字面量）。"""
    if target.type_tag:
        return target.type_tag
    if discriminator:
        for f in target.fields:
            if f.name == discriminator and f.constraint:
                return f.constraint.strip("`")
    return ""


def _find_base_record(catalog: Dict[str, "ModelRecord"], base_name: str) -> Optional["ModelRecord"]:
    for r in catalog.values():
        if r.class_name == base_name:
            return r
    return None


def _dispatch_derived_items(
    record: "ModelRecord",
    refs: Sequence[Tuple[str, str]],
    anchors: Dict[str, Tuple[str, str]],
    catalog: Dict[str, "ModelRecord"],
    discriminator: Optional[str],
) -> List[str]:
    """基础类块下「派生类」列表项：类名 + 判别值 + 一句说明 + 链接。"""
    items: List[str] = []
    for slug, _rel in refs:
        target = catalog.get(slug)
        if target is None:
            items.append(f"- `{slug}`")
            continue
        value = _dispatch_value(target, discriminator)
        desc = target.summary
        if slug in anchors:
            label, anchor = anchors[slug]
            link = f"本页 {_jump_link(label, anchor)}"
        else:
            link = f"《[{target.title}]({_rel_link(record, target)})》"
        value_txt = f"（`{discriminator or 'type'}: {value}`）" if value else ""
        items.append(f"- `{target.class_name}`{value_txt} — {desc} {link}")
    return items


def _render_param_blocks(
    record: "ModelRecord",
    catalog: Dict[str, "ModelRecord"],
    anchors: Dict[str, Tuple[str, str]],
    link_overrides: Optional[Dict[str, Tuple[str, str]]],
    is_expandable: Any,
    expand_nested: bool,
) -> str:
    """按类名组织「参数列表」：根配置 + 递归展开嵌套/分派配置。

    - 普通嵌套配置：``### 类名`` 平铺块；
    - type 分派字段：``### 基础类名`` 块 + 「派生类」列表 + 各派生类 ``####`` 子块。
    不再在子标题里显示 ``（ctx_path）`` 路径后缀。

    两阶段实现：先遍历分配全部页内锚点，再渲染块，保证参数表「引用配置」列
    能指向后续才出现的块锚点。
    """
    counter = [0]
    seen: Set[str] = set()
    plan: List[Tuple[str, Any]] = []
    rendered_bases: Dict[str, Tuple[str, str]] = {}

    def assign(key: str) -> Tuple[str, str]:
        counter[0] += 1
        label = f"2.{counter[0]}"
        anchor = _anchor(f"{label} {key}")
        anchors[key] = (label, anchor)
        return label, anchor

    def build(target: "ModelRecord", page: "ModelRecord") -> None:
        groups: Dict[str, List[Tuple[str, str]]] = {}
        for fp, slug, rel in target.nested_refs:
            groups.setdefault(fp, []).append((slug, rel))
        for fp, refs in groups.items():
            rels = {rel for _, rel in refs}
            is_dispatch = "type 分派" in rels or len(refs) > 1
            if is_dispatch:
                first = catalog.get(refs[0][0])
                base_name = target.dispatch_bases.get(fp) or (first.class_name if first else fp)
                base_record = _find_base_record(catalog, base_name)
                derived_refs = refs
                if base_record:
                    derived_refs = [r for r in refs if r[0] != base_record.slug]
                if base_name in rendered_bases:
                    # 同一页面同基础类只渲染一次：后续分派字段别名到已渲染块锚点。
                    blabel, banchor = rendered_bases[base_name]
                    anchors[fp] = (blabel, banchor)
                    if base_record:
                        anchors[base_record.slug] = (blabel, banchor)
                    continue
                base_key = base_record.slug if base_record else base_name
                label, anchor = assign(base_key)
                rendered_bases[base_name] = (label, anchor)
                anchors[fp] = (label, anchor)
                if base_record:
                    anchors[base_record.slug] = (label, anchor)
                field_name = _field_name_from_path(fp)
                discriminator = next(
                    (f.discriminator for f in target.fields if f.name == field_name),
                    None,
                )
                plan.append(("base", (base_name, base_record, label, anchor, discriminator, derived_refs)))
                for slug, _rel in derived_refs:
                    child = catalog.get(slug)
                    if child is None or slug in seen or not is_expandable(child):
                        continue
                    seen.add(slug)
                    clabel, canchor = assign(slug)
                    plan.append(("sub", (child, clabel, canchor, fp)))
                    build(replace(child, yaml_path=fp), page)
            else:
                slug = refs[0][0]
                child = catalog.get(slug)
                if child is None or slug in seen or not is_expandable(child):
                    continue
                seen.add(slug)
                clabel, canchor = assign(slug)
                plan.append(("config", (child, clabel, canchor, fp)))
                build(replace(child, yaml_path=fp), page)

    rlabel, ranchor = assign(record.slug)
    plan.append(("root", (record, rlabel, ranchor)))
    if expand_nested:
        build(record, record)

    blocks: List[str] = []
    for kind, payload in plan:
        if kind == "root":
            child, clabel, canchor = payload
            rows = _param_rows(child, catalog, anchors, page_record=record, link_overrides=link_overrides)
            blocks.append(
                f'<h3 id="{canchor}">{clabel} {child.class_name}</h3>\n\n'
                f"{_render_param_table(rows)}\n\n"
                f"**配置约束**\n\n{_constraint_text(child.constraints)}"
            )
        elif kind == "config":
            child, clabel, canchor, fp = payload
            ctx = replace(child, yaml_path=fp)
            rows = _param_rows(ctx, catalog, anchors, page_record=record, link_overrides=link_overrides)
            summary = ctx.class_summary
            summary_text = f"{summary}\n\n" if summary else ""
            blocks.append(
                f'<h3 id="{canchor}">{clabel} {ctx.class_name}</h3>\n\n'
                f"{summary_text}"
                f"{_render_param_table(rows)}\n\n"
                f"**配置约束**\n\n{_constraint_text(ctx.constraints)}"
            )
        elif kind == "sub":
            child, clabel, canchor, fp = payload
            ctx = replace(child, yaml_path=fp)
            rows = _param_rows(ctx, catalog, anchors, page_record=record, link_overrides=link_overrides)
            summary = ctx.class_summary
            summary_text = f"{summary}\n\n" if summary else ""
            blocks.append(
                f'<h4 id="{canchor}">{clabel} {ctx.class_name}</h4>\n\n'
                f"{summary_text}"
                f"{_render_param_table(rows)}\n\n"
                f"**配置约束**\n\n{_constraint_text(ctx.constraints)}"
            )
        else:  # base
            base_name, base_record, clabel, anchor, discriminator, derived_refs = payload
            derived_items = _dispatch_derived_items(record, derived_refs, anchors, catalog, discriminator)
            base_params = ""
            if base_record:
                base_params = (
                    f"{_render_param_table(_param_rows(base_record, catalog, anchors, page_record=record, link_overrides=link_overrides))}\n\n"
                    f"**配置约束**\n\n{_constraint_text(base_record.constraints)}\n\n"
                )
            header = base_name
            if discriminator:
                header += f"（按 `{discriminator}` 分派）"
            blocks.append(
                f'<h3 id="{anchor}">{clabel} {header}</h3>\n\n{base_params}**派生类**\n\n' + "\n".join(derived_items)
            )
    return "\n\n".join(blocks)


def render_markdown(
    record: ModelRecord,
    catalog: Dict[str, "ModelRecord"],
    expand_nested: bool = False,
    is_expandable: Any = None,
    link_depth: int = 4,
    link_overrides: Optional[Dict[str, Tuple[str, str]]] = None,
) -> str:
    source_name = Path(record.source_rel).name
    source_link = "/".join([".."] * link_depth + [record.source_rel.replace("\\", "/")])
    if is_expandable is None:
        is_expandable = is_expandable_nested

    anchors: Dict[str, Tuple[str, str]] = {}
    param_section = _render_param_blocks(
        record,
        catalog,
        anchors,
        link_overrides,
        is_expandable,
        expand_nested,
    )

    sections: List[Tuple[str, str, str]] = [
        (
            "1",
            "配置概述",
            f"{record.summary}\n\n| 项目 | 内容 |\n|------|------|\n"
            f"| 配置类 | `{record.class_name}` |\n"
            f"| 源码 | [{source_name}]({source_link}) |",
        ),
        ("2", "参数列表", param_section),
    ]
    next_num = 3
    if (record.example_yaml or "").strip():
        sections.append((str(next_num), "完整配置参考", f"```yaml\n{record.example_yaml.rstrip()}\n```"))

    body = "\n\n".join(f"## {num}. {title}\n\n{content}" for num, title, content in sections)
    return f"<!-- {GENERATED_MARKER} ; class: {record.qualname} -->\n# {record.title}\n\n{body}\n"
