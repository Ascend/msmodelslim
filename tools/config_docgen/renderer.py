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

将配置记录渲染为 Markdown 文档。

Schema 和注解抽取位于 schema.py；本模块只负责页面组织与链接。
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from tools.config_docgen.catalog import (
    BASE_CLASS_REPRESENTATIVE_TYPE_TAGS,
    CATEGORY_DIRS,
    SOURCE_REFERENCE_PATHS,
)
from tools.config_docgen.records import FieldRecord, ModelRecord, ScenarioRecord

GENERATED_MARKER = "generated-by: tools.config_docgen"
LEGACY_GENERATED_MARKERS = (
    "generated-by: tools/config_docgen/main.py",
    "generated-by: tools/config_docgen/generate.py",
    "generated-by: skills/docs-management/scripts/gen_quant_config_docs.py",
)
GENERATED_NOTICE = (
    "> **自动生成文档，请勿手工修改。**\n"
    ">\n"
    "> 本页由 `tools.config_docgen` 根据源码注解生成。\n"
    "> 修改字段描述、类概述或校验逻辑后，请重新生成并提交生成稿。"
)


def _md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def _doc_path(record: "ModelRecord") -> str:
    """记录对应的输出相对路径（按分类子目录组织）。"""
    directory = CATEGORY_DIRS.get(record.category, "other")
    return f"{directory}/{record.filename}"


def _rel_link(source: "ModelRecord", target: "ModelRecord") -> str:
    """从 source 页面所在目录到 target 文档页面的相对链接。"""
    return posixpath.relpath(_doc_path(target), start=CATEGORY_DIRS.get(source.category, "other"))


def _rel_link_to(source: "ModelRecord", doc_rel: str) -> str:
    """从 source 页面所在目录到指定输出相对路径（如 quant/modelslim_v1.md）的相对链接。"""
    return posixpath.relpath(doc_rel, start=CATEGORY_DIRS.get(source.category, "other"))


def _source_link(source_rel: str, link_depth: int) -> str:
    """从生成文档目录到仓库内源码文件的相对链接。"""
    return "/".join([".."] * link_depth + [source_rel.replace("\\", "/")])


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


_CLASS_CONFIG_REFERENCE_RE = re.compile(r"(?<![\[>])《([A-Za-z_][A-Za-z0-9_]*) 配置说明》")


def _link_class_references(
    text: str,
    catalog: Dict[str, "ModelRecord"],
    anchors: Dict[str, Tuple[str, str]],
    page_record: "ModelRecord",
    link_overrides: Optional[Dict[str, Tuple[str, str]]] = None,
    link_depth: int = 5,
) -> str:
    """把注解中的《ClassName 配置说明》解析为当前锚点对应的可点击链接。"""
    if "配置说明》" not in text:
        return text

    def replace_reference(match: re.Match[str]) -> str:
        class_name = match.group(1)
        targets = [record for record in catalog.values() if record.class_name == class_name]
        if not targets:
            targets = [record for record in catalog.values() if record.title == class_name]
        if len(targets) != 1:
            source_rel = SOURCE_REFERENCE_PATHS.get(class_name)
            if source_rel:
                href = _source_link(source_rel, link_depth)
                return f'<a href="{href}">{match.group(0)}</a>'
            return match.group(0)
        target = targets[0]
        anchor_info = anchors.get(target.slug) or anchors.get(target.class_name)
        if anchor_info:
            href = f"#{anchor_info[1]}"
        elif link_overrides and target.slug in link_overrides:
            doc_rel, anchor = link_overrides[target.slug]
            href = f"{_rel_link_to(page_record, doc_rel)}{anchor}"
        else:
            href = _rel_link(page_record, target)
        return f'<a href="{href}">{match.group(0)}</a>'

    return _CLASS_CONFIG_REFERENCE_RE.sub(replace_reference, text)


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
    link_depth: int = 5,
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
                source_rel = SOURCE_REFERENCE_PATHS.get(nested_key)
                if source_rel:
                    links.append(f"《[{nested_key}]({_source_link(source_rel, link_depth)})》")
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
    link_depth: int = 5,
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
            link_depth=link_depth,
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
        description = _link_class_references(
            item.description,
            catalog,
            anchors,
            page_record,
            link_overrides=link_overrides,
            link_depth=link_depth,
        )
        rows.append(
            f"| {path} | `{item.type_name}` | {req} | {_md_cell(default)} | "
            f"{_md_cell(item.constraint)} | {_md_cell(description)} | {ref} |"
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


def _notes_text(notes: Sequence[str]) -> str:
    return "\n".join(f"- {note}" for note in notes)


def _class_description_text(description: Sequence[str]) -> str:
    """将类 docstring「说明」条目渲染为紧邻概述的项目列表。"""
    return "\n".join(f"- {item}" for item in description)


def _scenarios_text(scenarios: Sequence[ScenarioRecord]) -> str:
    blocks: List[str] = []
    for scenario in scenarios:
        block = f"### {scenario.title}"
        if scenario.description:
            block += f"\n\n{scenario.description}"
        block += f"\n\n```yaml\n{scenario.yaml}\n```"
        blocks.append(block)
    return "\n\n".join(blocks)


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


def _dispatch_derived_table(
    record: "ModelRecord",
    refs: Sequence[Tuple[str, str]],
    anchors: Dict[str, Tuple[str, str]],
    catalog: Dict[str, "ModelRecord"],
    discriminator: Optional[str],
) -> str:
    """基础类块下「派生类」表格：类名 + 判别值 + 一句说明 + 链接。"""
    has_type_tags = any((catalog.get(slug) and catalog[slug].type_tag) for slug, _rel in refs)
    dispatch_column = discriminator or ("type" if has_type_tags else None)
    if dispatch_column:
        header = f"| 配置类 | `{dispatch_column}` | 说明 | 文档 |\n|--------|----------|------|------|\n"
    else:
        header = "| 配置类 | 说明 | 文档 |\n|--------|------|------|\n"

    rows: List[str] = []
    for slug, _rel in refs:
        target = catalog.get(slug)
        if target is None:
            if dispatch_column:
                rows.append(f"| `{slug}` | — | — | — |")
            else:
                rows.append(f"| `{slug}` | — | — |")
            continue
        if slug in anchors:
            label, anchor = anchors[slug]
            doc_link = f"本页 {_jump_link(label, anchor)}"
        else:
            doc_link = f"《[{target.title}]({_rel_link(record, target)})》"
        if dispatch_column:
            value = _dispatch_value(target, discriminator) or "—"
            rows.append(f"| `{target.class_name}` | `{value}` | {_md_cell(target.summary)} | {doc_link} |")
        else:
            rows.append(f"| `{target.class_name}` | {_md_cell(target.summary)} | {doc_link} |")
    return header + "\n".join(rows)


def _representative_record_for_base(
    catalog: Dict[str, "ModelRecord"],
    base_name: str,
) -> Optional["ModelRecord"]:
    """返回协议基类在公开文档中的代表页。"""
    type_tag = BASE_CLASS_REPRESENTATIVE_TYPE_TAGS.get(base_name)
    if not type_tag:
        return None
    for target in catalog.values():
        if target.type_tag == type_tag:
            return target
    return None


def _base_dispatch_header(
    record: "ModelRecord",
    base_name: str,
    catalog: Dict[str, "ModelRecord"],
    discriminator: Optional[str],
) -> str:
    """分派基类标题：协议基类链接到其公开规范配置页。"""
    representative = _representative_record_for_base(catalog, base_name)
    if representative:
        header = f'《<a href="{_rel_link(record, representative)}">{base_name}</a>》'
    else:
        header = base_name
    if discriminator:
        header += f"（按 <code>{discriminator}</code> 分派）"
    return header


def _render_param_blocks(
    record: "ModelRecord",
    catalog: Dict[str, "ModelRecord"],
    anchors: Dict[str, Tuple[str, str]],
    link_overrides: Optional[Dict[str, Tuple[str, str]]],
    is_expandable: Any,
    expand_nested: bool,
    link_depth: int = 5,
) -> str:
    """按类名组织「参数列表」：根配置 + 递归展开嵌套/分派配置。

    - 普通嵌套配置：``### 类名`` 平铺块；
    - type 分派字段：``### 基础类名`` 块 + 「派生类」表格 + 各派生类 ``####`` 子块。
    不再在子标题里显示 ``（ctx_path）`` 路径后缀。

    两阶段实现：先遍历分配全部页内锚点，再渲染块，保证参数表「引用配置」列
    能指向后续才出现的块锚点。
    """
    counter = [0]
    child_counters: Dict[str, int] = {}
    seen: Set[str] = set()
    plan: List[Tuple[str, Any]] = []
    rendered_bases: Dict[str, Tuple[str, str]] = {}

    def assign_top(key: str) -> Tuple[str, str]:
        counter[0] += 1
        label = f"2.{counter[0]}"
        anchor = _anchor(f"{label} {key}")
        anchors[key] = (label, anchor)
        return label, anchor

    def assign_child(parent_label: str, key: str) -> Tuple[str, str]:
        """二级及以下配置统一作为 h4 子块，编号延续所属顶层配置。"""
        child_counters[parent_label] = child_counters.get(parent_label, 0) + 1
        label = f"{parent_label}.{child_counters[parent_label]}"
        anchor = _anchor(f"{label} {key}")
        anchors[key] = (label, anchor)
        return label, anchor

    def build(target: "ModelRecord", page: "ModelRecord", top_label: Optional[str] = None) -> None:
        """展开嵌套配置；根配置的直接子块用 h3，其余后代统一用 h4。"""
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
                if top_label is None:
                    label, anchor = assign_top(base_key)
                    base_kind = "base"
                    child_scope = label
                else:
                    label, anchor = assign_child(top_label, base_key)
                    base_kind = "subbase"
                    child_scope = top_label
                rendered_bases[base_name] = (label, anchor)
                anchors[fp] = (label, anchor)
                if base_record:
                    anchors[base_record.slug] = (label, anchor)
                field_name = _field_name_from_path(fp)
                discriminator = next(
                    (f.discriminator for f in target.fields if f.name == field_name),
                    None,
                )
                plan.append((base_kind, (base_name, base_record, label, anchor, discriminator, derived_refs)))
                for slug, _rel in derived_refs:
                    child = catalog.get(slug)
                    if child is None or slug in seen or not is_expandable(child):
                        continue
                    seen.add(slug)
                    clabel, canchor = assign_child(child_scope, slug)
                    plan.append(("sub", (child, clabel, canchor, fp)))
                    build(replace(child, yaml_path=fp), page, child_scope)
            else:
                slug = refs[0][0]
                child = catalog.get(slug)
                if child is None or slug in seen or not is_expandable(child):
                    continue
                seen.add(slug)
                if top_label is None:
                    clabel, canchor = assign_top(slug)
                    child_scope = clabel
                    kind = "config"
                else:
                    clabel, canchor = assign_child(top_label, slug)
                    child_scope = top_label
                    kind = "sub"
                plan.append((kind, (child, clabel, canchor, fp)))
                build(replace(child, yaml_path=fp), page, child_scope)

    rlabel, ranchor = assign_top(record.slug)
    plan.append(("root", (record, rlabel, ranchor)))
    if expand_nested:
        build(record, record)

    blocks: List[str] = []
    for kind, payload in plan:
        if kind == "root":
            child, clabel, canchor = payload
            rows = _param_rows(
                child,
                catalog,
                anchors,
                page_record=record,
                link_overrides=link_overrides,
                link_depth=link_depth,
            )
            blocks.append(
                f'<h3 id="{canchor}">{clabel} {child.class_name}</h3>\n\n'
                f"{_render_param_table(rows)}\n\n"
                f"**配置约束**\n\n{_constraint_text(child.constraints)}"
            )
        elif kind == "config":
            child, clabel, canchor, fp = payload
            ctx = replace(child, yaml_path=fp)
            rows = _param_rows(
                ctx,
                catalog,
                anchors,
                page_record=record,
                link_overrides=link_overrides,
                link_depth=link_depth,
            )
            summary = ctx.class_summary
            summary_text = f"{summary}\n\n" if summary else ""
            description = _class_description_text(ctx.class_description)
            description_text = f"{description}\n\n" if description else ""
            blocks.append(
                f'<h3 id="{canchor}">{clabel} {ctx.class_name}</h3>\n\n'
                f"{summary_text}{description_text}"
                f"{_render_param_table(rows)}\n\n"
                f"**配置约束**\n\n{_constraint_text(ctx.constraints)}"
            )
        elif kind == "sub":
            child, clabel, canchor, fp = payload
            ctx = replace(child, yaml_path=fp)
            rows = _param_rows(
                ctx,
                catalog,
                anchors,
                page_record=record,
                link_overrides=link_overrides,
                link_depth=link_depth,
            )
            summary = ctx.class_summary
            summary_text = f"{summary}\n\n" if summary else ""
            description = _class_description_text(ctx.class_description)
            description_text = f"{description}\n\n" if description else ""
            blocks.append(
                f'<h4 id="{canchor}">{clabel} {ctx.class_name}</h4>\n\n'
                f"{summary_text}{description_text}"
                f"{_render_param_table(rows)}\n\n"
                f"**配置约束**\n\n{_constraint_text(ctx.constraints)}"
            )
        else:  # base / subbase
            base_name, base_record, clabel, anchor, discriminator, derived_refs = payload
            derived_table = _dispatch_derived_table(record, derived_refs, anchors, catalog, discriminator)
            base_summary = base_record.class_summary if base_record else ""
            base_summary_text = f"{base_summary}\n\n" if base_summary else ""
            base_description = _class_description_text(base_record.class_description) if base_record else ""
            base_description_text = f"{base_description}\n\n" if base_description else ""
            base_params = ""
            if base_record:
                base_params = (
                    f"{_render_param_table(_param_rows(base_record, catalog, anchors, page_record=record, link_overrides=link_overrides, link_depth=link_depth))}\n\n"
                    f"**配置约束**\n\n{_constraint_text(base_record.constraints)}\n\n"
                )
            header = _base_dispatch_header(record, base_name, catalog, discriminator)
            heading = "h3" if kind == "base" else "h4"
            blocks.append(
                f'<{heading} id="{anchor}">{clabel} {header}</{heading}>\n\n'
                f"{base_summary_text}{base_description_text}{base_params}**派生类**\n\n{derived_table}"
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
        link_depth=link_depth,
    )

    class_description = _class_description_text(record.class_description)
    overview = record.summary
    if class_description:
        overview += f"\n\n{class_description}"
    sections: List[Tuple[str, str, str]] = [
        (
            "1",
            "配置概述",
            f"{overview}\n\n| 项目 | 内容 |\n|------|------|\n"
            f"| 配置类 | `{record.class_name}` |\n"
            f"| 源码 | [{source_name}]({source_link}) |",
        ),
        ("2", "参数列表", param_section),
    ]
    next_num = 3
    if (record.example_yaml or "").strip():
        sections.append((str(next_num), "完整配置参考", f"```yaml\n{record.example_yaml.rstrip()}\n```"))
        next_num += 1
    if record.notes:
        sections.append((str(next_num), "注意事项", _notes_text(record.notes)))
        next_num += 1
    if record.scenarios:
        sections.append((str(next_num), "配置场景示例", _scenarios_text(record.scenarios)))

    body = "\n\n".join(f"## {num}. {title}\n\n{content}" for num, title, content in sections)
    return f"<!-- {GENERATED_MARKER} ; class: {record.qualname} -->\n{GENERATED_NOTICE}\n\n# {record.title}\n\n{body}\n"
