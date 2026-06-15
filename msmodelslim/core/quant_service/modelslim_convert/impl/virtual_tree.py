#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
VirtualModelTreeBuilder（convert_design.md §8）。

根据 ``module_rules`` 与预处理后的 ``TensorCatalog`` 构建懒加载虚拟 ``nn.Module`` 树：
  1. 收集 module_rules 命中的 binding key；
  2. ``CheckpointReader.enrich_catalog`` 仅读取相关 shard 的 dtype/shape；
  3. 为每个 module_path 创建 ``ModelFreeLinear`` / ``PassthroughModule`` 并绑定 ``TensorRef``。
"""

from __future__ import annotations

import fnmatch
from collections import defaultdict

from torch import nn
from tqdm import tqdm

from msmodelslim.core.quant_service.modelslim_convert.impl.source_ir import bind_tensors_for_module, infer_source_ir
from msmodelslim.core.quant_service.modelslim_convert.virtual_module import (
    ModelFreeModule,
    PassthroughModule,
    create_model_free_module,
    set_submodule_by_path,
)
from msmodelslim.core.convert.catalog import TensorCatalog
from msmodelslim.core.convert.config import ConvertConfig, ModuleRule
from msmodelslim.ir.kernels import WEIGHT_SCALE_INV_SUFFIX
from msmodelslim.core.convert.protocol import ConvertContext, IVirtualModelTreeBuilder
from msmodelslim.core.convert.types import IRKind, SourceIR, TensorRef
from msmodelslim.infra.io.checkpoint_reader import CheckpointReader
from msmodelslim.utils.logging import get_logger

logger = get_logger()


def collect_bound_catalog_keys(tree: nn.Module) -> set[str]:
    """All checkpoint keys referenced by ``tensor_bindings`` on the virtual tree."""
    handled: set[str] = set()
    for mod in tree.modules():
        if isinstance(mod, ModelFreeModule):
            for ref in mod.tensor_bindings.values():
                handled.add(ref.key)
                fused = (ref.meta or {}).get("fused_from")
                if fused:
                    handled.add(fused)
    return handled


def _candidate_module_paths(catalog_keys: set[str], rule_match: str) -> list[str]:
    """
    从 catalog key 反推 module_path，再用 fnmatch 过滤。

    处理 ``*.weight`` 与 ``*.weight_packed`` 结尾的 key。后者用于 INT4 packed
    权重，转换时再通过 ``weight_scale`` / ``weight_shape`` 反量化到 FLOAT。
    """
    seen: set[str] = set()
    out: list[str] = []
    for key in catalog_keys:
        if key.endswith(WEIGHT_SCALE_INV_SUFFIX):
            continue
        if key.endswith(".weight"):
            path = key[: -len(".weight")]
        elif key.endswith(".weight_packed"):
            path = key[: -len(".weight_packed")]
        else:
            continue
        if path in seen:
            continue
        if fnmatch.fnmatch(path, rule_match):
            seen.add(path)
            out.append(path)
    return out


def _collect_binding_keys(config: ConvertConfig, catalog_keys: set[str]) -> set[str]:
    """汇总 module_rules 需要 enrich 的 checkpoint tensor key。"""
    keys: set[str] = set()
    for rule in config.module_rules:
        for path in _candidate_module_paths(catalog_keys, rule.match):
            weight_key = f"{path}.weight"
            if weight_key in catalog_keys:
                keys.add(weight_key)
            scale_key = path + WEIGHT_SCALE_INV_SUFFIX
            if scale_key in catalog_keys:
                keys.add(scale_key)
            for suffix in (".weight_packed", ".weight_scale", ".weight_shape", ".bias"):
                key = f"{path}{suffix}"
                if key in catalog_keys:
                    keys.add(key)
            for _, pat in rule.tensor_map.items():
                resolved = pat.replace("{module}", path)
                if resolved in catalog_keys:
                    keys.add(resolved)
    return keys


def _resolve_bindings(
    module_path: str,
    rule: ModuleRule,
    catalog: TensorCatalog,
    catalog_keys: set[str],
) -> dict[str, TensorRef]:
    bindings = bind_tensors_for_module(module_path, rule, catalog_keys)
    weight_key = f"{module_path}.weight"
    if not bindings and weight_key in catalog_keys:
        e = catalog.get(weight_key)
        bindings["weight"] = TensorRef(
            "weight",
            weight_key,
            e.shard,
            e.dtype,
            e.shape,
            meta=dict(e.meta),
        )
        inv_key = module_path + WEIGHT_SCALE_INV_SUFFIX
        if inv_key in catalog_keys:
            ie = catalog.get(inv_key)
            bindings["weight_scale_inv"] = TensorRef(
                "weight_scale_inv",
                inv_key,
                ie.shard,
                ie.dtype,
                ie.shape,
            )
    else:
        bindings = {k: _fill_ref(v, catalog) for k, v in bindings.items()}
    if not bindings:
        packed_key = f"{module_path}.weight_packed"
        if packed_key in catalog_keys:
            for logical, suffix in (
                ("weight_packed", ".weight_packed"),
                ("weight_scale", ".weight_scale"),
                ("weight_shape", ".weight_shape"),
                ("bias", ".bias"),
            ):
                key = f"{module_path}{suffix}"
                if key in catalog_keys:
                    e = catalog.get(key)
                    bindings[logical] = TensorRef(
                        logical,
                        key,
                        e.shard,
                        e.dtype,
                        e.shape,
                        meta=dict(e.meta),
                    )
    return bindings


class VirtualModelTreeBuilder(IVirtualModelTreeBuilder):
    def build(self, context: ConvertContext, catalog: TensorCatalog) -> nn.Module:
        root = nn.Module()
        catalog_keys = set(catalog.keys())

        reader = context.reader
        if isinstance(reader, CheckpointReader):
            needed = _collect_binding_keys(context.config, catalog_keys)
            for key in catalog_keys:
                entry = catalog.get(key)
                if entry and entry.meta.get("fused_from"):
                    needed.add(entry.meta["fused_from"])
            with tqdm(total=1, desc="enrich catalog metadata", leave=False) as pbar:
                reader.enrich_catalog(catalog, keys=needed)
                pbar.update(1)
            logger.info("Enriched metadata for %d keys", len(needed))

        matched_paths: set[str] = set()
        for rule in sorted(context.config.module_rules, key=lambda r: r.convert):
            _install_rule_modules(root, catalog, catalog_keys, rule, matched_paths, context)

        handled = collect_bound_catalog_keys(root)
        _attach_preserve_all_catalog(root, catalog, catalog_keys, handled, context, matched_paths)
        return root


def _install_rule_modules(
    root: nn.Module,
    catalog: TensorCatalog,
    catalog_keys: set[str],
    rule: ModuleRule,
    matched_paths: set[str],
    context: ConvertContext,
) -> None:
    candidates = _candidate_module_paths(catalog_keys, rule.match)
    logger.info("module_rule %r (convert=%s) -> %d modules", rule.match, rule.convert, len(candidates))

    for module_path in candidates:
        if module_path in matched_paths:
            continue
        matched_paths.add(module_path)

        bindings = _resolve_bindings(module_path, rule, catalog, catalog_keys)
        mod = create_model_free_module(
            module_path=module_path,
            tensor_bindings=bindings,
            source_format=rule.source_format,
            source_ir=SourceIR(
                kind=rule.source_ir or IRKind.UNKNOWN,
                source_format=rule.source_format,
            ),
            target_ir=None,
            module_kind=rule.module_kind,
        )
        mod.source_ir = infer_source_ir(mod, context.config)
        set_submodule_by_path(root, module_path, mod)


def _fill_ref(ref: TensorRef, catalog: TensorCatalog) -> TensorRef:
    """将 catalog 中 shard/dtype/shape/meta 填入 TensorRef。"""
    entry = catalog.get(ref.key)
    if entry is None:
        return ref
    return TensorRef(
        logical_name=ref.logical_name,
        key=ref.key,
        shard=entry.shard,
        dtype=entry.dtype,
        shape=entry.shape,
        meta=dict(entry.meta),
    )


def _split_catalog_key(key: str) -> tuple[str, str]:
    if "." in key:
        parent, leaf = key.rsplit(".", 1)
        return parent, leaf
    return key, "tensor"


def _get_submodule(root: nn.Module, path: str) -> nn.Module | None:
    parent = root
    for part in path.split("."):
        if not hasattr(parent, part):
            return None
        parent = getattr(parent, part)
    return parent


def _submodule_has_leaves(root: nn.Module, path: str) -> bool:
    """``path`` 上是否已挂载子模块（避免用 Passthrough 覆盖 ``experts.*`` 容器）。"""
    sub = _get_submodule(root, path)
    if sub is None:
        return False
    return len(list(sub.named_children())) > 0


def _install_passthrough_group(
    root: nn.Module,
    install_path: str,
    bindings: dict[str, TensorRef],
    matched_paths: set[str],
) -> int:
    if install_path in matched_paths:
        existing = _get_submodule(root, install_path)
        if isinstance(existing, ModelFreeModule):
            for leaf, ref in bindings.items():
                if leaf not in existing.tensor_bindings:
                    existing.tensor_bindings[leaf] = ref
            existing.lazy_initialized = False
            return len(bindings)

    mod = PassthroughModule(
        full_name=install_path,
        tensor_bindings=bindings,
        source_ir=SourceIR(kind=IRKind.FLOAT, source_format="bf16"),
    )
    set_submodule_by_path(root, install_path, mod)
    matched_paths.add(install_path)
    return len(bindings)


def _attach_preserve_all_catalog(
    root: nn.Module,
    catalog: TensorCatalog,
    catalog_keys: set[str],
    handled: set[str],
    context: ConvertContext,
    matched_paths: set[str],
) -> None:
    """
    将 catalog 中尚未绑定的张量挂为 ``PassthroughModule``，保存时原样 FLOAT 落盘。

    覆盖 ``embed_tokens``、``lm_head``、``norm`` 等非 linears 匹配的 key。
    """
    remaining: list[str] = []
    for key in catalog_keys:
        if key in handled:
            continue
        if key.endswith(WEIGHT_SCALE_INV_SUFFIX):
            continue
        remaining.append(key)

    if not remaining:
        return

    reader = context.reader
    if isinstance(reader, CheckpointReader):
        reader.enrich_catalog(catalog, keys=set(remaining))

    groups: dict[str, dict[str, TensorRef]] = defaultdict(dict)
    for key in remaining:
        entry = catalog.get(key)
        if entry is None:
            continue
        parent, leaf = _split_catalog_key(key)
        groups[parent][leaf] = TensorRef(
            logical_name=leaf,
            key=key,
            shard=entry.shard,
            dtype=entry.dtype,
            shape=entry.shape,
            meta=dict(entry.meta),
        )

    attached = 0
    for module_path, bindings in groups.items():
        if _submodule_has_leaves(root, module_path):
            for leaf, ref in bindings.items():
                attached += _install_passthrough_group(
                    root,
                    ref.key,
                    {leaf: ref},
                    matched_paths,
                )
            continue
        attached += _install_passthrough_group(root, module_path, bindings, matched_paths)

    logger.info(
        "Attached %d catalog tensor(s) across %d passthrough module(s) (non-linear preserve)",
        attached,
        len(groups),
    )
