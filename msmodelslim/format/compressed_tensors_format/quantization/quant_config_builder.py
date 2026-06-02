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

从量化后的 QIR 模型反向推导 compressed-tensors ``QuantizationConfig``。
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

import msmodelslim.ir as qir
from msmodelslim import logger
from msmodelslim.format.compressed_tensors_format.config.base import QuantizationFormat
from msmodelslim.format.compressed_tensors_format.quantization.quant_args import (
    QuantizationStrategy,
)
from msmodelslim.format.compressed_tensors_format.quantization.quant_scheme import (
    QuantizationScheme,
    scheme_for_qir_module,
)
from msmodelslim.ir.qal import QParam, QScheme
from torch import nn


def infer_config_groups(model: nn.Module) -> Dict[str, QuantizationScheme]:
    """扫描模型，按唯一 ``QuantizationScheme`` 分组。"""
    targets = infer_targets(model)
    unique: Dict[str, QuantizationScheme] = {}
    quantized_module_count = 0

    for module in model.modules():
        scheme = build_scheme_from_module(module, targets)
        if scheme is None:
            continue
        quantized_module_count += 1
        unique.setdefault(_scheme_key(scheme), scheme)

    config_groups = {f"group_{idx}": scheme for idx, scheme in enumerate(unique.values())}
    if config_groups:
        logger.info(
            "Inferred %d config group(s) from %d quantized module(s), targets=%s",
            len(config_groups),
            quantized_module_count,
            targets,
        )
    else:
        logger.info("No quantized modules found; config groups inference skipped")
    return config_groups


def infer_targets(model: nn.Module) -> List[str]:
    """从模型中出现的 QIR 量化层推断 ``targets``（当前均为 Linear 替换）。"""
    targets: Set[str] = set()
    for module in model.modules():
        if isinstance(module, qir.AutoFakeQuantLinear):
            targets.add("Linear")
    return sorted(targets) or ["Linear"]


def build_scheme_from_module(module: nn.Module, targets: List[str]) -> Optional[QuantizationScheme]:
    """
    由 QIR 模块实例构造对应的 ``QuantizationScheme``。
    """
    scheme = scheme_for_qir_module(module, targets=targets)

    if scheme is None:
        return None
    return apply_runtime_overrides(scheme, module)


def apply_runtime_overrides(scheme: QuantizationScheme, module: nn.Module) -> QuantizationScheme:
    """根据模块上实际的量化参数覆盖 preset 中的静态默认值。"""
    updates: Dict[str, Any] = {}

    if scheme.weights is not None:
        w_updates: Dict[str, Any] = {}
        w_scheme = _resolve_w_scheme(module)
        if w_scheme is not None:
            w_updates["symmetric"] = w_scheme.symmetric
        group_size = getattr(module, "group_size", None)
        if group_size is not None and scheme.weights.strategy == QuantizationStrategy.GROUP:
            w_updates["group_size"] = group_size
        if not scheme.weights.dynamic:
            w_updates["observer"] = _resolve_weight_observer(module)
        if w_updates:
            updates["weights"] = scheme.weights.model_copy(update=w_updates)

    if scheme.input_activations is not None and scheme.input_activations.dynamic:
        updates["input_activations"] = scheme.input_activations.model_copy(update={"observer": None})

    return scheme.model_copy(update=updates) if updates else scheme


def infer_root_format(config_groups: Dict[str, QuantizationScheme]) -> str:
    formats: List[str] = []
    for scheme in config_groups.values():
        if scheme.format is None:
            continue
        fmt = scheme.format.value if isinstance(scheme.format, QuantizationFormat) else str(scheme.format)
        if fmt not in formats:
            formats.append(fmt)

    if len(formats) > 1:
        root_format = QuantizationFormat.mixed_precision.value
        logger.info(
            "Multiple group formats detected %s; using root format %s",
            formats,
            root_format,
        )
        return root_format
    if len(formats) == 1:
        return formats[0]

    return QuantizationFormat.int_quantized.value


def infer_ignore(model: nn.Module) -> List[str]:
    """
    与官方 ``QuantizationConfig.from_pretrained`` 对齐：
    仅忽略「与已量化层同类型、但本身未量化」的模块（当前即未量化的 ``nn.Linear``）。
    """
    quantized_types: Set[str] = set()
    ignore_by_type: Dict[str, List[str]] = defaultdict(list)

    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, qir.AutoFakeQuantLinear):
            quantized_types.add("Linear")
        elif isinstance(module, nn.Linear):
            ignore_by_type["Linear"].append(name)

    consolidated: List[str] = []
    for layer_type, names in ignore_by_type.items():
        if layer_type in quantized_types:
            consolidated.extend(names)
    ignore_patterns = _compress_module_names_to_regex(consolidated)
    if consolidated:
        logger.info(
            "Inferred ignore list: %d unquantized %s layer(s) compressed to %d pattern(s)",
            len(consolidated),
            ", ".join(sorted(quantized_types)) or "module",
            len(ignore_patterns),
        )
    return ignore_patterns


def _resolve_w_scheme(module: nn.Module) -> Optional[QScheme]:
    scheme = getattr(module, "w_scheme", None)
    if isinstance(scheme, QScheme):
        return scheme
    if isinstance(scheme, QParam):
        return scheme.scheme
    return None


def _resolve_weight_observer(module: nn.Module) -> str:
    """Map QIR weight calibration method to compressed-tensors observer name."""
    w_scheme = _resolve_w_scheme(module)
    if w_scheme is None:
        return "minmax"
    method = getattr(w_scheme, "method", None)
    if method is None and isinstance(getattr(module, "w_scheme", None), QParam):
        method = getattr(module.w_scheme, "method", None)
    if isinstance(method, str) and method:
        return method
    return "minmax"


def _scheme_key(scheme: QuantizationScheme) -> str:
    return json.dumps(scheme.model_dump(), sort_keys=True)


def _compress_module_names_to_regex(module_names: List[str]) -> List[str]:
    cleaned = sorted({name.strip() for name in module_names if isinstance(name, str) and name.strip()})
    if not cleaned:
        return []

    trie: Dict[str, Any] = {}
    for name in cleaned:
        node = trie
        for token in name.split("."):
            node = node.setdefault(token, {})

    def _signature(node: Dict[str, Any]) -> str:
        if not node:
            return "#"
        return "|".join(f"{token}:{_signature(child)}" for token, child in sorted(node.items()))

    def _token_pattern(tokens: List[str]) -> str:
        if all(token.isdigit() for token in tokens):
            alt = "|".join(str(n) for n in sorted({int(t) for t in tokens}))
            return f"(?:{alt})"
        escaped = [re.escape(token) for token in sorted(tokens)]
        return escaped[0] if len(escaped) == 1 else f"(?:{'|'.join(escaped)})"

    def _emit(node: Dict[str, Any]) -> str:
        if not node:
            return ""
        groups: Dict[str, List[str]] = {}
        child_by_signature: Dict[str, Dict[str, Any]] = {}
        for token, child in node.items():
            sig = _signature(child)
            groups.setdefault(sig, []).append(token)
            child_by_signature[sig] = child

        parts: List[str] = []
        for sig, tokens in groups.items():
            token_part = _token_pattern(tokens)
            suffix = _emit(child_by_signature[sig])
            parts.append(f"{token_part}\\.{suffix}" if suffix else token_part)

        if len(parts) == 1:
            return parts[0]
        return f"(?:{'|'.join(sorted(parts))})"

    regex_body = _emit(trie)
    if not regex_body:
        return []
    return [f"re:{regex_body}(?![.\\w])"]
