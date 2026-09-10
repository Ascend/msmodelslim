#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You may use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------

AscendV1 format decoder: holds the tensor store and three dispatch dicts
(linear / fa3 / kv_cache). Mirrors the save-side ``AscendV1Saver.process_map``
pattern but with three separate dicts for the three IR categories.

Also absorbs the FA3 quant_type parsing and the FA3 module reverse-lookup.
"""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

from torch import nn

from msmodelslim.ir.const import (
    fp8_e4m3_per_head_sym,
    fp8_e4m3_per_token_sym,
    int8_per_head_sym,
    int8_per_token_sym,
    mxfp4_per_block_sym,
    mxfp8_per_block_sym,
)
from msmodelslim.ir.qal import QScheme
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError

from .decoders.base import Fa3IrDecoder, KvCacheIrDecoder, LinearIrDecoder
from .decoders.fa3_per_block import Fa3PerBlockDecoder
from .decoders.fa3_per_head import Fa3PerHeadDecoder
from .decoders.fa3_per_token import Fa3PerTokenDecoder
from .decoders.kv_cache_c8 import KvCacheC8Decoder
from .decoders.w8a8_dynamic import W8A8DynamicDecoder
from .decoders.w8a8_mx import W8A8MxDecoder
from .decoders.w8a8_static import W8A8StaticDecoder

# ---- FA3 quant_type parsing ----

QUANT_TYPE_SUFFIX = ".quant_type"
SCALE_SUFFIX = ".scale"

_ACTS_LONGEST_FIRST = (
    "QKVP",
    "QKV",
    "QKP",
    "QVP",
    "KVP",
    "QK",
    "QV",
    "QP",
    "KV",
    "KP",
    "VP",
    "Q",
    "K",
    "V",
    "P",
)
_DTYPES_LONGEST_FIRST = ("MXFP8", "MXFP4", "INT8", "FP8")
_DEFAULT_BRANCHES = ("Q", "K", "V")
_FALLBACK_SUBMODULE = {
    "Q": ("fa_q", "fa3_q"),
    "K": ("fa_k", "fa3_k"),
    "V": ("fa_v", "fa3_v"),
    "P": ("knope_p", "fa_p", "fa3_p"),
}

# FA3 activation schemes are the single vocabulary used end-to-end: the disk
# ``*.quant_type`` string is resolved to one of these QSchemes at parse time,
# decoders dispatch on the QScheme, and built IR modules expose it via
# ``x_q_scheme``. ``(dtype_label, is_dynamic)`` is only an intermediate wording
# of the disk string.
_FA3_SCHEMES = frozenset(
    {
        int8_per_head_sym,
        fp8_e4m3_per_head_sym,
        int8_per_token_sym,
        fp8_e4m3_per_token_sym,
        mxfp4_per_block_sym,
        mxfp8_per_block_sym,
    }
)

# Disk quant_type wording -> QScheme. Static means per-head, dynamic means
# per-token (INT8/FP8) or per-block (MXFP4/MXFP8), mirroring the save side.
_FA3_SCHEME_BY_DTYPE_AND_DYNAMIC = {
    ("INT8", False): int8_per_head_sym,
    ("FP8", False): fp8_e4m3_per_head_sym,
    ("INT8", True): int8_per_token_sym,
    ("FP8", True): fp8_e4m3_per_token_sym,
    ("MXFP4", True): mxfp4_per_block_sym,
    ("MXFP8", True): mxfp8_per_block_sym,
}


@dataclass(frozen=True)
class Fa3BranchTarget:
    """One FA3 activation branch under an attention module."""

    attn_prefix: str
    branch: str
    scheme: QScheme
    module_name: Optional[str] = None


def _parse_fa_quant_type(quant_type: str) -> Dict[str, QScheme]:
    """Invert ``update_fa_quant_type`` into ``{branch: QScheme}``."""
    if not isinstance(quant_type, str) or not quant_type:
        raise SchemaValidateError(
            f"Invalid FA3 quant_type={quant_type!r}",
            action="Expected a non-empty string such as INT8 or Q_FP8_DYNAMIC.",
        )
    remaining = quant_type
    result: Dict[str, QScheme] = {}
    while remaining:
        remaining = remaining.lstrip("_")
        if not remaining:
            break
        act_prefix, remaining = _consume_act_prefix(remaining)
        dtype, remaining = _consume_dtype(remaining, quant_type)
        is_dynamic = remaining.startswith("_DYNAMIC")
        if is_dynamic:
            remaining = remaining[len("_DYNAMIC") :]
        scheme = _FA3_SCHEME_BY_DTYPE_AND_DYNAMIC.get((dtype, is_dynamic))
        if scheme is None:
            raise SchemaValidateError(
                f"Unsupported FA3 quant_type={quant_type!r} for dtype={dtype!r}, dynamic={is_dynamic}",
                action="Supported dtype/dynamic combos: INT8, INT8_DYNAMIC, FP8, "
                "FP8_DYNAMIC, MXFP4_DYNAMIC, MXFP8_DYNAMIC.",
            )
        branches = tuple(act_prefix) if act_prefix else _DEFAULT_BRANCHES
        for branch in branches:
            result[branch] = scheme
    if not result:
        raise SchemaValidateError(
            f"Could not parse FA3 quant_type={quant_type!r}",
            action="Supported forms: INT8, FP8_DYNAMIC, Q_FP8, QV_INT8_K_FP8, MXFP8_DYNAMIC.",
        )
    return result


def _consume_act_prefix(remaining: str) -> Tuple[str, str]:
    for acts in _ACTS_LONGEST_FIRST:
        if remaining == acts:
            return acts, ""
        prefix = acts + "_"
        if remaining.startswith(prefix):
            after = remaining[len(prefix) :]
            if after.startswith(_DTYPES_LONGEST_FIRST):
                return acts, after
    return "", remaining


def _consume_dtype(remaining: str, original: str) -> Tuple[str, str]:
    for dtype in _DTYPES_LONGEST_FIRST:
        if remaining == dtype:
            return dtype, ""
        if remaining.startswith(dtype):
            after = remaining[len(dtype) :]
            if after == "" or after.startswith("_"):
                return dtype, after
    raise SchemaValidateError(
        f"Could not parse dtype in FA3 quant_type={original!r} at {remaining!r}",
        action="Supported dtypes: INT8, FP8, MXFP4, MXFP8.",
    )


def _faquant_scale_module_names(description: Mapping[str, object]) -> Dict[str, Dict[str, str]]:
    names: Dict[str, Dict[str, str]] = {}
    for key, value in description.items():
        if value != "FAQuant" or not isinstance(key, str) or not key.endswith(SCALE_SUFFIX):
            continue
        module_name = key[: -len(SCALE_SUFFIX)]
        if "." not in module_name:
            continue
        attn_prefix, submodule = module_name.rsplit(".", 1)
        branch = submodule.split("_")[-1].upper()
        if branch not in _FALLBACK_SUBMODULE:
            continue
        names.setdefault(attn_prefix, {})[branch] = module_name
    return names


def discover_fa3_targets(description: Mapping[str, object]) -> List[Fa3BranchTarget]:
    """Collect FA3 branch targets from ``quant_model_description.json``."""
    parsed_by_attn: Dict[str, Dict[str, QScheme]] = {}
    for key, value in description.items():
        if not isinstance(key, str) or not key.endswith(QUANT_TYPE_SUFFIX):
            continue
        if not isinstance(value, str):
            continue
        attn_prefix = key[: -len(QUANT_TYPE_SUFFIX)]
        parsed_by_attn[attn_prefix] = _parse_fa_quant_type(value)

    scale_names = _faquant_scale_module_names(description)
    targets: List[Fa3BranchTarget] = []
    for attn_prefix, branch_info in parsed_by_attn.items():
        names_by_branch = scale_names.get(attn_prefix, {})
        for branch, scheme in branch_info.items():
            targets.append(
                Fa3BranchTarget(
                    attn_prefix=attn_prefix,
                    branch=branch,
                    scheme=scheme,
                    module_name=names_by_branch.get(branch),
                )
            )
    return targets


def fallback_submodule_names(branch: str) -> Tuple[str, ...]:
    return _FALLBACK_SUBMODULE.get(branch, ())


def branch_from_submodule(submodule: str) -> str:
    return submodule.split("_")[-1].upper()


# ---- FA3 module reverse-lookup ----


def fa3_scheme_from_module(module: nn.Module) -> Optional[QScheme]:
    """Recover the ``QScheme`` of an already-built FA3 activation IR module.

    Used during hydrate to find the right decoder for an existing ``AutoFakeQuantActivation``.
    Every FA3 activation module stores its scheme in ``x_q_scheme``; membership in
    ``_FA3_SCHEMES`` is sufficient to recognise it (no per-class dtype mapping needed).
    """
    scheme = getattr(module, "x_q_scheme", None)
    if scheme in _FA3_SCHEMES:
        return scheme
    return None


class AscendV1FormatDecoder:
    """AscendV1 tensor decoder with three dispatch dicts (linear / fa3 / kv_cache).

    Holds the ``AscendV1TensorStore`` and per-IR decoder instances. Callers use
    ``linear(label)`` / ``fa3(scheme)`` / ``kv_cache(label)`` to get
    the right decoder, then call ``structure`` / ``params`` on it.
    """

    def __init__(self, store) -> None:
        self.store = store
        self._linear: Dict[str, LinearIrDecoder] = {
            W8A8StaticDecoder.label: W8A8StaticDecoder(store),
            W8A8DynamicDecoder.label: W8A8DynamicDecoder(store),
            W8A8MxDecoder.label: W8A8MxDecoder(store),
        }
        self._fa3: Dict[QScheme, Fa3IrDecoder] = {
            int8_per_head_sym: Fa3PerHeadDecoder(store),
            fp8_e4m3_per_head_sym: Fa3PerHeadDecoder(store),
            int8_per_token_sym: Fa3PerTokenDecoder(store),
            fp8_e4m3_per_token_sym: Fa3PerTokenDecoder(store),
            mxfp4_per_block_sym: Fa3PerBlockDecoder(store),
            mxfp8_per_block_sym: Fa3PerBlockDecoder(store),
        }
        self._kv_cache: Dict[str, KvCacheIrDecoder] = {
            KvCacheC8Decoder.label: KvCacheC8Decoder(store),
        }

    def linear(self, label: str) -> LinearIrDecoder:
        decoder = self._linear.get(label)
        if decoder is None:
            raise UnsupportedError(
                f"No AscendV1 decoder for linear label={label!r}",
                action=f"Supported labels: {', '.join(sorted(self._linear.keys()))}.",
            )
        return decoder

    def fa3(self, scheme: QScheme) -> Fa3IrDecoder:
        decoder = self._fa3.get(scheme)
        if decoder is None:
            supported = ", ".join(repr(s) for s in sorted(self._fa3.keys(), key=repr))
            raise UnsupportedError(
                f"No AscendV1 decoder for fa3 scheme={scheme!r}",
                action=f"Supported: {supported}.",
            )
        return decoder

    def kv_cache(self, label: str) -> KvCacheIrDecoder:
        decoder = self._kv_cache.get(label)
        if decoder is None:
            raise UnsupportedError(
                f"No AscendV1 decoder for kv_cache label={label!r}",
                action=f"Supported labels: {', '.join(sorted(self._kv_cache.keys()))}.",
            )
        return decoder

    def has_linear_label(self, label: str) -> bool:
        return label in self._linear
