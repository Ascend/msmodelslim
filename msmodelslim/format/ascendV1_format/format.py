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

AscendV1Format: the concrete IFormatLoader for AscendV1 exports.
"""

import os
from typing import Any, Dict, List, Optional, Set

import torch
from torch import nn

from msmodelslim.ir.attention import FakeQuantDynamicCache
from msmodelslim.ir.auto import AutoFakeQuantActivation, AutoFakeQuantDynamicCache, AutoFakeQuantLinear
from msmodelslim.processor.quant.fa3.interface import FA3QuantAdapterInterface
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.security import json_safe_load
from msmodelslim.utils.security.path import get_valid_read_path

from .decoder import (
    AscendV1FormatDecoder,
    discover_fa3_targets,
    fallback_submodule_names,
    branch_from_submodule,
    fa3_scheme_from_module,
)
from ..common.ir_builder import IrBuilder, try_get_builder_for_module
from ..interface import IFormatLoader


ASCENDV1_DESC_JSON_NAME = "quant_model_description.json"
ASCENDV1_SAFETENSORS_NAME = "quant_model_weights.safetensors"

FLOAT_LABEL = "FLOAT"

META_KEYS = {
    "model_quant_type",
    "version",
    "group_size",
    "kv_quant_type",
    "kv_cache_type",
    "fa_quant_type",
    "reduce_quant_type",
    "metadata",
    "optional",
}

_LINEAR_TYPES = (nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear)

_DECODER_LAYER_SUFFIX = "DecoderLayer"


def _is_meta_module(module: nn.Module) -> bool:
    for tensor in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
        if tensor.device.type == "meta":
            return True
    return False


def _materialize_module_cpu(module: nn.Module) -> None:
    if _is_meta_module(module):
        module.to_empty(device="cpu")


@logger_setter(prefix="msmodelslim.format.ascend_v1")
class AscendV1Format(IFormatLoader):
    """AscendV1 format loader: IR structure from description, hydrate values separately.

    Mediates between ``AscendV1FormatDecoder`` (reads safetensors, inverse mapping, plain-tensor
    ``QuantParamSet``) and ``IrBuilder`` (shell transform, param binding). ``apply_ir`` walks the
    full tree once; ``hydrate`` is scoped by ``prefix`` / ``skip_prefixes``.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = get_valid_read_path(model_path, is_dir=True)
        self.device = torch.device(device)
        self.description = self._load_description()

        from .tensor_store import AscendV1TensorStore

        self.store = AscendV1TensorStore(self.model_path)
        self.group_size = int(self.description.get("group_size", 0) or 0)
        self._decoder = AscendV1FormatDecoder(self.store)
        self._adapter: Optional[object] = None

    # ---- IFormatLoader ----

    def bind_adapter(self, adapter: object) -> None:
        self._adapter = adapter

    def apply_ir(self, model: nn.Module) -> None:
        """Apply FakeQuant IR to every described Linear / FA3 / KV-cache in the full tree."""
        self._apply_linear_ir(model)
        self._apply_fa3_ir(model)
        self._apply_kv_cache_ir(model)

    def hydrate(self, model: nn.Module, prefix: Optional[str] = None, skip_prefixes: Optional[Set[str]] = None) -> None:
        """Load FakeQuant / FLOAT weights for modules in scope."""
        hydrated = 0
        for name, module in list(model.named_modules()):
            if not self._in_scope(name, prefix, skip_prefixes):
                continue
            hydrated += self._hydrate_named_module(name, module)

        # FLOAT keys
        state = model.state_dict()
        float_keys = [
            key
            for key, desc in self.description.items()
            if key not in META_KEYS
            and desc == FLOAT_LABEL
            and key in state
            and self._in_scope(key, prefix, skip_prefixes)
        ]
        hydrated += len(self._copy_float_keys(model, float_keys))
        self._materialize_runtime_tensors(model, prefix=prefix, skip_prefixes=skip_prefixes)
        if prefix:
            get_logger().debug("Hydrated AscendV1 prefix=%s, touches=%d", prefix, hydrated)
        else:
            get_logger().info(
                "Hydrated AscendV1 tensors (scope=%s): %d modules/keys touched",
                "skip" if skip_prefixes else "full",
                hydrated,
            )

    # ---- IR application ----

    def _apply_linear_ir(self, model: nn.Module) -> None:
        replaced = 0
        for name, module in list(model.named_modules()):
            if not isinstance(module, _LINEAR_TYPES):
                continue
            if isinstance(module, AutoFakeQuantLinear):
                continue
            weight_key = f"{name}.weight"
            if weight_key not in self.description:
                continue
            quant_type = self.layer_quant_type(name)
            if quant_type == FLOAT_LABEL:
                continue
            decoder = self._decoder.linear(quant_type)
            ir_type = decoder.ir_type(self.group_size)
            spec = decoder.structure(name, self.group_size)
            builder = IrBuilder.create(ir_type)
            result = builder.transform_shell(model, name, module, structure=spec)
            if result.mode.value != "skip":
                replaced += 1
        get_logger().info("Applied AscendV1 IR structure: replaced %d Linear", replaced)

    def _apply_fa3_ir(self, model: nn.Module) -> None:
        # Group FA3 targets by attn_prefix so we can iterate over the model
        # (not the description) and naturally skip layers that exist in the
        # export but not in the inference shell (e.g. MTP layer).
        targets_by_attn: Dict[str, List[Any]] = {}
        for target in discover_fa3_targets(self.description):
            targets_by_attn.setdefault(target.attn_prefix, []).append(target)
        if not targets_by_attn:
            return

        # FA3 requires the adapter to wrap attention forward (inject_fa3_placeholders);
        # without it the inserted IR modules would be dead code. Fail loudly.
        if not isinstance(self._adapter, FA3QuantAdapterInterface):
            raise UnsupportedError(
                f"Export description contains {sum(len(v) for v in targets_by_attn.values())} "
                f"FA3 target(s) but adapter {type(self._adapter).__name__} does not implement "
                f"FA3QuantAdapterInterface",
                action="Please use a model adapter that supports FA3 activation quantization.",
            )

        # Inject FA3 placeholders (creates modules + wraps attention forward).
        self._inject_fa3_placeholders(model)

        # Iterate over MODEL modules (not description targets) so that layers
        # present in the export but absent from the inference shell (e.g. MTP)
        # are simply never matched.
        inserted = 0
        for name, _module in model.named_modules():
            targets = targets_by_attn.get(name)
            if not targets:
                continue
            for target in targets:
                module_name = self._resolve_fa3_module_name(model, target)
                if module_name is None:
                    continue
                source = model.get_submodule(module_name)
                if isinstance(source, AutoFakeQuantActivation):
                    continue
                decoder = self._decoder.fa3(target.scheme)
                ir_type = decoder.ir_type(target.scheme)
                spec = decoder.structure(module_name, self.group_size, scheme=target.scheme)
                builder = IrBuilder.create(ir_type)
                builder.transform_shell(model, module_name, source, structure=spec)
                inserted += 1
        if inserted:
            get_logger().info("Applied AscendV1 IR structure: inserted %d FA3 IR modules", inserted)

    def _inject_fa3_placeholders(self, model: nn.Module) -> None:
        adapter = self._adapter
        try:
            adapter.inject_fa3_placeholders("", model, lambda _name: True)
        except Exception as exc:  # pylint: disable=broad-except
            get_logger().warning("inject_fa3_placeholders failed: %s", exc)

    def _apply_kv_cache_ir(self, model: nn.Module) -> None:
        """Apply FakeQuantDynamicCache IR to every attention layer that has C8 KV-cache tensors.

        Reuses ``inject_fa3_placeholders`` (same as FA3) to wrap attention forward and
        create ``fa_k`` / ``fa_v`` placeholders, then replaces those placeholders with
        real ``FakeQuantDynamicCache`` modules via ``KvCacheIrBuilder.transform_shell``.
        """
        kv_cache_type = self.description.get("kv_cache_type")
        if kv_cache_type != "C8":
            return
        if not isinstance(self._adapter, FA3QuantAdapterInterface):
            raise UnsupportedError(
                f"Export description has kv_cache_type='C8' but adapter "
                f"{type(self._adapter).__name__} does not implement FA3QuantAdapterInterface",
                action="Please use a model adapter that supports KV cache quantization.",
            )
        # Ensure attention forward is wrapped (idempotent).
        self._inject_fa3_placeholders(model)

        # Auto-discover attention layers (same logic as DynamicCacheQuantProcessor).
        import re

        attention_layers: Dict[int, str] = {}
        for name, module in model.named_modules():
            class_name = module.__class__.__name__.lower()
            if "attention" in class_name or "attn" in class_name:
                numbers = re.findall(r"\.(\d+)\.", name)
                if numbers:
                    layer_idx = int(numbers[0])
                    attention_layers[layer_idx] = name

        decoder = self._decoder.kv_cache("C8")
        builder = IrBuilder.create(FakeQuantDynamicCache)
        inserted = 0
        for _layer_idx, attn_name in attention_layers.items():
            k_prefix = f"{attn_name}.k_proj"
            v_prefix = f"{attn_name}.v_proj"
            if not self.store.has(f"{k_prefix}.kv_cache_scale"):
                continue
            # fa_k -> FakeQuantDynamicCache
            k_name = f"{attn_name}.fa_k"
            k_source = model.get_submodule(k_name)
            if isinstance(k_source, AutoFakeQuantDynamicCache):
                continue
            k_spec = decoder.structure(k_prefix, self.group_size)
            builder.transform_shell(model, k_name, k_source, structure=k_spec)
            # fa_v -> FakeQuantDynamicCache
            v_name = f"{attn_name}.fa_v"
            v_source = model.get_submodule(v_name)
            if isinstance(v_source, AutoFakeQuantDynamicCache):
                continue
            v_spec = decoder.structure(v_prefix, self.group_size)
            builder.transform_shell(model, v_name, v_source, structure=v_spec)
            inserted += 2
        if inserted:
            get_logger().info("Applied AscendV1 IR structure: inserted %d KV-cache IR modules", inserted)

    def _resolve_fa3_module_name(self, model: nn.Module, target) -> Optional[str]:
        if target.module_name:
            return target.module_name
        parent = model.get_submodule(target.attn_prefix)
        for child_name, _child in parent.named_children():
            if branch_from_submodule(child_name) == target.branch:
                return f"{target.attn_prefix}.{child_name}"
        fallbacks = fallback_submodule_names(target.branch)
        if not fallbacks:
            return None
        return f"{target.attn_prefix}.{fallbacks[0]}"

    # ---- Hydrate ----

    def _hydrate_named_module(self, name: str, module: nn.Module) -> int:
        if isinstance(module, AutoFakeQuantDynamicCache):
            # Map module name back to safetensors key prefix.
            # fa_k / key_states_quantizer -> {parent}.k_proj
            # fa_v / value_states_quantizer -> {parent}.v_proj
            parent_name, quantizer_name = name.rsplit(".", 1)
            if quantizer_name == "fa_k" or "key_states" in quantizer_name:
                sf_prefix = f"{parent_name}.k_proj"
            elif quantizer_name == "fa_v" or "value_states" in quantizer_name:
                sf_prefix = f"{parent_name}.v_proj"
            else:
                return 0
            decoder = self._decoder.kv_cache("C8")
            param_set = decoder.params(sf_prefix, self.device, self.group_size)
            builder = try_get_builder_for_module(module)
            if builder is None:
                raise SchemaValidateError(
                    f"No IrBuilder for {type(module).__name__} at '{name}'",
                    action="Please apply_ir before hydrate.",
                )
            builder.bind_from_param_set(module, param_set)
            return 1
        if isinstance(module, AutoFakeQuantActivation):
            scheme = fa3_scheme_from_module(module)
            if scheme is None:
                return 0
            decoder = self._decoder.fa3(scheme)
            param_set = decoder.params(name, self.device, self.group_size, scheme=scheme)
            builder = try_get_builder_for_module(module)
            if builder is None:
                raise SchemaValidateError(
                    f"No IrBuilder for {type(module).__name__} at '{name}'",
                    action="Please apply_ir before hydrate.",
                )
            builder.bind_from_param_set(module, param_set)
            return 1
        if f"{name}.weight" not in self.description:
            return 0
        if isinstance(module, AutoFakeQuantLinear):
            label = self.layer_quant_type(name)
            decoder = self._decoder.linear(label)
            param_set = decoder.params(name, self.device, self.group_size)
            builder = try_get_builder_for_module(module)
            if builder is None:
                raise SchemaValidateError(
                    f"No IrBuilder for {type(module).__name__} at '{name}'",
                    action="Please apply_ir before hydrate.",
                )
            builder.bind_from_param_set(module, param_set)
            return 1
        if isinstance(module, _LINEAR_TYPES) and self.layer_quant_type(name) == FLOAT_LABEL:
            self._hydrate_float_linear(name, module)
            return 1
        return 0

    def _hydrate_float_linear(self, name: str, module: nn.Module) -> None:
        weight_key = f"{name}.weight"
        if not self.store.has(weight_key):
            raise SchemaValidateError(
                f"Missing FLOAT weight '{weight_key}'",
                action="Please ensure AscendV1 export contains FLOAT Linear weights.",
            )
        _materialize_module_cpu(module)
        with torch.no_grad():
            weight = self.store.get(weight_key).to(dtype=module.weight.dtype, device="cpu")
            module.weight.copy_(weight)
            bias_key = f"{name}.bias"
            if module.bias is not None and self.store.has(bias_key):
                bias = self.store.get(bias_key).to(dtype=module.bias.dtype, device="cpu")
                module.bias.copy_(bias)

    def _copy_float_keys(self, model: nn.Module, float_keys: List[str]) -> Set[str]:
        loaded: Set[str] = set()
        with torch.no_grad():
            for key in float_keys:
                if not self.store.has(key):
                    continue
                parent_name = key.rsplit(".", 1)[0] if "." in key else ""
                modules = dict(model.named_modules())
                parent = modules.get(parent_name)
                if isinstance(parent, (AutoFakeQuantLinear, AutoFakeQuantActivation)):
                    continue
                if parent is not None and _is_meta_module(parent):
                    _materialize_module_cpu(parent)
                state = model.state_dict()
                if key not in state:
                    continue
                target = state[key]
                if target.device.type == "meta":
                    if parent is not None:
                        _materialize_module_cpu(parent)
                    state = model.state_dict()
                    target = state[key]
                tensor = self.store.get(key).to(dtype=target.dtype, device="cpu")
                if tuple(tensor.shape) != tuple(target.shape):
                    raise SchemaValidateError(
                        f"Shape mismatch for FLOAT tensor '{key}': "
                        f"file={tuple(tensor.shape)}, model={tuple(target.shape)}",
                        action="Please ensure model config matches the AscendV1 export.",
                    )
                target.copy_(tensor)
                loaded.add(key)
        return loaded

    def _materialize_runtime_tensors(
        self,
        model: nn.Module,
        *,
        prefix: Optional[str] = None,
        skip_prefixes: Optional[Set[str]] = None,
    ) -> None:
        for name, module in model.named_modules():
            if not self._in_scope(name, prefix, skip_prefixes):
                continue
            if _is_meta_module(module) and not list(module.children()):
                _materialize_module_cpu(module)
                get_logger().debug("Materialized remaining meta tensors for %s", name or type(module).__name__)

    # ---- Helpers ----

    def _load_description(self) -> Dict:
        desc_path = os.path.join(self.model_path, ASCENDV1_DESC_JSON_NAME)
        if not os.path.isfile(desc_path):
            raise SchemaValidateError(
                f"Missing {ASCENDV1_DESC_JSON_NAME} under {self.model_path}",
                action="Please provide a valid AscendV1 export directory.",
            )
        description = json_safe_load(desc_path, check_user_stat=False)
        if not isinstance(description, dict):
            raise SchemaValidateError(
                f"{ASCENDV1_DESC_JSON_NAME} must be a JSON object",
                action="Please check AscendV1 export integrity.",
            )
        return description

    def layer_quant_type(self, module_name: str) -> str:
        key = f"{module_name}.weight"
        if key not in self.description:
            raise SchemaValidateError(
                f"'{key}' not found in {ASCENDV1_DESC_JSON_NAME}",
                action="Description must cover every Linear weight in the model.",
            )
        quant_type = self.description[key]
        if quant_type == FLOAT_LABEL:
            return quant_type
        if not self._decoder.has_linear_label(quant_type):
            raise UnsupportedError(
                f"Unsupported layer quant type {quant_type!r} for '{key}'",
                action=f"Supported layer types: {', '.join(sorted(self._decoder._linear.keys()))} and {FLOAT_LABEL}.",
            )
        return quant_type

    def decoder_prefixes(self, model: nn.Module) -> Set[str]:
        """Return the set of decoder-layer module names (for shared/layer scoping)."""
        return {
            name for name, module in model.named_modules() if module.__class__.__name__.endswith(_DECODER_LAYER_SUFFIX)
        }

    @staticmethod
    def _in_scope(name: str, prefix: Optional[str], skip_prefixes: Optional[Set[str]]) -> bool:
        if prefix is not None:
            return name == prefix or name.startswith(prefix + ".")
        if skip_prefixes is not None:
            for sp in skip_prefixes:
                if name == sp or name.startswith(sp + "."):
                    return False
        return True
