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

# Convert compressed-tensors ``mxfp4-pack-quantized`` weights to BF16 ``nn.Linear``.
#
# Kimi-K3 expert weights are stored as:
#   - ``weight_packed``: uint8 ``[N, K/2]`` — two FP4 E2M1 nibbles per byte
#   - ``weight_scale``: uint8 ``[N, K/32]`` — E8M0 shared exponents (bias 127)
#
# This is NOT int4-pack (int32 packed + float scale + weight_shape).
# Dequant helpers live in this module (adapter load path), not processor.convert.

from __future__ import annotations

import gc
import os
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Optional

import torch
from safetensors.torch import load_file
from torch import nn
from tqdm import tqdm

from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.security import get_valid_read_path, MAX_READ_FILE_SIZE_32G, json_safe_load

_MXFP4_BLOCK_SIZE = 32
_E8M0_BIAS = 127
_E2M1_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def unpack_fp4_from_uint8(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 packed FP4 to float32 element values (no block scale)."""
    if packed.dtype != torch.uint8:
        packed = packed.to(torch.uint8)
    if packed.shape[-1] < 1:
        raise SchemaValidateError(f"Empty packed last dim: {tuple(packed.shape)}")

    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibbles = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)

    lut = torch.tensor(_E2M1_LUT, device=packed.device, dtype=torch.float32)
    sign = 1.0 - 2.0 * ((nibbles >> 3) & 1).to(torch.float32)
    abs_idx = (nibbles & 0x07).long()
    return sign * lut[abs_idx]


def e8m0_uint8_to_scale(scale_u8: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 uint8 storage to float32 power-of-two scales ``2^(x-127)``."""
    if scale_u8.dtype != torch.uint8:
        scale_u8 = scale_u8.to(torch.uint8)
    shared_exp = scale_u8.to(torch.float32) - float(_E8M0_BIAS)
    return torch.exp2(shared_exp)


def dequant_mxfp4_ct(
    packed: torch.Tensor,
    scale_u8: torch.Tensor,
    block_size: int = _MXFP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Dequantize CT MXFP4 packed weight to float32 ``[N, K]``."""
    if packed.ndim != 2:
        raise SchemaValidateError(f"Expected 2D weight_packed, got shape {tuple(packed.shape)}")

    values = unpack_fp4_from_uint8(packed)
    scale = e8m0_uint8_to_scale(scale_u8)
    while scale.ndim > 2 and scale.shape[-1] == 1:
        scale = scale.squeeze(-1)
    if scale.ndim != 2:
        raise SchemaValidateError(f"Expected 2D weight_scale after squeeze, got shape {tuple(scale.shape)}")

    n, k = values.shape
    scale_n, n_blocks = scale.shape
    if scale_n != n:
        raise SchemaValidateError(f"Mismatch in scale rows ({scale_n}) and weight rows ({n})")
    if n_blocks * block_size != k:
        raise SchemaValidateError(f"K ({k}) is not n_blocks ({n_blocks}) * block_size ({block_size})")

    scale_expanded = scale.repeat_interleave(block_size, dim=-1)
    return values * scale_expanded


npu_available = False
try:
    __import__("torch_npu")
except ImportError:
    pass
else:
    npu_available = True


@lru_cache(maxsize=1)
def get_full_weight_map(model_path: str) -> Dict[str, str]:
    model_index = json_safe_load(os.path.join(model_path, "model.safetensors.index.json"))
    return model_index["weight_map"]


@lru_cache(maxsize=1)
def get_mxfp4_weight_map(model_path: str) -> Dict[str, str]:
    """Map module full name -> safetensors shard for modules with ``.weight_packed``."""
    weight_map = get_full_weight_map(model_path)
    return {k[: -len(".weight_packed")]: v for k, v in weight_map.items() if k.endswith(".weight_packed")}


@lru_cache(maxsize=16)
def _load_shard_state_dict(model_path: str, file_name: str) -> Dict[str, torch.Tensor]:
    file_path = get_valid_read_path(os.path.join(model_path, file_name), "safetensors", size_max=MAX_READ_FILE_SIZE_32G)
    return load_file(file_path, device="cpu")


def load_tensor_by_full_name(model_path: str, full_name: str) -> Optional[torch.Tensor]:
    weight_map = get_full_weight_map(model_path)
    file_name = weight_map.get(full_name)
    if file_name is None:
        return None
    # Reuse shard cache (avoid re-reading the same safetensors for each tensor).
    return _load_shard_state_dict(model_path, file_name).get(full_name)


def _dequant_mxfp4_to_bf16(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.uint8:
        packed = packed.to(torch.uint8)
    if scale.dtype != torch.uint8:
        scale = scale.to(torch.uint8)
    return dequant_mxfp4_ct(packed, scale).to(torch.bfloat16)


def _is_mxfp4_compressed_module(mod: nn.Module) -> bool:
    packed = getattr(mod, "weight_packed", None)
    scale = getattr(mod, "weight_scale", None)
    return isinstance(packed, torch.Tensor) and isinstance(scale, torch.Tensor)


def _module_to_bf16_linear(
    full_name: str,
    mod: nn.Module,
    model_path: str,
    packed: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
) -> Optional[nn.Linear]:
    """Build a BF16 ``nn.Linear`` from mxfp4 tensors or plain weight."""
    weight = None

    if packed is not None and scale is not None:
        weight = _dequant_mxfp4_to_bf16(packed, scale)
    else:
        loaded_weight = load_tensor_by_full_name(model_path, f"{full_name}.weight")
        if loaded_weight is not None:
            weight = loaded_weight.to(torch.bfloat16)
        elif _is_mxfp4_compressed_module(mod):
            weight = _dequant_mxfp4_to_bf16(mod.weight_packed.detach(), mod.weight_scale.detach())
        elif hasattr(mod, "weight") and isinstance(mod.weight, torch.Tensor):
            # Already dense; keep as bf16 if shapes look like Linear.
            if getattr(mod, "in_features", None) is not None and getattr(mod, "out_features", None) is not None:
                expected = (mod.out_features, mod.in_features)
                if tuple(mod.weight.shape) == expected:
                    weight = mod.weight.detach().to(torch.bfloat16)

    if weight is None:
        return None

    out_features, in_features = weight.shape
    if getattr(mod, "in_features", None) is not None and getattr(mod, "out_features", None) is not None:
        if (mod.out_features, mod.in_features) != (out_features, in_features):
            raise SchemaValidateError(
                f"{full_name}: dequant shape {(out_features, in_features)} "
                f"!= Linear ({mod.out_features}, {mod.in_features})"
            )

    loaded_bias = load_tensor_by_full_name(model_path, f"{full_name}.bias")
    if loaded_bias is not None:
        bias = loaded_bias.to(torch.bfloat16)
    elif getattr(mod, "bias", None) is not None and isinstance(mod.bias, torch.Tensor):
        bias = mod.bias.detach().to(torch.bfloat16)
    else:
        bias = None

    new_linear = nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias is not None,
        device=weight.device,
        dtype=torch.bfloat16,
    )
    new_linear.weight.data.copy_(weight)
    if bias is not None:
        new_linear.bias.data.copy_(bias)
    new_linear.eval()
    return new_linear


def auto_convert_module_mxfp4_to_bf16(name: str, module: nn.Module, model_path: str):
    """Convert all mxfp4-pack submodules under ``module`` to BF16 Linear."""
    weight_map = get_mxfp4_weight_map(model_path)
    if not weight_map:
        get_logger().info("No mxfp4 weight_packed entries found, skip conversion.")
        return
    try:
        sub_weight_map = {
            sub_name: weight_map[sub_name]
            for sub_name, _ in module.named_modules(prefix=name)
            if sub_name in weight_map
        }
    except KeyError:
        get_logger().warning("Safetensors files not match index.json, skip mxfp4 to bf16.")
        return

    if not sub_weight_map:
        return
    convert_module_mxfp4_to_bf16(name, module, model_path, weight_map=sub_weight_map)


def replace_compressed_linear_with_bf16(root_module: nn.Module, root_prefix: str, model_path: str) -> nn.Module:
    """Replace CompressedLinear (or mxfp4 buffer modules) under ``root_module`` with BF16 Linear."""

    def _convert_one(full_name: str, mod: nn.Module) -> Optional[nn.Linear]:
        return _module_to_bf16_linear(full_name, mod, model_path)

    if root_module.__class__.__name__ == "CompressedLinear" or _is_mxfp4_compressed_module(root_module):
        new_root = _convert_one(root_prefix, root_module)
        return new_root if new_root is not None else root_module

    targets = [
        (name, mod)
        for name, mod in root_module.named_modules()
        if name and (mod.__class__.__name__ == "CompressedLinear" or _is_mxfp4_compressed_module(mod))
    ]
    for name, mod in targets:
        full_name = f"{root_prefix}.{name}" if root_prefix else name
        new_linear = _convert_one(full_name, mod)
        if new_linear is None:
            continue
        root_module.set_submodule(name, new_linear)

    return root_module


def dequant_subtree_mxfp4_to_bf16(module: nn.Module, prefix: str, model_path: str) -> nn.Module:
    """Convert mxfp4-pack / CompressedLinear under ``module`` to BF16 Linear.

    Runs index-based auto convert, then class-based CompressedLinear replacement.
    When ``prefix`` is empty (full VLM), also rebinds vision / mm_projector / lm_head
    in case those roots themselves are CompressedLinear.
    """
    auto_convert_module_mxfp4_to_bf16(prefix, module, model_path)

    if prefix:
        return replace_compressed_linear_with_bf16(module, prefix, model_path)

    # Full-model load: replace known tops that may themselves be CompressedLinear.
    if hasattr(module, "vision_tower") and module.vision_tower is not None:
        module.vision_tower = replace_compressed_linear_with_bf16(module.vision_tower, "vision_tower", model_path)
    if getattr(module, "mm_projector", None) is not None:
        module.mm_projector = replace_compressed_linear_with_bf16(module.mm_projector, "mm_projector", model_path)
    language_model = getattr(module, "language_model", None)
    if language_model is not None and hasattr(language_model, "lm_head"):
        language_model.lm_head = replace_compressed_linear_with_bf16(
            language_model.lm_head, "language_model.lm_head", model_path
        )
    return module


@torch.no_grad()
def convert_module_mxfp4_to_bf16(name: str, module: nn.Module, model_path: str, weight_map: Dict[str, str]):
    target_sub_modules = {
        sub_name: sub_module for sub_name, sub_module in module.named_modules(prefix=name) if sub_name in weight_map
    }
    file_to_sub_names: Dict[str, List[str]] = defaultdict(list)
    for sub_name in target_sub_modules:
        file_to_sub_names[weight_map[sub_name]].append(sub_name)

    with tqdm(total=len(target_sub_modules), desc="mxfp4 to bf16") as bars:
        for file_name, sub_names in file_to_sub_names.items():
            file_state = _load_shard_state_dict(model_path, file_name)
            for sub_name in sub_names:
                sub_module = target_sub_modules[sub_name]
                packed_key = f"{sub_name}.weight_packed"
                scale_key = f"{sub_name}.weight_scale"
                if packed_key not in file_state or scale_key not in file_state:
                    get_logger().warning("Missing mxfp4 tensors for %s, skip.", sub_name)
                    bars.update(1)
                    continue

                packed_weight = file_state[packed_key]
                scale = file_state[scale_key]
                dequant_weight = _dequant_mxfp4_to_bf16(packed_weight, scale)

                if (
                    sub_module.__class__.__name__ == "CompressedLinear"
                    or _is_mxfp4_compressed_module(sub_module)
                    or not hasattr(sub_module, "weight")
                ):
                    new_linear = _module_to_bf16_linear(
                        sub_name, sub_module, model_path, packed=packed_weight, scale=scale
                    )
                    if new_linear is None:
                        bars.update(1)
                        continue
                    relative_name = sub_name[len(name) + 1 :] if name else sub_name
                    module.set_submodule(relative_name, new_linear)
                    del new_linear
                else:
                    target_weight = sub_module.weight
                    if tuple(target_weight.shape) != tuple(dequant_weight.shape):
                        # Shape mismatch: replace whole module with Linear.
                        new_linear = _module_to_bf16_linear(
                            sub_name, sub_module, model_path, packed=packed_weight, scale=scale
                        )
                        if new_linear is not None:
                            relative_name = sub_name[len(name) + 1 :] if name else sub_name
                            module.set_submodule(relative_name, new_linear)
                            del new_linear
                    else:
                        sub_module.weight.data.copy_(
                            dequant_weight.to(device=target_weight.device, dtype=target_weight.dtype)
                        )

                del packed_weight, scale, dequant_weight
                bars.update(1)

            del file_state
            gc.collect()
            if npu_available:
                try:
                    torch.npu.empty_cache()
                except Exception as exc:
                    get_logger().warning("Failed to clear NPU cache: %s", exc)

    cache_clear = getattr(_load_shard_state_dict, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()
