# -*- coding: UTF-8 -*-

# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.

import os
import gc
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Union, Literal

import torch
from safetensors.torch import load_file
from torch import nn
from tqdm import tqdm

try:
    import numpy as np
except ImportError:
    np = None

from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.security import get_valid_read_path, MAX_READ_FILE_SIZE_32G, json_safe_load


@lru_cache(maxsize=1)
def get_full_weight_map(model_path: str) -> Dict[str, str]:
    model_index = json_safe_load(os.path.join(model_path, "model.safetensors.index.json"))
    return model_index["weight_map"]


def load_tensor_by_full_name(model_path: str, full_name: str):
    weight_map = get_full_weight_map(model_path)
    file_name = weight_map.get(full_name)
    if file_name is None:
        return None
    file_path = os.path.join(model_path, file_name)
    file_path = get_valid_read_path(file_path, extensions="safetensors", size_max=MAX_READ_FILE_SIZE_32G)
    file_state = load_file(file_path, device="cpu")
    return file_state.get(full_name)


npu_available = False
try:
    __import__("torch_npu")
except ImportError:
    pass
else:
    npu_available = True


def _unpack_from_int32_torch(
    value: torch.Tensor, num_bits: int, shape: torch.Size, packed_dim: Union[Literal[0], Literal[1]] = 1
) -> torch.Tensor:
    if value.dtype is not torch.int32:
        raise SchemaValidateError(f"Expected {torch.int32} but got {value.dtype}, Aborting unpack.")
    if num_bits > 8:
        raise SchemaValidateError("Unpacking is only supported for less than 8 bits")

    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1

    if packed_dim == 1:
        unpacked = torch.zeros((value.shape[0], value.shape[1] * pack_factor), device=value.device, dtype=torch.int32)
        for i in range(pack_factor):
            unpacked[:, i::pack_factor] = (value >> (num_bits * i)) & mask
        unpacked = unpacked[:, : int(shape[1])]
    else:
        unpacked = torch.zeros((value.shape[0] * pack_factor, value.shape[1]), device=value.device, dtype=torch.int32)
        for i in range(pack_factor):
            unpacked[i::pack_factor, :] = (value >> (num_bits * i)) & mask
        unpacked = unpacked[: int(shape[0]), :]

    return (unpacked - (1 << (num_bits - 1))).to(torch.int8)


def _unpack_from_int32_numpy(
    value: torch.Tensor, num_bits: int, shape: torch.Size, packed_dim: Union[Literal[0], Literal[1]] = 1
) -> torch.Tensor:
    if np is None:
        return _unpack_from_int32_torch(value, num_bits, shape, packed_dim)

    if value.dtype is not torch.int32:
        raise SchemaValidateError(f"Expected {torch.int32} but got {value.dtype}, Aborting unpack.")
    if num_bits > 8:
        raise SchemaValidateError("Unpacking is only supported for less than 8 bits")

    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1
    value_np = value.detach().cpu().numpy().astype(np.int32, copy=False)
    shifts = (np.arange(pack_factor, dtype=np.int32) * num_bits).reshape(1, 1, -1)

    if packed_dim == 1:
        unpacked = ((value_np[:, :, None] >> shifts) & mask).reshape(value_np.shape[0], value_np.shape[1] * pack_factor)
        unpacked = unpacked[:, : int(shape[1])]
    else:
        unpacked = ((value_np[:, None, :] >> shifts.transpose(0, 2, 1)) & mask).reshape(
            value_np.shape[0] * pack_factor, value_np.shape[1]
        )
        unpacked = unpacked[: int(shape[0]), :]

    unpacked = (unpacked - (1 << (num_bits - 1))).astype(np.int8, copy=False)
    return torch.from_numpy(unpacked)


def unpack_from_int32(
    value: torch.Tensor, num_bits: int, shape: torch.Size, packed_dim: Union[Literal[0], Literal[1]] = 1
) -> torch.Tensor:
    return _unpack_from_int32_numpy(value, num_bits, shape, packed_dim)


def weight_dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    m, n = weight.shape
    scale_m, scale_n = scale.shape
    if n % scale_n != 0:
        raise SchemaValidateError(f"N ({n}) is not divisible by K ({scale_n})")
    if scale_m != m:
        raise SchemaValidateError(f"Mismatch in scale rows ({scale_m}) and weight rows ({m}).")

    group_size = n // scale_n
    weight_f32 = weight.to(torch.float32)
    scale_f32 = scale.to(torch.float32)
    dequantized_weight = (weight_f32.reshape(m, scale_n, group_size) * scale_f32.unsqueeze(-1)).reshape(m, n)
    return dequantized_weight.to(torch.get_default_dtype())


@lru_cache(maxsize=1)
def get_int4_weight_map(model_path: str):
    model_index = json_safe_load(os.path.join(model_path, "model.safetensors.index.json"))
    return {
        k[: -len(".weight_packed")]: v for k, v in model_index["weight_map"].items() if k.endswith(".weight_packed")
    }


@lru_cache(maxsize=16)
def _load_int4_file_state_dict(model_path: str, file_name: str) -> Dict[str, torch.Tensor]:
    file_path = get_valid_read_path(os.path.join(model_path, file_name), "safetensors", size_max=MAX_READ_FILE_SIZE_32G)
    return load_file(file_path, device="cpu")


def auto_convert_module_int4_to_bf16(name: str, module: nn.Module, model_path: str):
    weight_map = get_int4_weight_map(model_path)
    if not weight_map:
        return
    try:
        sub_weight_map = {
            sub_name: weight_map[sub_name]
            for sub_name, _ in module.named_modules(prefix=name)
            if sub_name in weight_map
        }
        convert_module_int4_to_bf16(name, module, model_path, weight_map=sub_weight_map)
    except KeyError:
        get_logger().warning("Safetensors files not match index.json, please check whether model is of bf16.")
        get_logger().warning("Skip int4 to bf16.")


def replace_compressed_linear_with_bf16(root_module: nn.Module, root_prefix: str, model_path: str) -> nn.Module:
    def _to_linear(full_name: str, mod: nn.Module):
        expected_shape = (mod.out_features, mod.in_features)
        weight = None

        loaded_weight = load_tensor_by_full_name(model_path, f"{full_name}.weight")
        if loaded_weight is not None:
            weight = loaded_weight.to(torch.bfloat16)

        if weight is None and hasattr(mod, "weight") and isinstance(mod.weight, torch.Tensor):
            if tuple(mod.weight.shape) == expected_shape:
                weight = mod.weight.detach().to(torch.bfloat16)

        if weight is None and all(hasattr(mod, k) for k in ("weight_packed", "weight_scale", "weight_shape")):
            packed = mod.weight_packed.detach().to(torch.int32)
            shape = (
                mod.weight_shape.detach()
                if isinstance(mod.weight_shape, torch.Tensor)
                else torch.tensor(mod.weight_shape)
            )
            scale = mod.weight_scale.detach()
            unpacked = unpack_from_int32(packed, num_bits=4, shape=shape, packed_dim=1)
            weight = weight_dequant(unpacked, scale).to(torch.bfloat16)

        if weight is None or tuple(weight.shape) != expected_shape:
            return None

        loaded_bias = load_tensor_by_full_name(model_path, f"{full_name}.bias")
        if loaded_bias is not None:
            bias = loaded_bias.to(torch.bfloat16)
        elif getattr(mod, "bias", None) is not None and isinstance(mod.bias, torch.Tensor):
            bias = mod.bias.detach().to(torch.bfloat16)
        else:
            bias = None

        new_linear = nn.Linear(
            in_features=mod.in_features,
            out_features=mod.out_features,
            bias=bias is not None,
            device=weight.device,
            dtype=torch.bfloat16,
        )
        new_linear.weight.data.copy_(weight)
        if bias is not None:
            new_linear.bias.data.copy_(bias)
        new_linear.eval()
        return new_linear

    if root_module.__class__.__name__ == "CompressedLinear":
        new_root = _to_linear(root_prefix, root_module)
        return new_root if new_root is not None else root_module

    targets = [
        (name, mod)
        for name, mod in root_module.named_modules()
        if name and mod.__class__.__name__ == "CompressedLinear"
    ]
    for name, mod in targets:
        full_name = f"{root_prefix}.{name}"
        new_linear = _to_linear(full_name, mod)
        if new_linear is None:
            continue
        root_module.set_submodule(name, new_linear)

    return root_module


@torch.no_grad()
def convert_module_int4_to_bf16(name: str, module: nn.Module, model_path: str, weight_map: Dict[str, str]):
    target_sub_modules = {
        sub_name: sub_module for sub_name, sub_module in module.named_modules(prefix=name) if sub_name in weight_map
    }
    file_to_sub_names: Dict[str, List[str]] = defaultdict(list)
    for sub_name in target_sub_modules:
        file_to_sub_names[weight_map[sub_name]].append(sub_name)

    with tqdm(total=len(target_sub_modules), desc="int4 to bf16") as bars:
        for file_name, sub_names in file_to_sub_names.items():
            file_state = _load_int4_file_state_dict(model_path, file_name)
            for sub_name in sub_names:
                sub_module = target_sub_modules[sub_name]
                packed_weight = file_state[f"{sub_name}.weight_packed"]
                scale = file_state[f"{sub_name}.weight_scale"]
                shape = file_state[f"{sub_name}.weight_shape"]

                unpacked_weight = unpack_from_int32(
                    packed_weight if packed_weight.dtype is torch.int32 else packed_weight.to(torch.int32),
                    num_bits=4,
                    shape=shape,
                    packed_dim=1,
                )
                dequant_weight = weight_dequant(unpacked_weight, scale)

                if sub_module.__class__.__name__ == "CompressedLinear":
                    bias_data = None
                    if getattr(sub_module, "bias", None) is not None:
                        bias_data = sub_module.bias.detach().to(torch.get_default_dtype())
                    new_linear = nn.Linear(
                        sub_module.in_features,
                        sub_module.out_features,
                        bias=bias_data is not None,
                        device=dequant_weight.device,
                        dtype=torch.get_default_dtype(),
                    )
                    new_linear.weight.data.copy_(dequant_weight)
                    if bias_data is not None:
                        new_linear.bias.data.copy_(bias_data)
                    new_linear.eval()
                    relative_name = sub_name[len(name) + 1 :] if name else sub_name
                    module.set_submodule(relative_name, new_linear)
                else:
                    target_weight = sub_module.weight
                    sub_module.weight[:] = dequant_weight.to(target_weight.device, dtype=target_weight.dtype)

                del packed_weight, scale, shape, unpacked_weight, dequant_weight
                if "bias_data" in locals():
                    del bias_data
                if "new_linear" in locals():
                    del new_linear
                bars.update(1)

            del file_state
            gc.collect()
            if npu_available:
                try:
                    torch.npu.empty_cache()
                except Exception as exc:
                    get_logger().warning("Failed to clear NPU cache: %s", exc)

    cache_clear = getattr(_load_int4_file_state_dict, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()
