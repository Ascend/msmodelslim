# Portions of this file are derived from compressed_tensors.quantization.quant_scheme
# (https://github.com/vllm-project/compressed-tensors).
# See msmodelslim/Third_Party_Open_Source_Software_Notice.
#
# Copyright (c) 2021-present Neuralmagic Inc. All Rights Reserved.
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
#
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
"""
IR edge: INT4_PACKED -> FLOAT (bf16 ``nn.Linear``).

The packed format stores 4-bit signed weights in int32 words together with
per-group scales and the original 2-D weight shape.
"""

from __future__ import annotations

import torch
from torch import nn

from msmodelslim.core.quant_service.modelslim_convert.virtual_module import ModelFreeLinear
from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.convert.types import IRKind, LossLevel
from msmodelslim.processor.convert.base import BaseConvertProcessor
from msmodelslim.utils.exception import SchemaValidateError


class Int4PackedToFloatProcessor(BaseConvertProcessor):
    name = "Int4PackedToFloatProcessor"
    src_ir = IRKind.INT4_PACKED
    dst_ir = IRKind.FLOAT
    loss_level = LossLevel.LOSSLESS.value

    def transform(self, module: nn.Module, context: ConvertContext) -> nn.Module:
        if not isinstance(module, ModelFreeLinear):
            return module
        if not module.lazy_initialized:
            module.lazy_init(context.reader, device="cpu")

        packed = module._buffers.get("weight_packed")
        scale = module._buffers.get("weight_scale")
        shape = module._buffers.get("weight_shape")
        if packed is None or scale is None or shape is None:
            return module

        unpacked = _unpack_from_int32(
            packed if packed.dtype is torch.int32 else packed.to(torch.int32),
            num_bits=4,
            shape=shape,
            packed_dim=1,
        )
        weight_bf16 = _dequant_per_group(unpacked, scale).to(torch.bfloat16)

        bias = getattr(module, "bias", None)
        out = nn.Linear(weight_bf16.shape[1], weight_bf16.shape[0], bias=bias is not None)
        out.weight = nn.Parameter(weight_bf16, requires_grad=False)
        if bias is not None:
            out.bias = nn.Parameter(bias.detach().to(torch.bfloat16), requires_grad=False)
        return out


def _unpack_from_int32(
    value: torch.Tensor,
    num_bits: int,
    shape: torch.Tensor | torch.Size | tuple[int, ...],
    packed_dim: int = 1,
) -> torch.Tensor:
    if value.dtype is not torch.int32:
        raise SchemaValidateError(f"Expected {torch.int32} but got {value.dtype}, Aborting unpack.")
    if num_bits > 8:
        raise SchemaValidateError("Unpacking is only supported for less than 8 bits")

    target_shape = _shape_to_tuple(shape)
    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1

    if packed_dim == 1:
        unpacked = torch.zeros(
            (value.shape[0], value.shape[1] * pack_factor),
            device=value.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked[:, i::pack_factor] = (value >> (num_bits * i)) & mask
        unpacked = unpacked[:, : target_shape[1]]
    else:
        unpacked = torch.zeros(
            (value.shape[0] * pack_factor, value.shape[1]),
            device=value.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked[i::pack_factor, :] = (value >> (num_bits * i)) & mask
        unpacked = unpacked[: target_shape[0], :]

    return (unpacked - (1 << (num_bits - 1))).to(torch.int8)


def _dequant_per_group(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    m, n = weight.shape
    scale_m, scale_n = scale.shape
    if n % scale_n != 0:
        raise SchemaValidateError(f"N ({n}) is not divisible by K ({scale_n})")
    if scale_m != m:
        raise SchemaValidateError(f"Mismatch in scale rows ({scale_m}) and weight rows ({m}).")

    group_size = n // scale_n
    return (weight.to(torch.float32).reshape(m, scale_n, group_size) * scale.to(torch.float32).unsqueeze(-1)).reshape(
        m, n
    )


def _shape_to_tuple(shape: torch.Tensor | torch.Size | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(shape, torch.Tensor):
        return tuple(int(v) for v in shape.detach().cpu().reshape(-1).tolist())
    return tuple(int(v) for v in shape)
