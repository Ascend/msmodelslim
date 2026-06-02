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
# Portions of this file are derived from compressed_tensors.quantization.quant_scheme
# (https://github.com/vllm-project/compressed-tensors).
# See msmodelslim/Third_Party_Open_Source_Software_Notice.
#
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.

"""compressed-tensors QuantizationScheme 预设，与 msmodelslim QIR 模块一一对应。

每个 PRESET 名称对应 ``msmodelslim.ir`` 中一个 ``AutoFakeQuantLinear`` 子类；
``QIR_MODULE_PRESET_MAP`` 提供 QIR 类型 → preset 名称的反向索引。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Type
from msmodelslim import logger
from msmodelslim.format.compressed_tensors_format.config.base import QuantizationFormat
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.format.compressed_tensors_format.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from pydantic import BaseModel, ConfigDict, model_validator
from torch import nn

import msmodelslim.ir as qir

# MX block 量化默认 block/group 大小，与 QDType.mx_finfo.block_size 一致。
_MX_BLOCK_SIZE = 32
# INT4 group 量化常用默认 group_size（可被模块上的 group_size 覆盖）。
_INT4_GROUP_SIZE = 128

__all__ = [
    "QuantizationScheme",
    "preset_name_to_scheme",
    "scheme_for_qir_module",
]


class QuantizationScheme(BaseModel):
    """
    compressed-tensors 层组量化方案，描述 weights / input_activations 的 QuantizationArgs。

    :param targets: 目标模块列表，通常为 ``["Linear"]``
    :param weights: 权重量化参数
    :param input_activations: 输入激活量化参数；仅权重量化时为 ``None``
    :param output_activations: 输出激活量化参数
    :param format: 层压缩格式
    """

    targets: List[str]
    weights: Optional[QuantizationArgs] = None
    input_activations: Optional[QuantizationArgs] = None
    output_activations: Optional[QuantizationArgs] = None
    format: Optional[QuantizationFormat] = None

    @model_validator(mode="after")
    def validate_model_after(self) -> "QuantizationScheme":
        inputs = self.input_activations
        outputs = self.output_activations
        weights = self.weights
        fmt = self.format

        if inputs is not None:
            if inputs.strategy == QuantizationStrategy.GROUP and inputs.dynamic is True:
                raise NotImplementedError("Static and local group-wise activation quantization is not supported")

            if inputs.strategy not in (
                QuantizationStrategy.TOKEN,
                QuantizationStrategy.TENSOR,
                QuantizationStrategy.GROUP,
                QuantizationStrategy.TENSOR_GROUP,
                QuantizationStrategy.ATTN_HEAD,
            ):
                raise NotImplementedError(
                    f"Using {inputs.strategy} strategy is not supported for activation quantization"
                )

            if inputs.actorder is not None:
                raise ValueError("Cannot apply actorder to input activations")

        if outputs is not None and outputs.actorder is not None:
            raise ValueError("Cannot apply actorder to output activations")

        if fmt == QuantizationFormat.mixed_precision:
            raise ValueError("mixed-precision cannot be set as a format for a QuantizationScheme")

        if (
            inputs
            and weights
            and weights.strategy == QuantizationStrategy.GROUP
            and inputs.strategy == QuantizationStrategy.GROUP
            and weights.group_size != inputs.group_size
        ):
            logger.warning(
                "GROUP strategy on weights and input_activations with different "
                "group sizes (%s vs %s) may complicate fused kernels; consider "
                "TENSOR_GROUP or matching group sizes",
                weights.group_size,
                inputs.group_size,
            )

        return self

    model_config = ConfigDict(extra="forbid")


def scheme_for_qir_module(module: nn.Module, targets: Optional[List[str]] = None) -> Optional[QuantizationScheme]:
    """由 QIR 模块实例构造对应的 ``QuantizationScheme``。"""
    preset = QIR_MODULE_PRESET_MAP.get(type(module))
    if preset is None:
        return None
    return preset_name_to_scheme(preset, targets or ["Linear"])


def preset_name_to_scheme(name: str, targets: List[str]) -> QuantizationScheme:
    """由 preset 名称构造 ``QuantizationScheme``。"""
    name = name.upper()
    if name not in PRESET_SCHEMES:
        available = list(PRESET_SCHEMES.keys())
        raise SchemaValidateError(
            f"Unknown preset scheme name {name}, available names: {available}",
            action="Check QIR_MODULE_PRESET_MAP and PRESET_SCHEMES configuration.",
        )
    scheme_args = deepcopy(PRESET_SCHEMES[name])
    return QuantizationScheme(targets=targets, **scheme_args)


# ---------------------------------------------------------------------------
# QIR 对齐的 preset 定义
# ---------------------------------------------------------------------------

UNQUANTIZED: dict = {}

# W8A8StaticFakeQuantLinear: int8_per_tensor(asym) × int8_per_channel(sym)
W8A8_STATIC = dict(
    format=QuantizationFormat.int_quantized,
    weights=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.CHANNEL,
        symmetric=True,
        dynamic=False,
    ),
    input_activations=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.TENSOR,
        symmetric=False,
        dynamic=False,
    ),
)

# W8A8DynamicPerChannelFakeQuantLinear: int8_per_token(sym) × int8_per_channel(sym)
W8A8_DYNAMIC = dict(
    format=QuantizationFormat.int_quantized,
    weights=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.CHANNEL,
        symmetric=True,
        dynamic=False,
    ),
    input_activations=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.TOKEN,
        symmetric=True,
        dynamic=True,
    ),
)

PRESET_SCHEMES: Dict[str, dict] = {
    "UNQUANTIZED": UNQUANTIZED,
    # W8A8 系列
    "W8A8_STATIC": W8A8_STATIC,
    "W8A8_DYNAMIC": W8A8_DYNAMIC,
}

QIR_MODULE_PRESET_MAP: Dict[Type[nn.Module], str] = {
    qir.W8A8StaticFakeQuantLinear: "W8A8_STATIC",
    qir.W8A8DynamicPerChannelFakeQuantLinear: "W8A8_DYNAMIC",
}
