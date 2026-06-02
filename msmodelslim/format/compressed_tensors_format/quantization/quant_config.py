#!/usr/bin/env python
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
#
# Portions of this file are derived from compressed_tensors.quantization.quant_config
# (https://github.com/neuralmagic/compressed-tensors).
# See msmodelslim/Third_Party_Open_Source_Software_Notice.
#
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.

"""与 ``compressed_tensors.quantization.quant_config`` 对齐的量化配置模型。"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from msmodelslim import logger
from msmodelslim.format.compressed_tensors_format.config.base import (
    COMPRESSION_VERSION_NAME,
    DEFAULT_COMPRESSION_VERSION,
    QUANTIZATION_METHOD_NAME,
    SPARSITY_CONFIG_NAME,
    TRANSFORM_CONFIG_NAME,
    QuantizationFormat,
)
from msmodelslim.format.compressed_tensors_format.quantization.quant_config_builder import (
    infer_config_groups,
    infer_ignore,
    infer_root_format,
)
from msmodelslim.format.compressed_tensors_format.quantization.quant_args import QuantizationArgs
from msmodelslim.format.compressed_tensors_format.quantization.quant_scheme import (
    QuantizationScheme,
    preset_name_to_scheme,
)
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

__all__ = [
    "QuantizationStatus",
    "QuantizationConfig",
    "DEFAULT_QUANTIZATION_METHOD",
]

DEFAULT_QUANTIZATION_METHOD = "compressed-tensors"


class QuantizationStatus(str, Enum):
    INITIALIZED = "initialized"
    CALIBRATION = "calibration"
    FROZEN = "frozen"
    COMPRESSED = "compressed"


class QuantizationConfig(BaseModel):
    """
    compressed-tensors 量化配置，结构与官方 ``QuantizationConfig`` 一致。
    通过 ``from_model`` 从量化后的 QIR 模型反向推导全部字段。
    """

    config_groups: Dict[str, Union[QuantizationScheme, List[str]]]
    quant_method: str = DEFAULT_QUANTIZATION_METHOD
    format: Union[str, QuantizationFormat] = QuantizationFormat.int_quantized
    quantization_status: QuantizationStatus = QuantizationStatus.COMPRESSED
    global_compression_ratio: Optional[float] = None
    ignore: Optional[List[str]] = Field(default_factory=list)
    # kv cache now not supported
    kv_cache_scheme: Optional[QuantizationArgs] = None

    model_config = ConfigDict(use_enum_values=True)

    def model_post_init(self, __context: Any) -> None:
        for group_name, targets_or_scheme in self.config_groups.items():
            if isinstance(targets_or_scheme, QuantizationScheme):
                continue
            self.config_groups[group_name] = preset_name_to_scheme(
                name=group_name,
                targets=targets_or_scheme,
            )

    @classmethod
    def from_model(cls, model: nn.Module) -> Optional["QuantizationConfig"]:
        """从量化后的模型反向推导完整的 ``QuantizationConfig``。"""
        # 暂不支持kv cache量化, 先设置为None; todo: 支持kv cache量化
        config_groups = infer_config_groups(model)
        if not config_groups:
            logger.warning("No quantized modules found; config groups inference skipped")
            return None

        return cls(
            config_groups=config_groups,
            kv_cache_scheme=None,
            format=infer_root_format(config_groups),
            quantization_status=QuantizationStatus.COMPRESSED,
            ignore=infer_ignore(model),
        )

    def to_quantization_config_dict(self) -> Dict[str, Any]:
        """生成写入 ``config.json`` 的 ``quantization_config`` 字段内容。"""
        qconfig_data = self.model_dump(exclude={"quant_method"})
        return {
            COMPRESSION_VERSION_NAME: DEFAULT_COMPRESSION_VERSION,
            QUANTIZATION_METHOD_NAME: DEFAULT_QUANTIZATION_METHOD,
            SPARSITY_CONFIG_NAME: {},
            TRANSFORM_CONFIG_NAME: {},
            **qconfig_data,
        }
