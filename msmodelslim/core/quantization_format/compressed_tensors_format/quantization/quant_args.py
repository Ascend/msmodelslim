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
# Portions of this file are derived from compressed_tensors.quantization.quant_args
# (https://github.com/neuralmagic/compressed-tensors).
# See msmodelslim/Third_Party_Open_Source_Software_Notice.
#
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.

import warnings
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


# Minimal shims forked from compressed_tensors.utils / quant_args; see Third_Party notice.
TorchDtype = Union[str, torch.dtype]


class Aliasable:
    @staticmethod
    def get_aliases() -> Dict[str, str]:
        return {}


class _FP8E4M3Data:
    dtype = getattr(torch, "float8_e4m3fn", torch.float16)


FP8_E4M3_DATA = _FP8E4M3Data()


__all__ = [
    "QuantizationType",
    "QuantizationStrategy",
    "QuantizationArgs",
]


class QuantizationType(str, Enum):
    """
    Enum storing quantization type options
    """

    INT = "int"
    FLOAT = "float"


class QuantizationStrategy(str, Enum):
    """
    Enum storing quantization strategy options
    """

    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"
    TOKEN = "token"  # nosec B105 - quantization strategy name, not a credential
    TENSOR_GROUP = "tensor_group"
    ATTN_HEAD = "attn_head"


class DynamicType(str, Enum):
    """
    Enum storing potential dynamic types.

    1. If dynamic is True, all quantization parameters are generated on the fly.
    2. If dynamic is False, all quantization parameters generated are static.
    3. If "local" is provided, only local quantization parameters are dynamic.

    Note: "local" is only currently supported for NVFP4.

    """

    LOCAL = "local"


class ActivationOrdering(Aliasable, str, Enum):
    """
    Enum storing strategies for activation ordering

    Group: reorder groups and weight\n
    Weight: only reorder weight, not groups. Slightly lower accuracy but also lower
    latency when compared to group actorder\n
    Dynamic: alias for Group\n
    Static: alias for Weight\n
    """

    GROUP = "group"
    WEIGHT = "weight"
    # aliases
    DYNAMIC = "dynamic"
    STATIC = "static"

    @staticmethod
    def get_aliases() -> Dict[str, str]:
        return {
            "dynamic": "group",
            "static": "weight",
        }


class QuantizationArgs(BaseModel, use_enum_values=True):
    """
    User facing arguments used to define a quantization config for weights or
    activations

    :param num_bits: quantization bit depth
    :param type: dtype to quantized to, either int or float
    :param symmetric: whether or not quantization scale is symmetric about zero-point
    :param strategy: string id determining the scope of scale/zero-point to apply
    :param group_size: group length to use for the group strategy
    :param block_structure: 2d block structure to use for the block strategy; must be
        a list of two ints [rows, cols] like [128, 128].
    :param dynamic: set True to perform dynamic quantization - values will not be
        calibrated during calibration phase, instead during inference new quantization
        ranges will be observed with every sample. Defaults to False for static
        quantization. Note that enabling dynamic quantization will change the default
        observer to a memoryless one
    :param actorder: whether to apply group quantization in decreasing order of
        activation. Defaults to None for arbitrary ordering
    """

    num_bits: int = 8
    type: QuantizationType = QuantizationType.INT
    symmetric: bool = True
    group_size: Optional[int] = None
    strategy: Optional[QuantizationStrategy] = None
    block_structure: Optional[List[int]] = None
    dynamic: Union[DynamicType, bool] = False
    actorder: Optional[Union[ActivationOrdering, bool]] = None
    scale_dtype: Optional[TorchDtype] = None
    zp_dtype: Optional[TorchDtype] = None
    observer: Optional[str] = Field(
        default=None,
        description=(
            "Determines the method of computing quantization parameters (scales and "
            "zero-points). Defaults to min-max when not using dynamic quantization"
        ),
    )
    observer_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "optional dict of kwargs to be passed directly to torch quantization "
            "Observers constructor excluding quantization range or symmetry"
        ),
    )

    @field_serializer("zp_dtype")
    def serialize_dtype(self, dtype: torch.dtype):
        if self.symmetric:
            return None
        return str(dtype)

    @field_validator("type", mode="before")
    @classmethod
    def validate_type(cls, value) -> QuantizationType:
        if isinstance(value, str):
            return QuantizationType(value.lower())

        return value

    @field_validator("group_size", mode="before")
    @classmethod
    def validate_group(cls, value) -> Optional[int]:
        if value is None:
            return value

        if value < -1:
            raise ValueError(
                f"Invalid group size {value}. Use group_size > 0 for strategy='group' and group_size = -1 for 'channel'"
            )

        return value

    @field_validator("block_structure", mode="before")
    @classmethod
    def validate_block_structure(cls, value) -> Optional[List[int]]:
        if value is None:
            return value
        # For backward compatibility, allow string format "2x4", "8x16", etc.
        if isinstance(value, str):
            try:
                return [int(x) for x in value.split("x")]
            except Exception:
                raise ValueError(f"Invalid block_structure '{value}'. Must be a list of ints [rows, cols].")
        if isinstance(value, (list, tuple)):
            if len(value) != 2 or not all(isinstance(v, int) for v in value):
                raise ValueError(f"Invalid block_structure '{value}'. Must be a list of ints [rows, cols].")
            return list(value)
        raise ValueError(f"Invalid block_structure '{value}'. Must be a list of ints [rows, cols].")

    @field_validator("strategy", mode="before")
    @classmethod
    def validate_strategy(cls, value) -> Optional[QuantizationStrategy]:
        if isinstance(value, str):
            return QuantizationStrategy(value.lower())

        return value

    @field_validator("actorder", mode="before")
    @classmethod
    def validate_actorder(cls, value) -> Optional[ActivationOrdering]:
        if isinstance(value, bool):
            return ActivationOrdering.GROUP if value else None

        if isinstance(value, str):
            return ActivationOrdering(value.lower())

        return value

    @field_validator("dynamic", mode="before")
    @classmethod
    def validate_dynamic(cls, value) -> Union[DynamicType, bool]:
        if isinstance(value, str):
            return DynamicType(value.lower())
        return value

    @model_validator(mode="after")
    def validate_model_after(self) -> "QuantizationArgs":
        # extract user-passed values from dictionary
        strategy = self.strategy
        group_size = self.group_size
        block_structure = self.block_structure
        actorder = self.actorder
        dynamic = self.dynamic
        observer = self.observer
        dynamic = self.dynamic
        zp_dtype = self.zp_dtype

        # infer strategy
        if strategy is None:
            if group_size is None:
                strategy = QuantizationStrategy.TENSOR
            elif group_size > 0:
                strategy = QuantizationStrategy.GROUP
            elif group_size == -1:
                strategy = QuantizationStrategy.CHANNEL
            else:
                raise ValueError(
                    f"Invalid group size {group_size}. Use group_size > 0 for "
                    "strategy='group' and group_size = -1 for 'channel'"
                )

        # validate token strategy
        if strategy == QuantizationStrategy.TOKEN and not dynamic:
            raise ValueError("Cannot perform static token quantization, please use `dynamic=True`")

        # validate group strategy
        if strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
            if group_size is None or group_size <= 0:
                raise ValueError(f"strategy {strategy} requires group_size to be set to a positive value")
        if (
            group_size is not None
            and group_size > 0
            and strategy not in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP)
        ):
            raise ValueError("group_size requires strategy to be set to 'group'")

        # validate block strategy
        has_block_strategy = strategy == QuantizationStrategy.BLOCK
        has_block_structure = block_structure is not None
        if has_block_strategy and not has_block_structure:
            raise ValueError(f"Block strategy requires block structure\n{self}")
        if has_block_structure and not has_block_strategy:
            raise ValueError(f"Block structure requires block strategy\n{self}")

        # validate activation ordering and strategy
        if actorder is not None and strategy not in (
            QuantizationStrategy.GROUP,
            QuantizationStrategy.TENSOR_GROUP,
        ):
            raise ValueError(
                "Must use group or tensor_group quantization strategy in order to apply activation ordering"
            )

        # infer observer w.r.t. dynamic
        if dynamic:
            supported_strategies = (
                QuantizationStrategy.TOKEN,
                QuantizationStrategy.TENSOR,
                QuantizationStrategy.TENSOR_GROUP,
                QuantizationStrategy.GROUP,
            )
            if strategy not in supported_strategies:
                raise ValueError(f"One of {supported_strategies} must be used for dynamic quant.")

            if dynamic == DynamicType.LOCAL and strategy != QuantizationStrategy.TENSOR_GROUP:
                raise ValueError("local is only supported for strategy tensor_group")

            if observer is not None:
                if dynamic is True:  # checking if dynamic is True, not "local"
                    if observer != "memoryless":  # avoid annoying users with old configs
                        warnings.warn("No observer is used for dynamic quant., setting to None")
                    observer = None
            else:
                if dynamic == DynamicType.LOCAL:
                    observer = "minmax"

        elif observer is None:
            # default to minmax for non-dynamic cases
            observer = "memoryless_minmax"

        if zp_dtype is None:
            if self.num_bits == 4 and self.type == QuantizationType.FLOAT:
                zp_dtype = FP8_E4M3_DATA.dtype
            else:
                zp_dtype = self.pytorch_dtype()

        # write back modified values
        self.strategy = strategy
        self.observer = observer
        self.zp_dtype = zp_dtype
        return self

    def pytorch_dtype(self) -> torch.dtype:
        if self.type == QuantizationType.FLOAT:
            if self.num_bits == 8:
                return FP8_E4M3_DATA.dtype
            else:
                raise NotImplementedError("Only num_bits in (8) are supported")
        elif self.type == QuantizationType.INT:
            if self.num_bits <= 8:
                return torch.int8
            elif self.num_bits <= 16:
                return torch.int16
            else:
                return torch.int32
        else:
            raise ValueError(f"Invalid quantization type {self.type}")

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
