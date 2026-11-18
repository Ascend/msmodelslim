#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from msmodelslim.core.train.task import TrainAlgorithmConfig
from msmodelslim.utils.exception import SchemaValidateError

WeightDType = Literal["int0", "int1", "int2", "int3", "int4"]
ActDType = Literal["int4", "int8"]
WeightScope = Literal["per_channel", "layer_wise"]
ActScope = Literal["per_token", "per_tensor"]
WeightQParamInit = Literal["amax", "mean2_sqrt_qp"]
ActQParamInit = Literal["amax"]

_WEIGHT_BITS: Final[dict[str, int]] = {f"int{bits}": bits for bits in range(5)}
_ACT_SCOPE: Final[dict[str, ActScope]] = {
    "int4": "per_token",
    "int8": "per_token",
}


class DeployAct(str, Enum):
    """ParetoQ 部署期激活模式，同时是 checkpoint 中 ``deploy_act`` 的取值。"""

    A4 = "a4"
    A8 = "a8"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def parse(cls, value: Any) -> "DeployAct":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as exc:
            choices = ", ".join(mode.value for mode in cls)
            raise ValueError(f"Unsupported ParetoQ deploy_act={value!r}; expected one of: {choices}") from exc


_ACT_DEPLOY: Final[dict[str, DeployAct]] = {
    "int4": DeployAct.A4,
    "int8": DeployAct.A8,
}


@dataclass(frozen=True)
class ParetoQQuantSpec:
    """ParetoQ 训练壳与部署共用的完整量化规格。"""

    w_bits: int
    weight_layerwise: bool
    deploy_act: DeployAct

    def __post_init__(self) -> None:
        object.__setattr__(self, "deploy_act", DeployAct.parse(self.deploy_act))
        if self.w_bits not in (0, 1, 2, 3, 4):
            raise ValueError(f"Unsupported ParetoQ weight bits={self.w_bits}; expected 0..4")


class ParetoQWeightConfig(BaseModel):
    """对应 YAML ``algorithm[].weight``。"""

    model_config = ConfigDict(extra="forbid")

    dtype: WeightDType = Field(
        default="int4",
        description="权重量化数据类型，支持 `int0`、`int1`、`int2`、`int3`、`int4`。",
    )
    scope: WeightScope = Field(
        default="per_channel",
        description="权重量化粒度，支持 `per_channel` 或 `layer_wise`。",
    )
    symmetric: bool = Field(
        default=True,
        description="是否使用对称权重量化；ParetoQ 当前仅支持 `true`。",
    )
    qparam_init: WeightQParamInit = Field(
        default="amax",
        description="可学习权重裁剪参数 `weight_clip_val` 的初始化策略，支持 `amax` 或 `mean2_sqrt_qp`。",
    )

    @field_validator("symmetric")
    @classmethod
    def _validate_symmetric(cls, value: bool) -> bool:
        if not value:
            raise SchemaValidateError(
                "ParetoQ weight.symmetric must be true",
                action="Please set algorithm.weight.symmetric to true.",
            )
        return value

    @property
    def w_bits(self) -> int:
        return _WEIGHT_BITS[self.dtype]

    @property
    def layerwise(self) -> bool:
        return self.scope == "layer_wise"


class ParetoQActConfig(BaseModel):
    """对应 YAML ``algorithm[].act``。"""

    model_config = ConfigDict(extra="forbid")

    dtype: ActDType = Field(
        default="int8",
        description="激活量化数据类型，支持 `int4`（A4）或 `int8`（A8）。",
    )
    scope: Optional[ActScope] = Field(
        default=None,
        description="激活量化粒度；A4/A8 仅支持 `per_token`，省略时按 dtype 自动推导。",
    )
    symmetric: bool = Field(
        default=True,
        description="是否使用对称激活量化；ParetoQ 当前仅支持 `true`。",
    )
    qparam_init: ActQParamInit = Field(
        default="amax",
        description="激活量化参数初始化策略，当前仅支持 `amax`。",
    )

    @field_validator("symmetric")
    @classmethod
    def _validate_symmetric(cls, value: bool) -> bool:
        if not value:
            raise SchemaValidateError(
                "ParetoQ act.symmetric must be true",
                action="Please set algorithm.act.symmetric to true.",
            )
        return value

    @model_validator(mode="after")
    def _validate_scope(self):
        expected = _ACT_SCOPE[self.dtype]
        if self.scope is None:
            self.scope = expected
        elif self.scope != expected:
            raise SchemaValidateError(
                f"ParetoQ act.scope={self.scope!r} is incompatible with act.dtype={self.dtype!r}",
                action=f"Please set algorithm.act.scope to {expected!r} for dtype {self.dtype!r}.",
            )
        return self

    @property
    def deploy_act(self) -> DeployAct:
        return _ACT_DEPLOY[self.dtype]


class ParetoQConfig(TrainAlgorithmConfig):
    """YAML ``algorithm[]`` item with ``type: paretoq``."""

    type: Literal["paretoq"] = Field(
        default="paretoq",
        description="训练量化算法类型，固定为 `paretoq`。",
    )
    weight: ParetoQWeightConfig = Field(
        description="权重量化配置，包含 dtype、scope、symmetric 与 qparam_init。",
    )
    act: ParetoQActConfig = Field(
        description="激活量化配置，包含 dtype、scope、symmetric 与 qparam_init。",
    )

    @model_validator(mode="after")
    def _validate_quant_spec(self) -> "ParetoQConfig":
        try:
            self.quant_spec
        except ValueError as exc:
            raise SchemaValidateError(
                f"Invalid ParetoQ quantization config: {exc}",
                action="Please check algorithm.weight and algorithm.act.",
            ) from exc
        return self

    @property
    def quant_spec(self) -> ParetoQQuantSpec:
        """将 weight/act 两段 YAML 配置折算成模块与部署共用的单一规格。"""
        return ParetoQQuantSpec(
            w_bits=self.weight.w_bits,
            weight_layerwise=self.weight.layerwise,
            deploy_act=self.act.deploy_act,
        )
