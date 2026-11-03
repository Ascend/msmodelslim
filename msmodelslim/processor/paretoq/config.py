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
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

from typing import Any, Final, Literal, Mapping

from pydantic import Field, field_validator, model_validator

from msmodelslim.core.train.task import TrainAlgorithmConfig
from msmodelslim.processor.paretoq.core import WEIGHT_CLIP_INIT_AMAX
from msmodelslim.utils.exception import SchemaValidateError

# 与 ``fake_quant_weight`` / convert 边已实现范围一致；YAML 误配时尽早拒绝。
SUPPORTED_W_BITS: Final = frozenset({0, 1, 2, 3, 4})
SUPPORTED_A_BITS: Final = frozenset({4, 8})
_DEPLOY_ACT_TO_A_BITS: Final = {"a4": 4, "a8": 8}
_A_BITS_TO_DEPLOY_ACT: Final[Mapping[int, Literal["a4", "a8"]]] = {4: "a4", 8: "a8"}


class ParetoQConfig(TrainAlgorithmConfig):
    """YAML ``algorithm[]`` item with ``type: paretoq``."""

    type: Literal["paretoq"] = Field(default="paretoq", description="训练量化算法类型，固定为 paretoq。")
    w_bits: int = Field(
        default=4,
        description=(f"权重量化比特宽，由外部 YAML 传入；仅支持 {sorted(SUPPORTED_W_BITS)}（0/2→SEQ，1/3/4→LSQ）。"),
    )
    a_bits: int = Field(
        default=8,
        description=(f"激活量化比特宽，由外部 YAML 传入；仅支持 {sorted(SUPPORTED_A_BITS)}（4→W4A4，8→W4A8）。"),
    )
    weight_scope: Literal["per_channel", "layerwise"] = Field(
        default="per_channel",
        description="权重量化粒度：per_channel 或 layerwise。",
    )
    weight_clip_init: Literal["amax", "mean2_sqrt_qp"] = Field(
        default=WEIGHT_CLIP_INIT_AMAX,
        description="可学习 α（weight_clip_val）初始化策略。",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_act_bits(cls, data: Any) -> Any:
        """兼容旧字段 ``deploy_act``（a4/a8）→ ``a_bits``。"""
        if not isinstance(data, Mapping):
            return data
        payload = dict(data)
        deploy_act = payload.pop("deploy_act", None)
        if deploy_act is None:
            return payload
        if deploy_act not in _DEPLOY_ACT_TO_A_BITS:
            raise SchemaValidateError(
                f"Unsupported ParetoQ deploy_act={deploy_act!r}",
                action=("Please set algorithm.a_bits to 4 or 8 in YAML (legacy deploy_act: 'a4' or 'a8')."),
            )
        mapped = _DEPLOY_ACT_TO_A_BITS[deploy_act]
        if "a_bits" in payload and int(payload["a_bits"]) != mapped:
            raise SchemaValidateError(
                f"Conflicting activation bits: a_bits={payload['a_bits']} vs deploy_act={deploy_act!r}",
                action="Please specify only one of a_bits or deploy_act in the YAML algorithm config.",
            )
        payload["a_bits"] = mapped
        return payload

    @field_validator("w_bits")
    @classmethod
    def _validate_w_bits(cls, value: int) -> int:
        if value not in SUPPORTED_W_BITS:
            raise SchemaValidateError(
                f"Unsupported ParetoQ w_bits={value}",
                action=(
                    f"Please set algorithm.w_bits in YAML to one of {sorted(SUPPORTED_W_BITS)}; "
                    "other bit-widths are not implemented."
                ),
            )
        return value

    @field_validator("a_bits")
    @classmethod
    def _validate_a_bits(cls, value: int) -> int:
        if value not in SUPPORTED_A_BITS:
            raise SchemaValidateError(
                f"Unsupported ParetoQ a_bits={value}",
                action=(f"Please set algorithm.a_bits in YAML to one of {sorted(SUPPORTED_A_BITS)} (4→W4A4, 8→W4A8)."),
            )
        return value

    @property
    def deploy_act(self) -> Literal["a4", "a8"]:
        """部署侧激活标签，由 ``a_bits`` 派生（供壳模块 / convert meta 使用）。"""
        return _A_BITS_TO_DEPLOY_ACT[self.a_bits]
