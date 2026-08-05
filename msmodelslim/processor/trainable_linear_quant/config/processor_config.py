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

# Trainable linear quant processor configuration.

from __future__ import annotations

from typing import Any, List, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator, SerializeAsAny

from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.processor.base import AutoProcessorConfig
from msmodelslim.utils.exception import SchemaValidateError

from msmodelslim.processor.trainable_linear_quant.core.ops import (
    MinmaxTuneOpConfig,
    RoundTuneOpConfig,
    TLQOpConfig,
    registered_tlq_op_types,
)
from msmodelslim.processor.trainable_linear_quant.core.kernels import ensure_tlq_kernel_registered
from .train_config import BlockTrainConfig


def _needs_act_fake_quant(act_qconfig: QConfig) -> bool:
    if QDType(act_qconfig.dtype) == QDType.FLOAT:
        return False
    if act_qconfig.method == "none":
        return False
    return True


def validate_trainable_linear_qconfig(q: LinearQConfig) -> None:
    ensure_tlq_kernel_registered(q.weight, role="weight")
    if _needs_act_fake_quant(q.act):
        ensure_tlq_kernel_registered(q.act, role="act")


def _validate_qconfig_group_size(qconfig: QConfig, field_path: str) -> None:
    is_per_group = qconfig.scope == QScope.PER_GROUP
    has_group_size = "group_size" in qconfig.ext

    if is_per_group:
        if not has_group_size:
            raise SchemaValidateError(
                f"When quantization config scope is per_group, "
                f"ext field must contain group_size, "
                f"but {field_path} does not have group_size field",
                action=f"Please add group_size parameter to {field_path} ext field",
            )
        group_size = qconfig.ext["group_size"]
        if not isinstance(group_size, int) or group_size <= 0:
            raise SchemaValidateError(
                f"When quantization config scope is per_group, "
                f"group_size in ext field must be a positive integer, "
                f"but {field_path} has group_size={group_size}",
                action=f"Please set {field_path} ext.group_size to a positive integer",
            )
    elif has_group_size:
        raise SchemaValidateError(
            f"When quantization config scope is not per_group, "
            f"ext field should not contain group_size, but {field_path} has group_size field",
            action=f"Please remove group_size parameter from {field_path} ext field",
        )


def _validate_linear_qconfig(qconfig: LinearQConfig, prefix: str) -> None:
    _validate_qconfig_group_size(qconfig.weight, f"{prefix}.weight")
    _validate_qconfig_group_size(qconfig.act, f"{prefix}.act")
    try:
        validate_trainable_linear_qconfig(qconfig)
    except Exception as exc:
        raise SchemaValidateError(
            f"{prefix} is not supported by trainable linear quant: {exc}",
            action="Please fix the qconfig dtype/method or register the required TLQ kernel",
        ) from exc


class QuantStrategyConfig(BaseModel):
    qconfig: LinearQConfig = Field(description="量化配置")
    include: List[str] = Field(default_factory=lambda: ["*"], description="包含的模块名称")
    exclude: List[str] = Field(default_factory=list, description="排除的模块名称")

    @model_validator(mode="after")
    def _validate_qconfig(self) -> "QuantStrategyConfig":
        _validate_linear_qconfig(self.qconfig, prefix="qconfig")
        return self


def _default_tlq_op_configs() -> List[TLQOpConfig]:
    return [MinmaxTuneOpConfig(), RoundTuneOpConfig()]


class TrainableLinearQuantProcessorConfig(AutoProcessorConfig):
    type: Literal["trainable_linear_quant"] = Field(
        default="trainable_linear_quant",
        description="处理器类型标识",
    )
    operations: List[SerializeAsAny[TLQOpConfig]] = Field(
        default_factory=_default_tlq_op_configs,
        min_length=1,
        description="可训练量化管线 OP 配置列表；每项含 type，其余字段由各插件定义",
        validation_alias=AliasChoices("operations", "ops"),
    )
    strategies: List[QuantStrategyConfig] = Field(
        default_factory=list,
        min_length=1,
        description="量化策略配置列表",
    )
    train_with_act_quant: bool = Field(
        default=False,
        description=(
            "块级训练前向是否对激活做伪量化（经 x_kernel）；"
            "false 与 autoround 的 train_with_act_quant=False 一致；"
            "导出 IR 仍由 qconfig.act 决定，不受此项影响"
        ),
    )
    enable_quanted_input: bool = Field(
        default=False,
        description=(
            "是否将本层量化前向结果作为下一层训练/量化传播的旁路输入（q_input）；"
            "不影响浮点 teacher：Runner 层间 datas 始终传递 teacher 输出"
        ),
    )
    train_config: BlockTrainConfig = Field(
        default_factory=BlockTrainConfig,
        description=(
            "块级 Trainer 超参：iters、gradient_accumulate_steps、select_best、"
            "lr（或 learning_rate）、loss_type；各 OP 可单独配置 lr 覆盖全局值"
        ),
    )

    @field_validator("operations", mode="before")
    @classmethod
    def _operations_before(cls, v: Any) -> Any:
        if v is None:
            return _default_tlq_op_configs()
        if isinstance(v, dict) and "type" in v:
            v = [v]
        if not isinstance(v, list):
            return _default_tlq_op_configs()
        out: List[TLQOpConfig] = []
        registered = set(registered_tlq_op_types())
        for item in v:
            if isinstance(item, TLQOpConfig):
                out.append(item)
                continue
            if isinstance(item, dict):
                op_type = item.get("type")
                if op_type and op_type not in registered:
                    raise SchemaValidateError(
                        f"operations type={op_type!r} is not a registered TLQ op; "
                        f"available types: {sorted(registered)}",
                        action="Please use a supported operation type from the TLQ op registry",
                    )
            out.append(TLQOpConfig.model_validate(item))
        return out


__all__ = [
    "QuantStrategyConfig",
    "TrainableLinearQuantProcessorConfig",
    "validate_trainable_linear_qconfig",
]
