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

# Trainer-only hyperparameters for block-level TLQ optimization.

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class EmaSelectBest(BaseModel):
    """EMA 滑动平均选最优；支持 early stop。"""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    mode: Literal["ema"] = Field(default="ema", description="选取策略：ema")
    ema_beta: float = Field(
        default=0.7,
        gt=0.0,
        le=1.0,
        description="best_loss 的 EMA 衰减系数",
    )
    ema_window_size: int = Field(
        default=5,
        ge=1,
        description="mean_loss 滑动平均窗口长度",
    )
    early_stop_patience: int = Field(
        default=-1,
        ge=-1,
        description="连续多少 iter 无更优快照后早停；-1 表示禁用",
    )


class MinLossSelectBest(BaseModel):
    """当轮 loss 历史最小值选最优；支持 early stop。"""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    mode: Literal["min_loss"] = Field(default="min_loss", description="选取策略：min_loss")
    early_stop_patience: int = Field(
        default=-1,
        ge=-1,
        description="连续多少 iter 无更优快照后早停；-1 表示禁用",
    )


class LastSelectBest(BaseModel):
    """仅保存 iter 0 与最后一轮；无 early stop。"""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    mode: Literal["last"] = Field(default="last", description="选取策略：last")


SelectBestConfig = Annotated[
    Union[EmaSelectBest, MinLossSelectBest, LastSelectBest],
    Field(discriminator="mode"),
]


class BlockTrainConfig(BaseModel):
    """块级训练（block train）超参配置。

    控制 trainable_linear_quant 块级 Trainer 的迭代、梯度累加、学习率、
    最优快照选择策略与损失函数。
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    iters: int = Field(
        default=50,
        ge=0,
        description="块级训练迭代次数；为 0 时 Trainer 跳过优化",
    )
    gradient_accumulate_steps: int = Field(
        default=8,
        ge=1,
        description="梯度累加步数，用于在有限显存下调节等效 batch",
    )
    lr: float = Field(
        default=0.01,
        gt=0.0,
        validation_alias=AliasChoices("lr", "learning_rate"),
        description="全局基础学习率",
    )
    select_best: SelectBestConfig = Field(
        default_factory=EmaSelectBest,
        description="最优 iter 快照策略（按 mode 区分字段：ema / min_loss / last）",
    )
    loss_type: Literal["l1", "custom_outlier"] = Field(
        default="l1",
        description=("块级训练损失：l1（L1Loss reduction=none）、custom_outlier（0.3*全量 L1 + 0.7*3σ 内区域 L1）"),
    )
    train_seed: int = Field(
        default=42,
        description="块级训练随机种子（用于 sample 打乱与确定性算子）",
    )


__all__ = [
    "BlockTrainConfig",
    "EmaSelectBest",
    "LastSelectBest",
    "MinLossSelectBest",
    "SelectBestConfig",
]
