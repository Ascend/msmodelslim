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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny

from msmodelslim.core.quantization_format.registry import QuantFormatConfigList
from msmodelslim.core.train.train_dataset_builder import TrainDatasetConfig
from msmodelslim.utils.plugin import TypedConfig

TRAIN_TASK_PLUGIN_GROUP = "msmodelslim.train_task.plugins"
TRAIN_ALGORITHM_PLUGIN_GROUP = "msmodelslim.train_algorithm.plugins"

_TRAIN_MODEL_CONFIG = ConfigDict(extra="forbid")
_TRAIN_HYPER_CONFIG = ConfigDict(extra="allow")

# 分块交叉熵默认 chunk 大小（单一事实源，loss.py 反向引用）。
DEFAULT_CE_CHUNK_SIZE: int = 512


@TypedConfig.plugin_entry(entry_point_group=TRAIN_ALGORITHM_PLUGIN_GROUP)
class TrainAlgorithmConfig(TypedConfig):
    """Thin TypedConfig base for flat ``algorithm[]`` plugin dispatch (e.g. paretoq)."""

    # NOTE: do not use postponed annotations — TypedConfig.detect_type_field needs live Annotated.
    type: TypedConfig.TypeField = Field(
        default="Unknown",
        description="训练量化算法类型，如 paretoq；YAML 须显式指定。",
    )
    include: List[str] = Field(
        default_factory=lambda: ["*"],
        description="参与该算法的模块名模式，默认 *。",
    )
    exclude: List[str] = Field(
        default_factory=list,
        description="排除的模块名模式；优先级高于 include。",
    )
    model_config = ConfigDict(
        extra="forbid",
        validate_by_name=True,
        validate_by_alias=False,
        serialize_by_alias=False,
    )


class TrainingArgumentsConfig(BaseModel):
    """通用训练参数（train.training_arguments 段）。

    声明字段与 extra 扩展字段（如 max_steps / lr_scheduler_type / save_only_model）
    均直通 ``transformers.TrainingArguments``；当前 transformers 版本不识别的
    extra 键由 args_builder 告警丢弃。QAT 专属字段请放入 ``train.qat`` 段。
    """

    model_config = _TRAIN_HYPER_CONFIG

    num_train_epochs: float = Field(default=1.0, gt=0.0, description="训练轮数。")
    learning_rate: float = Field(default=1e-5, gt=0.0, description="基础学习率。")
    per_device_train_batch_size: int = Field(
        default=1,
        ge=1,
        description="单卡训练 batch size。",
    )
    per_device_eval_batch_size: int = Field(
        default=1,
        ge=1,
        description="单卡评估 batch size。",
    )
    gradient_accumulation_steps: int = Field(
        default=8,
        ge=1,
        description="梯度累积步数。",
    )
    warmup_ratio: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="warmup 步数占总训练步数的比例。",
    )


class QATConfig(BaseModel):
    """QAT 专属超参块（train.qat 段）；注意区别于任务级 ``QATTaskConfig``。"""

    model_config = _TRAIN_MODEL_CONFIG

    model_max_length: int = Field(
        ...,
        gt=0,
        description="模型输入最大序列长度；由 RankWorker 应用到 tokenizer，非 TrainingArguments 字段。",
    )
    qparam_lr_scale: float = Field(
        default=10.0,
        gt=0.0,
        description="量化参数相对基础学习率的缩放系数。",
    )
    ce_chunk_size: int = Field(
        default=DEFAULT_CE_CHUNK_SIZE,
        ge=1,
        description="分块交叉熵 chunk 大小。",
    )
    fake_quant_after_n_steps: Optional[int] = Field(
        default=None,
        ge=0,
        description="训练开始后第 N 步起启用 fake quant；None 表示立即启用。",
    )
    log_qparams: bool = Field(
        default=False,
        description="启用 qparam 记录与 RelativeAlphaAdamW 优化器。",
    )
    alpha_min_ratio: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="log-space alpha 下限相对 PTQ alpha 的比例。",
    )
    alpha_max_ratio: float = Field(
        default=2.0,
        ge=1.0,
        description="log-space alpha 上限相对 PTQ alpha 的比例。",
    )
    trainable_scope: Optional[str] = Field(
        default=None,
        description="可训练参数范围。占位声明字段，当前训练循环未读取。",
    )
    freeze_qparams: bool = Field(
        default=False,
        description="是否冻结量化参数。占位声明字段，当前训练循环未读取。",
    )


class CheckpointingConfig(BaseModel):
    """通用 checkpoint 策略。"""

    model_config = _TRAIN_MODEL_CONFIG

    save_eval_interval: int = Field(
        default=200,
        ge=1,
        description="每隔多少 step 做一次评估并保存 checkpoint。",
    )
    eval_on_start: bool = Field(default=False, description="训练开始前是否先评估一次。")
    save_total_limit: int = Field(
        default=2,
        ge=1,
        le=2,
        description="最多保留的 checkpoint 数量；强制仅 best+latest。",
    )
    load_best_model_at_end: bool = Field(
        default=True,
        description="训练结束时是否加载评估最优的 checkpoint。",
    )


class NestedTrainConfig(BaseModel):
    """训练段：通用训练参数 + QAT 专属超参 + checkpoint。"""

    model_config = _TRAIN_MODEL_CONFIG

    training_arguments: TrainingArgumentsConfig = Field(
        ...,
        description="通用训练参数（直通 transformers.TrainingArguments）。",
    )
    qat: QATConfig = Field(..., description="QAT 专属超参（强类型）。")
    checkpointing: CheckpointingConfig = Field(
        default_factory=CheckpointingConfig,
        description="checkpoint 保存与评估策略。",
    )


class CoreTrainTaskConfig(BaseModel):
    """
    通用训练任务配置。

    在 setup（YAML / 应用层）完成强校验后，原样传入 worker，worker 侧不再二次校验。
    具体训练范式（如 qat）在子类上声明 ``type`` 并挂插件分发。
    """

    model_config = ConfigDict(extra="forbid")

    algorithm: List[SerializeAsAny[TrainAlgorithmConfig]] = Field(
        ...,
        min_length=1,
        description="训练算法列表（按 type 插件分发，如 paretoq）。",
    )
    dataset: TrainDatasetConfig = Field(..., description="训练数据集配置。")

    train: NestedTrainConfig = Field(..., description="训练超参与 checkpoint 配置。")
    save: QuantFormatConfigList = Field(
        default_factory=list,
        description="导出 / 保存格式列表。",
    )
    qparam_names: List[str] = Field(
        default_factory=list,
        description="可训练量化参数名（如 weight_clip_val）。",
    )
    source_model_path: Optional[str] = Field(
        default=None,
        description="源模型路径。",
    )


@dataclass
class CoreTrainTask:
    """Worker-side mutable training runtime state."""

    config: CoreTrainTaskConfig
    dist_info: Dict[str, Any] = field(default_factory=dict)
    qparam_names: List[str] = field(default_factory=list)
    save_path: Optional[Path] = None
    best_checkpoint: Optional[Path] = None
    latest_checkpoint: Optional[Path] = None
