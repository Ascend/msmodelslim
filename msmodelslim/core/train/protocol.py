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

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from torch import nn
from transformers import PreTrainedTokenizerBase

from msmodelslim.utils.exception import UnsupportedError


@dataclass
class TrainHookContext:
    """
    训练钩子运行时上下文（由 RankWorker 在 load 之后注入）。

    - model: 已加载的训练模型（``pre_train`` 必需）
    - core_task: worker 侧 ``CoreTrainTask`` 运行态（可写回 qparam_names）
    - extra: 钩子可读写的扩展字典（如 ``qparam_names``）
    """

    model: Optional[nn.Module] = None
    core_task: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)


class ITrainHook(ABC):
    """
    训练算法钩子接口（插件侧实现，如 ParetoQ）。

    语义：
    - 一个 hook = 一种训练算法在「开训前」对模型图的改写能力；
    - 典型动作：Linear → FakeQuant 壳、初始化可学习 α（如 weight_clip_val）等。

    约定：
    - 由 setup 侧组装后下发；RankWorker 在 load_model 之后、训练循环之前调用 ``pre_train``；
    - 可向 ``ctx.extra`` / ``ctx.core_task`` 回写训练所需元数据（如 qparam_names）。

    Note:
        当前与仓库其它接口一致，先用 ABC；后续插件化统一时再收敛为 Protocol。
    """

    @abstractmethod
    def pre_train(self, ctx: TrainHookContext) -> None:
        """
        训练前改写模型图并完成算法侧初始化。

        参数:
            ctx: 运行时上下文；实现通常要求 ``ctx.model`` 非空

        返回:
            None（原地修改 ``ctx.model`` / ``ctx.extra`` / ``ctx.core_task``）
        """
        raise UnsupportedError(
            f"{self.__class__.__name__} does not support pre_train",
            action="Please implement pre_train before running QAT training.",
        )


class QATLoadInterface(ABC):
    """
    QAT 模型加载协议（Adapter 侧实现）。

    职责边界：
    - 负责为 QAT 训练加载 tokenizer / HF 模型（避免 PTQ 的 device_map 路径）；
    - 不负责 FakeQuant 改写或训练循环。
    """

    @abstractmethod
    def load_tokenizer_for_qat(self, model_max_length: int) -> PreTrainedTokenizerBase:
        """
        加载并配置训练用 tokenizer。

        参数:
            model_max_length: 序列最大长度（写入 tokenizer.model_max_length）

        返回:
            已配置的 ``PreTrainedTokenizerBase`` 实例
        """
        raise UnsupportedError(
            f"{self.__class__.__name__} does not support load_tokenizer_for_qat",
            action="Please implement load_tokenizer_for_qat before running QAT training.",
        )

    @abstractmethod
    def load_model_for_qat(self) -> nn.Module:
        """
        加载训练用浮点模型。

        返回:
            可用于 QAT 的 ``nn.Module``（通常关闭 use_cache）
        """
        raise UnsupportedError(
            f"{self.__class__.__name__} does not support load_model_for_qat",
            action="Please implement load_model_for_qat before running QAT training.",
        )
