#!/usr/bin/env python
# -*- coding: UTF-8 -*-

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

from abc import abstractmethod
from typing import Dict, Optional, Literal

import torch
from torch import nn
from pydantic import BaseModel, field_validator, model_validator, ConfigDict

from ..common.quarot_utils import QuaRotMode


class RotationConfig(BaseModel):
    """
    旋转配置数据类。

    用于配置模块的旋转方式，支持用户提供旋转矩阵或自动生成。
    """

    rotation_type: str  # "input", "output", "replace", "offline"
    rotation_matrix: Optional[torch.Tensor] = None  # 用户提供的旋转矩阵
    rotation_size: Optional[int] = None  # 自动生成时使用的尺寸
    rotation_mode: Optional[QuaRotMode] = None  # 自动生成时使用的模式
    block_size: int = -1  # 块大小，用于自动生成
    rot_step: int = 1  # 旋转步数，用于自动生成
    eye_step: tuple = (-1,)  # eye步数，用于自动生成
    seed: int = 1234  # 随机数种子，用于自动生成，默认为1234
    dtype: torch.dtype = torch.float32  # 数据类型，用于自动生成，默认为 torch.float32
    rotation_side: Optional[Literal["left", "right"]] = None  # 旋转方向，仅用于 offline 类型，默认为 "right"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator('rotation_type')
    @classmethod
    def validate_rotation_type(cls, v):
        """验证 rotation_type 的有效性"""
        if v not in ["input", "output", "replace", "offline"]:
            raise ValueError(f"rotation_type must be one of ['input', 'output', 'replace', 'offline'], got {v}")
        return v

    @model_validator(mode='after')
    def validate_rotation_config(self):
        """验证配置的有效性"""
        # 对于 offline 类型，必须指定 rotation_side
        if self.rotation_type == "offline":
            if self.rotation_side is None:
                self.rotation_side = "right"  # 默认为右旋
            if self.rotation_side not in ["left", "right"]:
                raise ValueError(
                    f"rotation_side must be 'left' or 'right' for offline rotation, got {self.rotation_side}"
                )

        # 如果用户提供了旋转矩阵，则不需要其他参数
        if self.rotation_matrix is not None:
            return self

        # 如果自动生成，需要提供 rotation_size
        if self.rotation_size is None:
            raise ValueError("Either rotation_matrix or rotation_size must be provided")

        return self


class OnlineQuaRotInterface:
    """
    简化的在线旋转接口。

    模型适配器需要实现此接口以支持在线旋转功能。
    """

    RotationConfig = RotationConfig
    QuaRotMode = QuaRotMode

    @abstractmethod
    def get_online_rotation_configs(self, model: Optional[nn.Module] = None) -> Dict[str, RotationConfig]:
        """
        返回模块名到旋转配置的映射。

        该方法应该返回一个字典，键是模块的完整名称（如 "model.layers.0.self_attn.o_proj"），
        值是对应的 RotationConfig 对象。

        如果提供了 model 参数，可以在此方法中直接挂载 Identity 模块到模型上。

        Args:
            model: 可选的模型实例，如果提供，可以在此方法中挂载模块

        Returns:
            Dict[str, RotationConfig]: 模块名到旋转配置的映射
                - rotation_type: "input" | "output" | "replace" | "offline"
                - rotation_matrix: Optional[torch.Tensor] (用户提供)
                - rotation_size: Optional[int] (自动生成时使用)
                - rotation_mode: Optional[QuaRotMode] (自动生成时使用)
                - block_size: int (自动生成时使用，默认-1)
                - rot_step: int (自动生成时使用，默认1)
                - eye_step: tuple (自动生成时使用，默认(-1,))
                - seed: int (随机数种子，用于自动生成，默认1234)
                - rotation_side: Optional["left" | "right"] (仅用于 offline 类型，默认 "right")

        Example:
            ```python
            def get_online_rotation_configs(self, model=None):
                # 可以在这里使用 model 参数来挂载 Identity 模块
                if model is not None:
                    for layer_idx in range(self.config.num_hidden_layers):
                        attn = model.get_submodule(f"model.layers.{layer_idx}.self_attn")
                        attn.register_module('q_rot', nn.Identity())

                return {
                    "model.layers.0.self_attn.q_rot": RotationConfig(
                        rotation_type="replace",
                        rotation_size=768
                    )
                }
            ```
        """
        pass
