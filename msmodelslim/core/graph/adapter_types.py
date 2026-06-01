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
from typing import Dict, List, Any, Optional

# 支持的子图类型常量
SUPPORTED_SUBGRAPH_TYPES = ['norm-linear', 'linear-linear', 'ov', 'up-down']


@dataclass
class MappingConfig:
    """映射关系配置"""

    targets: List[str]
    # 非融合场景用 source 配置为None
    source: Optional[str] = None


@dataclass
class FusionConfig:
    """融合配置联合体

    用于管理不同类型的融合配置，支持扩展新的融合类型
    """

    fusion_type: str = "none"  # 融合类型：none, qkv, custom等

    # QKV融合相关配置
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None

    # 自定义融合配置（可扩展）
    custom_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """验证融合配置的有效性"""
        if self.fusion_type == "qkv":
            if self.num_attention_heads is None or self.num_key_value_heads is None:
                raise ValueError("QKV融合类型必须提供num_attention_heads和num_key_value_heads")
        elif self.fusion_type == "kv":
            if self.num_attention_heads is None:
                raise ValueError("KV融合类型必须提供num_attention_heads")
            if not self.custom_config:
                raise ValueError("KV融合必须在custom_config提供qk_nope_head_dim和v_head_dim")
        elif self.fusion_type == "custom":
            if not self.custom_config:
                raise ValueError("自定义融合类型必须提供custom_config")
        elif self.fusion_type != "none":
            raise ValueError(f"不支持的融合类型: {self.fusion_type}")


@dataclass
class AdapterConfig:
    """融合配置结构体

    包含模型融合和子图处理的所有配置参数
    """

    # 子图类型（必需）
    subgraph_type: str
    # 自定义的映射关系（必需），支持融合 / 非融合两种配置
    mapping: MappingConfig
    # 融合配置（可选）
    fusion: Optional[FusionConfig] = None
    # 额外配置（可选）
    extra_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """验证配置的有效性"""
        if self.subgraph_type is None:
            raise ValueError("subgraph_type is required")

        if self.subgraph_type not in SUPPORTED_SUBGRAPH_TYPES:
            raise ValueError(f"subgraph_type: {self.subgraph_type} 不是支持的子图类型")
