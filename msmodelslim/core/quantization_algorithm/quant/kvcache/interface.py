#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from abc import ABC, abstractmethod
from typing import Callable, Dict

from torch import nn

from msmodelslim.utils.exception import UnsupportedError


class DynamicCacheAdapterInterface(ABC):
    @abstractmethod
    def inject_kvcache_placeholders(
        self,
        root_name: str,
        root_module: nn.Module,
        should_inject: Callable[[str], bool],
        placeholder_factory: Callable[[], nn.Module],
    ) -> Dict[str, nn.Module]:
        """在模型指定位置安装 KV cache 量化 placeholder。

        Adapter 应遍历 root_module.named_modules()，识别 attention 模块，
        对 should_inject(full_name) 返回 True 的模块，在其下安装
        ``key_states_quantizer`` 和 ``value_states_quantizer`` 两个占位子模块
        （通过 placeholder_factory() 创建），并返回 {full_name: attention_module} 映射。

        要求：
        - full_name 为完整模块路径（如 ``model.language_model.layers.3.self_attn``）
        - should_inject 已包含 include/exclude 过滤逻辑
        - 返回值用于 processor 后续创建统计器 / 部署 IR
        """
        raise UnsupportedError(
            f"{self.__class__.__name__} does not support inject_kvcache_placeholders",
            action="Please implement this method before using KV cache quantization",
        )
