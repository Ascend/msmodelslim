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
import fnmatch
from abc import ABC, abstractmethod
from typing import Callable, Dict, Generic, List, TypeVar

import torch.nn as nn

from msmodelslim.utils.exception import UnsupportedError


class AnalysisTargetMatcher(ABC):
    """抽象基类：按模型结构匹配目标层。子类只需实现 _matches(module)，基类统一做 named_modules 遍历。"""

    def get_target_layers(self, model: nn.Module, prefix: str = "") -> List[str]:
        """遍历 model.named_modules()，收集满足 _matches(module) 的层名（唯一循环）。"""
        _all_target_layers = []
        for name, module in model.named_modules(prefix=prefix):
            if self._matches(module):
                _all_target_layers.append(name)
        return _all_target_layers

    @staticmethod
    def filter_layers_by_patterns(layer_names: List[str], patterns: List[str]) -> List[str]:
        """按 patterns 过滤层名（支持 fnmatch）。"""
        if not patterns or patterns == ['*']:
            return layer_names
        filtered = []
        for layer_name in layer_names:
            for pattern in patterns:
                if fnmatch.fnmatch(layer_name, pattern):
                    filtered.append(layer_name)
                    break
        return filtered

    @abstractmethod
    def _matches(self, module: nn.Module) -> bool:
        """当前 module 是否算作目标层。"""
        ...


class LayerAnalysisMethod(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the analysis method"""
        ...

    @abstractmethod
    def get_hook(self) -> Callable:
        """Get the hook function to collect data during model inference."""
        ...


TMethod = TypeVar("TMethod", bound=LayerAnalysisMethod)


class BaseMethodFactory(Generic[TMethod]):
    """
    Generic method factory backed by a registry.

    The registry maps `metrics` string -> method class.
    """

    def __init__(self) -> None:
        self._methods: Dict[str, type[TMethod]] = {}

    def _get_methods(self) -> Dict[str, type[TMethod]]:
        return self._methods

    def create_method(self, method_name: str, **kwargs) -> TMethod:
        methods = self._get_methods()
        if method_name not in methods:
            supported = list(methods.keys())
            raise UnsupportedError(f"Selected analysis method '{method_name}' is not supported.",
                                      action=f"Please use a supported analysis method. Supported methods: {supported}")
        return methods[method_name](**kwargs)

    def register_method(self, method_name: str, method_class: type[TMethod]) -> None:
        if not issubclass(method_class, LayerAnalysisMethod):
            raise TypeError("Method class must inherit from LayerAnalysisMethod")
        self._get_methods()[method_name] = method_class

    def get_supported_methods(self) -> List[str]:
        return list(self._get_methods().keys())
     