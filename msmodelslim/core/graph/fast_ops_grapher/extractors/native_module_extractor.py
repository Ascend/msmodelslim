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

from typing import Any, Tuple, Dict

from torch import nn

from msmodelslim.core.graph.fast_ops_grapher.extractors.base_extractor import BaseExtractor


class NativeModuleExtractor(BaseExtractor):
    """NativeModuleExtractor 用于从通用的 PyTorch nn.Module 中提取计算图。"""

    def __init__(self, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        super().__init__()
        self._module = module
        self._args = args
        self._kwargs = kwargs

    @staticmethod
    # pylint: disable=arguments-differ
    def create(module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> 'NativeModuleExtractor':
        return NativeModuleExtractor(module, args, kwargs)

    @property
    def target_module(self) -> nn.Module:
        return self._module

    @property
    def dummy_inputs(self) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        return (self._args, self._kwargs)
