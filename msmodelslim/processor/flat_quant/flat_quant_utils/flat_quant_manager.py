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
# pylint: disable=consider-using-generator

import torch
from collections import OrderedDict  # noqa: F401
from typing import Dict, List, Union, Any

from typing import Type, Pattern, Tuple, Callable  # noqa: F401

from msmodelslim.processor.flat_quant.flat_quant_utils.structure_pair import StructurePair
from msmodelslim.processor.flat_quant.flat_quant_utils.flat_fake_quant_linear import ForwardMode
from msmodelslim.processor.flat_quant.flat_quant_utils.utils import remove_after_substring
from msmodelslim.utils.exception import SchemaValidateError


class FlatQuantLayerManager:
    """逐层管理 FlatQuant 算法的全流程操作：结构对注册、变换应用、模式切换与回退。"""

    def __init__(self, module: torch.nn.Module, config: Dict[str, Any] = None) -> None:
        """初始化 FlatQuant 管理器，设置模型、配置和结构对注册表。"""
        self.module = module
        self.config = config or getattr(module, 'config', None)
        self._structure_pair_map: Dict[str, List[StructurePair]] = {}
        self._layer_pairs_list: List[StructurePair] = []

    def register_structure_pair(self, pair: StructurePair) -> None:
        """注册一个结构对实例（如 MLP 与 Linear 的配对），避免重复注册。"""
        if not isinstance(pair, StructurePair):
            raise SchemaValidateError(f"pair must be an instance of StructurePair, but got {type(pair)}")

        class_name = pair.__class__.__name__
        if class_name not in self._structure_pair_map:
            self._structure_pair_map[class_name] = []
        pair_list = self._structure_pair_map[class_name]
        if not any(str(pair) == str(obj) for obj in pair_list):
            pair_list.append(pair)

    def register_layer_pairs(self, structure_configs: List[Dict[str, Any]], layer_name: str) -> None:
        """根据结构配置列表分析模型结构，并注册所有结构对实例。"""
        for name, _ in self.module.named_modules(prefix=layer_name):
            for config in structure_configs:
                if config["source"] in name:
                    targets = []
                    source_clean_name = remove_after_substring(name, config["source"])
                    for target_name in config["targets"]:
                        # 此处用 tatget_name 对 source_clean_name 进行替换，是考虑到前缀可能存在
                        targets.append(source_clean_name.replace(config["source"], target_name))
                    kwargs = config.get('extra_config', {})
                    self.register_structure_pair(
                        config["pair_class"](self.config, source_clean_name, targets, layer_name, self.module, **kwargs)
                    )

        pairs_dict = self._structure_pair_map
        self._layer_pairs_list = []
        support_structure_pairs = StructurePair.support_structure_pairs
        num = max([len(pairs_dict[pair_type.__name__]) for pair_type in support_structure_pairs])
        for i in range(num):
            for pair_type in support_structure_pairs:
                if i < len(pairs_dict[pair_type.__name__]):
                    self._layer_pairs_list.append(pairs_dict[pair_type.__name__][i])

    def wrap_linear(self, device: Union[str, torch.device] = None) -> None:
        """替换指定前缀下的线性层，应用 FlatQuant 的变换逻辑。"""
        with torch.device(device=device):
            self._call_method_on_pairs(method_name="wrap_linear")

    def rollback_trans(self, pair_name: str = "") -> None:
        """回退已应用的变换矩阵（trans），恢复原始状态。"""
        self._call_method_on_pairs(method_name="rollback_trans", pair_name=pair_name)

    def change_mode(self, mod: ForwardMode) -> None:
        """切换所有结构对的前向传播模式（如训练/推理/量化模式）。"""
        self._call_method_on_pairs(method_name="change_mode", mod=mod)

    def _call_method_on_pairs(self, method_name: str, *args, **kwargs) -> None:
        """遍历指定前缀下的所有结构对，统一调用指定方法。"""
        for pair in self._layer_pairs_list:
            method = getattr(pair, method_name)
            method(*args, **kwargs)

    def match_pair(self, proj_name: str) -> None:
        """（预留）根据模块名称匹配对应的结构对。"""
        pass
