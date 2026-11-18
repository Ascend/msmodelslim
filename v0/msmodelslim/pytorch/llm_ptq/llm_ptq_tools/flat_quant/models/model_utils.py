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

from typing import List, Dict, Any

import torch
from torch.nn import Module
from tqdm import tqdm


class StructurePair:
    """负责管理模块对"""
    def __init__(self, sources: str, targets: List[str], prefix_name: str):
        if not isinstance(sources, str):    
            raise ValueError(f"sources must be a string, got {type(sources)}")
        if not isinstance(targets, List):
            raise ValueError(f"targets must be a list of strings, got {type(targets)}")
        self.source_modules = sources
        self.target_modules = targets
        self.name = prefix_name + '.' + self._name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def accept(self, visitor) -> Any:
        """接受访问者模式"""
        return visitor.visit(self)
        
    def contain(self, name: str) -> bool:
        return any(name == target for target in self.target_modules)


class ModelStructureBridge:
    """模型结构桥接器基类"""
    def __init__(self, model, config=None):
        self.model = model
        if config:
            self.config = config
        else:
            self.config = getattr(model, 'config', None)
        self._structure_pair_registry = {} 
        self._layers_name = None

    def register_structure_pair(self, pair: type):
        """注册结构对类型"""
        if not isinstance(pair, StructurePair):
            raise TypeError(f"pair_class must be a instance of StructurePair, got {pair}")
        if pair.__class__.__name__ not in self._structure_pair_registry:
            self._structure_pair_registry[pair.__class__.__name__] = [pair]
        else:
            pair_list = self._structure_pair_registry[pair.__class__.__name__]
            contains_target = any(str(pair) == str(obj) for obj in pair_list)
            if not contains_target:
                pair_list.append(pair)
        
    def get_structure_pairs(self) -> List[StructurePair]:
        """获取所有结构对"""
        return self._structure_pair_registry

    def get_layers(self) -> str:
        """获取所有transformers层"""
        return self._layers_name

    def get_layer_by_index(self, index: int) -> str:
        """获取指定层"""
        return self._layers_name + f".{index}"

    def analyze_structure(self):
        """分析模型结构，由子类实现"""
        raise NotImplementedError


class AttnNormLinearPair(StructurePair):
    _name = "self_attn.qkv_proj"
    """注意力层和线性层结构对"""
    def __init__(self, config, attn_norm_name, linear_name, prefix_name: str):
        super(AttnNormLinearPair, self).__init__(attn_norm_name, linear_name, prefix_name)
        self.config = config

    def accept(self, visitor) -> Any:
        return visitor.visit_attn_norm_linear_pair(self)


class AttnLinearLinearPair(StructurePair):
    _name = "self_attn.o_proj"
    """注意力层线性层和线性层结构对"""
    def __init__(self, config, pre_linear_name, post_linear_name, prefix_name: str):
        super(AttnLinearLinearPair, self).__init__(pre_linear_name, post_linear_name, prefix_name)
        self.config = config

    def accept(self, visitor) -> Any:
        return visitor.visit_attn_linear_linear_pair(self)


class MLPNormLinearPair(StructurePair):
    _name = "mlp.gate_up_proj"
    """注意力层和线性层结构对"""
    def __init__(self, config, mlp_norm_name, linear_name, prefix_name: str):
        super(MLPNormLinearPair, self).__init__(mlp_norm_name, linear_name, prefix_name)
        self.config = config

    def accept(self, visitor) -> Any:
        return visitor.visit_mlp_norm_linear_pair(self)


class MLPLinearLinearPair(StructurePair):
    _name = "mlp.down_proj"
    """线性层和线性层结构对"""
    def __init__(self, config, pre_linear_name, post_linear_name, prefix_name: str):
        super(MLPLinearLinearPair, self).__init__(pre_linear_name, post_linear_name, prefix_name)
        self.config = config

    def accept(self, visitor) -> Any:
        return visitor.visit_mlp_linear_linear_pair(self)


def get_module_by_name(model: Module, submodule_key: str) -> Module:
    """根据名称获取模块"""
    module_tokens = submodule_key.split('.')
    cur_mod = model
    for s in module_tokens:
        cur_mod = getattr(cur_mod, s)
    return cur_mod


def set_module_by_name(model: Module, submodule_key: str, module: Module, clone_hooks: bool = True):
    """根据名称设置模块
    
    Args:
        model: 要修改的模型
        submodule_key: 模块的路径名称，以点分隔
        module: 新的模块实例
        clone_hooks: 是否将原模块的钩子克隆到新模块，默认为True
    """
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    
    # 保存原始模块的引用
    if clone_hooks:
        old_module = getattr(cur_mod, tokens[-1])
        clone_module_hooks(old_module, module)
    
    # 设置新模块
    setattr(cur_mod, tokens[-1], module)


def clone_module_hooks(source_module: Module, target_module: Module):
    hook_types = [
        ('_forward_pre_hooks', 'register_forward_pre_hook'),
        ('_forward_hooks', 'register_forward_hook'),
        ('_backward_pre_hooks', 'register_backward_pre_hook'), 
        ('_backward_hooks', 'register_backward_hook')
    ]
    
    for hook_attr, register_method in hook_types:
        if hasattr(source_module, hook_attr):
            hooks_dict = getattr(source_module, hook_attr, {})
            if hooks_dict:
                register_func = getattr(target_module, register_method, None)
                if register_func:
                    for hook_fn in hooks_dict.values():
                        try:
                            register_func(hook_fn)
                        except (TypeError, AttributeError):
                            # 某些钩子可能不兼容，静默跳过
                            continue


def remove_after_substring(text, substring):
    index = text.find(substring)
    if index != -1:
        return text[:index + len(substring)]
    return text


class TransformerStructurePairVisitor:
    """访问者抽象基类"""
    def visit(self, pair: StructurePair) -> Any:
        """访问结构对"""
        pass

    def visit_attn_norm_linear_pair(self, pair: AttnNormLinearPair) -> Any:
        """访问注意力层norm和线性层结构对"""
        pass

    def visit_attn_linear_linear_pair(self, pair: AttnLinearLinearPair) -> Any:
        """访问注意力层线性层和线性层结构对"""
        pass

    def visit_mlp_norm_linear_pair(self, pair: MLPNormLinearPair) -> Any:
        """访问MLP层norm和线性层结构对"""
        pass

    def visit_mlp_linear_linear_pair(self, pair: MLPLinearLinearPair) -> Any:
        """访问MLP层线性层和线性层结构对"""
        pass


class RunnerStopExecution(Exception):
    """停止执行"""
    pass

