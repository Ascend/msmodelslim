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


import os
import copy
from collections import OrderedDict
import torch
import torch.nn as nn
from msmodelslim import logger

try:
    import torch_npu
except ImportError:
    logger.warning("Unable to import torch_npu.")
from .dag_utils.torch_dag_adapter import TorchDAGAdapter, DagTorchHook


class GraphOpt:
    @staticmethod
    def set_module(model,
                   submodule_key,
                   module):
        tokens = submodule_key.split('.')
        sub_tokens = tokens[:-1]
        cur_mod = model
        for s in sub_tokens:
            cur_mod = getattr(cur_mod, s)
        setattr(cur_mod, tokens[-1], module)


def extract_dag(
        model: nn.Module,
        dummy_input=None,
        hook_nodes=None,
        anti_method=None):
    return TorchDAGAdapter(model, dummy_input, hook_nodes=hook_nodes, anti_method=anti_method)


def norm_class_detect(
        model: nn.Module,
):
    norm_class_list = list(
        OrderedDict.fromkeys([m.__class__ for m in model.modules() if "norm" in m.__class__.__name__.lower()]))
    norm_class = [norm_class_list[0]]

    return norm_class


def class_detect(
        model: nn.Module,
        name: str,
):
    class_list = list(
        OrderedDict.fromkeys([m.__class__ for m in model.modules() if name in m.__class__.__name__.lower()]))
    class_name = class_list[0] if len(class_list) != 0 else None

    return class_name


def input_to_cpu(ipt):
    return DagTorchHook.input_to_cpu(ipt)


def get_module_name_list(dag, model_type):
    if model_type.title() == "Llama":
        attn_list, proj_list, mhsa_ln_list = dag.get_llama_mhsa_ln_pattern()
        ffn_list, ffn_ln_list = dag.get_llama_ffn_ln_pattern()
    else:
        attn_list, proj_list, mhsa_ln_list = dag.get_mhsa_ln_pattern()
        ffn_list, ffn_ln_list = dag.get_ffn_ln_pattern()
    ret = attn_list, proj_list, mhsa_ln_list, ffn_list, ffn_ln_list
    return ret


class PatternProcess:
    @staticmethod
    def get_attn_name(qkv_name):
        qkv_combined = 1
        qkv_separate = 3
        if len(qkv_name) == qkv_combined:
            attn_name = '.'.join(qkv_name[0].split('.')[:-1])
        elif len(qkv_name) == qkv_separate:
            attn_name = os.path.commonprefix(qkv_name)[:-1]
        else:
            raise ValueError("length of qkv_name should be equal to 1 or 3.")
        return attn_name

    @staticmethod
    def get_module_by_name(model, submodule_key=None):
        if submodule_key is None:
            return submodule_key
        tokens = submodule_key.split('.')
        cur_mod = model
        for s in tokens:
            cur_mod = getattr(cur_mod, s, None)
        return cur_mod

    @classmethod
    def get_qkv_name(cls, attn_list):
        qkv = []
        for qkv_name in attn_list:
            if attn_list is None:
                return attn_list
            attn_name = cls.get_attn_name(qkv_name) + '.'
            for item in qkv_name:
                qkv.append(item.replace(attn_name, ''))
            break
        return qkv


def replace_conv1d(model):
    for name, module in model.named_modules():
        if "c_attn" in name or "c_proj" in name or "c_fc" in name:
            new_module = nn.Linear(module.weight.size(0), module.weight.size(1))
            new_module.bias.data.copy_(module.bias.data)
            new_module.bias.data = new_module.bias.data.type(module.bias.data.dtype)
            new_module.weight.data.copy_(torch.permute(module.weight.data, (1, 0)))
            new_module.weight.data = new_module.weight.data.type(module.weight.data.dtype)
            GraphOpt.set_module(model, name, new_module)
    return model


class LlamaRMSNormBias(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states + self.bias


class NormBias(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        if not hasattr(module, "weight"):
            raise AttributeError(f"norm_class {module.__class__.__name__} don't have weight variable.")
        self.weight = nn.Parameter(copy.deepcopy(module.weight))
        hidden_size = module.weight.size(0)
        self.bias = nn.Parameter(torch.zeros(hidden_size).to(module.weight.device))

    def forward(self, hidden_states):
        hidden_states = self.module(hidden_states)
        hidden_states += self.bias

        return hidden_states


def replace_rmsnorm(model):
    for name, module in model.named_modules():
        if "norm" in name:
            new_module = LlamaRMSNormBias(module.weight.size(0), module.variance_epsilon)
            new_module.weight.data = new_module.weight.data.type(module.weight.data.dtype)
            new_module.bias.data = new_module.bias.data.type(module.weight.data.dtype)
            new_module.to(module.weight.data.device)
            GraphOpt.set_module(model, name, new_module)
    return model
