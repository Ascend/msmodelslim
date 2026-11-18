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


from collections import OrderedDict, defaultdict
from abc import ABC, abstractmethod

import torch

import torch.nn as nn

from megatron.inference.text_generation import generate
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.legacy.model.fused_layer_norm import MixedFusedLayerNorm
from megatron.core.tensor_parallel.mappings import (
    reduce_from_tensor_model_parallel_region,
    )
from megatron.training import get_args

from ascend_utils.common.security import check_dict_element, check_type
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator
from msmodelslim.pytorch.llm_ptq.anti_outlier.dag_utils.torch_dag_adapter import TorchDAGAdapter

_SUPPORTED_DEVICES = ["npu", 'gpu']
_SUPPORTED_PREFIX = ["model.", "model.module."]


def get_norm_linear_subgraph(self):
    norm_linear_subgraph = defaultdict(list)

    self.node_list = [node for node in self.node_list if "attn_linear" not in node.name_in_network]

    norm_positions = [i for i, x in enumerate(self.node_list) if x.op_type.lower() in self.norm_nodes]
    num_norm = len(norm_positions)

    for i in range(num_norm - 1):
        start = norm_positions[i]
        end = norm_positions[i + 1]
        interval_linears = [node.name_in_network for node in self.node_list[start+1:end-1]]
        norm_node = self.node_list[start].name_in_network
        
        if len(interval_linears) <= 4:
            norm_linear_subgraph[norm_node].extend(interval_linears)
        else:
            qkv_linears = [node.name_in_network for node in self.node_list[start+1:start+4]]
            norm_linear_subgraph[norm_node].extend(qkv_linears)
    norm_linear_subgraph = {k: v for k, v in norm_linear_subgraph.items() if len(v) > 0}
    return norm_linear_subgraph


def modelslim_adaption():
    from mindspeed.patch_utils import MindSpeedPatchesManager as aspm
    string1 = 'msmodelslim.pytorch.llm_ptq.anti_outlier.dag_utils.torch_dag_adapter.'
    string2 = 'TorchDAGAdapter.get_norm_linear_subgraph'
    patch_string = string1 + string2
    aspm.register_patch(patch_string, get_norm_linear_subgraph)
    aspm.apply_patches()


modelslim_adaption()


class Forward(ABC):
    @abstractmethod
    def __call__(self, model, *args, **kwargs):
        pass


class GenerateForward(Forward):
    def __call__(self, model, x):
        if isinstance(x[0], list):
            return generate(model, x[0], tokens_to_generate=1)
        else:
            return generate(model, x, tokens_to_generate=1)


class ModelGenerateForward(Forward):
    def __call__(self, model, x):
        args = get_args()
        max_new_tokens = args.max_new_tokens
        res = model.generate(x, max_new_tokens=1)
        args.max_new_tokens = max_new_tokens
        return res


def set_module(ori_mod, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = ori_mod
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


class ModelAdapter(nn.Module):
    def __init__(self, model: nn.Module, dev_type='npu', forward_step=None, prefix="model."):
        super(ModelAdapter, self).__init__()
        check_type(model, nn.Module, param_name="model")
        if forward_step and not callable(forward_step):
            raise ValueError("forward_step function must callable")
        self.model = self.convert_model(model)
        if dev_type not in _SUPPORTED_DEVICES:
            raise ValueError("Configuration param `dev_type` cannot be correctly parsed! "
                             "Please make sure a valid device type is input")
        self.device = dev_type
        self.forward_step = forward_step
        if hasattr(model, 'args'):
            self.config = model.args
            if self.forward_step is None:
                self.forward_step = GenerateForward()
        elif hasattr(model, 'config'):
            self.config = model.config
            if self.forward_step is None:
                self.forward_step = ModelGenerateForward()
        else:
            raise ValueError("Model does't find a config!")
        self.config.torch_dtype = self.config.params_dtype
        self.dtype = self.config.params_dtype
        check_type(prefix, str, "prefix")
        if prefix not in _SUPPORTED_PREFIX:
            raise ValueError("doesn't support this prefix! only support prefix options: 'model.' or 'model.module.'")
        self.prefix = prefix
    
    def convert_model(self, model):
        for name, mod in model.named_modules():
            if isinstance(mod, (ColumnParallelLinear, RowParallelLinear)):
                mod_adapter = MegatronLinearAdapter(mod)
                set_module(model, name, mod_adapter)
            if isinstance(mod, MixedFusedLayerNorm):
                new_mod = nn.LayerNorm(mod.normalized_shape, eps=mod.eps)
                new_mod.weight = mod.weight
                new_mod.bias = mod.bias
                set_module(model, name, new_mod)
        return model

    def state_dict(self, prefix=""):
        if prefix == '':
            org_state = self.model.state_dict(prefix=self.prefix)
        else:
            org_state = self.model.state_dict(prefix=prefix)
        state = OrderedDict()
        for key, value in org_state.items():
            if value is not None:
                state[key] = value
        return state
    
    def forward(self, *args, **kwargs):
        return self.forward_step(self.model, *args, **kwargs)


class CalibratorAdapter(Calibrator):
    def extract_dag(self, model):
        if not self.init_dag():
            return None
        if not self.calib_data:
            dummy_input = torch.randint(0, 100, (1, 128)).type(torch.int64)
        else:
            dummy_input = self.calib_data[0]
        norm_class = self.get_norm_class(model, norm_class_name=self.norm_class_name)
        norm_class.append(Linear)
        dag = TorchDAGAdapter(model, dummy_input, hook_nodes=norm_class)
        return dag
    
    def get_ori_model_weight(self, model, cfg):
        ori_fp_weight = {}
        for key, value in model.state_dict().items():
            if 'norm.module.weight' in key.lower():
                key_list = key.split('.')
                key_list = key_list[:-2] + [key_list[-1]]
                key = '.'.join(key_list)
            if 'norm.bias' in key.lower():
                keys = model.state_dict().keys()
                key_list = key.split('.')[:-1]
                tmp_key = '.'.join(key_list) + '.module.weight'
                if tmp_key in keys:
                    continue
            if not isinstance(value, torch.Tensor):
                self.logger.warning("The original float weight[{key}]is not torch.Tensor, "
                                    "it won't be saved, may raise error.")
                continue
            ori_fp_weight[key] = value.cpu()
        check_dict_element(ori_fp_weight, value_type=torch.Tensor, param_name='ori_fp_weight',
                              additional_msg="Failed to get original float weight, please check the model.")
        return ori_fp_weight


class MegatronLinearAdapter(nn.Module):
    def __init__(self, linear):
        super(MegatronLinearAdapter, self).__init__()
        self._adapter_linear = Linear(linear)
        if hasattr(linear, 'quant_params'):
            self.quant_params = linear.quant_params
        self.skip_bias_add = linear.skip_bias_add
        self.is_row = isinstance(linear, RowParallelLinear)

    @property
    def weight(self):
        return self._adapter_linear.weight
    
    @property
    def bias(self):
        return self._adapter_linear.bias
    
    def get_linear(self):
        if hasattr(self._adapter_linear, "get_linear"):
            return self._adapter_linear.get_linear()
        else:
            return self._adapter_linear
    
    def forward(self, x):
        res = self._adapter_linear(x)
        if self.is_row and not isinstance(self.get_linear(), RowParallelLinear):
            res = reduce_from_tensor_model_parallel_region(res)
        output_bias = self.bias
        return res, output_bias
    

class Linear(nn.Linear):
    def __init__(self, linear):
        super(Linear, self).__init__(linear.input_size, linear.output_size, bias=False)
        self._adapter_linear = linear
        self.in_features = linear.input_size
        self.out_features = linear.output_size
        self.weight = None
        self.bias = None

    @property
    def weight(self):
        return self._adapter_linear.weight
    
    @property
    def bias(self):
        return self._adapter_linear.bias
    
    def forward(self, x):
        return self._adapter_linear(x)[0]
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        if keep_vars:
            destination[prefix + 'weight'] = self.weight
            if self.bias is not None:
                destination[prefix + 'bias'] = self.bias
        else:
            destination[prefix + 'weight'] = self.weight.detach()
            if self.bias is not None:
                destination[prefix + 'bias'] = self.bias.detach()
        return destination
    
    def load_state_dict(self, state_dict, strict=True):
        self._adapter_linear.load_state_dict(state_dict, strict)

    def get_linear(self):
        return self._adapter_linear
