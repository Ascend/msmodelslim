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

import functools
import re

import torch
import torch.nn as nn
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.models.model_utils import (
    TransformerStructurePairVisitor, 
    get_module_by_name, 
    set_module_by_name
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.models.model_utils import (
    StructurePair, 
    AttnNormLinearPair, 
    AttnLinearLinearPair, 
    MLPNormLinearPair, 
    MLPLinearLinearPair, 
    ModelStructureBridge
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.components.flat_linear import (
    FlatQuantizedLinear, 
    FlatNormWrapper, 
    FakeQuantizedLinearConfig, 
    FakeQuantizedLinear
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.components.trans import GeneralMatrixTrans
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.utils.function_utils import (
    get_decompose_dim, 
    get_init_scale
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.processors.quantizer_manager import QuantizerMapper
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.components.flat_linear import FakeQuantizedLinearConfig


from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType


def stat_input_hook(m, x, y, name, act_stats):
    if isinstance(x, tuple):
        x = x[0]
    stat_tensor(act_stats, name, x)


def stat_tensor(act_stats, name, x):
    if 'input_max' not in act_stats[name]:
        act_stats[name]['input_max'] = x.view(-1, x.shape[-1]).abs().max(0)[0].clone().detach().cpu()
    else:
        tmp = x.view(-1, x.shape[-1]).abs().max(0)[0].clone().detach().cpu()
        act_stats[name]['input_max'] = torch.maximum(act_stats[name]['input_max'], tmp)


class FakeQuantizerVisitor(TransformerStructurePairVisitor): 
    def __init__(self, model, config: FakeQuantizedLinearConfig):
        self.model = model
        self.config = config
        self.quantizer_dict = {}
        self.linear_quantizer = FakeQuantizedLinear
        self.act_stats = {}
        self.hooks = {}

    def register_forward_hook(self, mod, name):
        self.act_stats[name] = {}
        self.act_stats[name]['input_max'] = torch.full([mod.weight.shape[1]], 1e-5, dtype=mod.weight.dtype)
        self.hooks[name] = mod.register_forward_hook(functools.partial(stat_input_hook, 
                                                                        name=name, 
                                                                        act_stats=self.act_stats))

    def remove_forward_hook(self, prefix=""):
        for name, hook in self.hooks.items():
            if prefix and not name.startswith(prefix):
                continue
            hook.remove()

    def set_quant_config(self, config: FakeQuantizedLinearConfig):
        self.config = config

    def visit_attn_norm_linear_pair(self, pair: AttnNormLinearPair):
        """访问注意力层norm和线性层结构对"""
        self._visit_linear_pair(pair)

    def visit_attn_linear_linear_pair(self, pair: AttnLinearLinearPair):
        """访问注意力层线性层和线性层结构对"""
        self._visit_linear_pair(pair)

    def visit_mlp_norm_linear_pair(self, pair: MLPNormLinearPair):
        """访问MLP层norm和线性层结构对"""
        self._visit_linear_pair(pair)

    def visit_mlp_linear_linear_pair(self, pair: MLPLinearLinearPair):
        """访问MLP层线性层和线性层结构对"""
        self._visit_linear_pair(pair)

    def to_org_mode(self, prefix=""):
        """设置模型为原始模式"""
        for name, quantizer in self.quantizer_dict.items():
            if prefix and not name.startswith(prefix):
                continue
            quantizer.to_org_mode()

    def to_calib_mode(self, prefix=""):
        """设置模型为训练模式"""
        self.remove_forward_hook(prefix)
        for name, quantizer in self.quantizer_dict.items():
            if prefix and not name.startswith(prefix):
                continue
            quantizer.to_calib_mode()

    def fake_quant_weight(self, prefix=""):
        """量化权重"""
        for name, quantizer in self.quantizer_dict.items():
            if prefix and not name.startswith(prefix):
                continue
            quantizer.fake_quant_weight()

    def to_eval_mode(self, prefix="", quant_weight=True):
        """设置模型为评估模式"""
        self.remove_forward_hook(prefix)
        for name, quantizer in self.quantizer_dict.items():
            if prefix and not name.startswith(prefix):
                continue
            quantizer.to_eval_mode()
        if quant_weight:
            self.fake_quant_weight(prefix=prefix)

    def _visit_linear_pair(self, pair: StructurePair):
        """访问norm和线性层结构对"""
        linear_names = pair.target_modules
        clip_factor = nn.Parameter(torch.ones((1, )) * 1.0, requires_grad=True)
        for linear_name in linear_names:
            linear_module = get_module_by_name(self.model, linear_name)
            flat_linear = self.linear_quantizer(self.config, linear_module)
            self.register_forward_hook(flat_linear, linear_name)
            set_module_by_name(self.model, linear_name, flat_linear)
            self.quantizer_dict[linear_name] = flat_linear
            flat_linear.set_act_clip_factor(clip_factor)


class FlatQuantQuantizerConfig(FakeQuantizedLinearConfig):
    def __init__(self, w_bits=16, 
                        a_bits=16,
                        w_asym=False, 
                        a_asym=False, 
                        lwc=False, 
                        lac=False, 
                        a_groupsize=-1, 
                        a_per_tensor=False, 
                        add_diag=True, 
                        diag_alpha=0.5, 
                        diag_relu=False, 
                        tran_type="svd"):
        super(FlatQuantQuantizerConfig, self).__init__(w_bits, 
                                                        a_bits, 
                                                        w_asym, 
                                                        a_asym, 
                                                        lwc, 
                                                        lac, 
                                                        a_groupsize, 
                                                        a_per_tensor)
        self.add_diag = add_diag
        self.diag_alpha = diag_alpha
        self.diag_relu = diag_relu
        self.tran_type = tran_type


class FlatQuantQuantizerMapVisitor(FakeQuantizerVisitor):
    def __init__(self, model, config: FlatQuantQuantizerConfig):
        super(FlatQuantQuantizerMapVisitor, self).__init__(model, config)
        self.decompose_trans_dict = {}
        self.norm_dict = {}
        self.add_diag = config.add_diag
        self.linear_quantizer = FlatQuantizedLinear
        self.diag_alpha = config.diag_alpha
        self.diag_relu = config.diag_relu
        self.tran_type = config.tran_type

    """FlatQuant量化器访问者"""
    def visit_attn_norm_linear_pair(self, pair: AttnNormLinearPair):
        """访问注意力层norm和线性层结构对"""
        super(FlatQuantQuantizerMapVisitor, self).visit_attn_norm_linear_pair(pair)
        self._visit_norm_linear_pair(pair)

    def visit_attn_linear_linear_pair(self, pair: AttnLinearLinearPair):
        """访问注意力层线性层和线性层结构对"""
        super(FlatQuantQuantizerMapVisitor, self).visit_attn_linear_linear_pair(pair)
        config = pair.config
        pre_linear_name = pair.source_modules
        pre_linear_module = get_module_by_name(self.model, pre_linear_name)
        if hasattr(config, 'head_dim'):
            head_dim = config.head_dim
        else:
            if config.num_attention_heads == 0:
                raise ValueError("num_attention_heads can not be zero in config.")
            head_dim = config.hidden_size // config.num_attention_heads
        trans = GeneralMatrixTrans(config.num_attention_heads, 
                                    head_dim, 
                                    add_diag=False, 
                                    diag_relu=self.diag_relu, 
                                    tran_type=self.tran_type)
        pre_linear_module.set_trans(weight_out_trans=trans.right_trans)
        self.decompose_trans_dict[pair] = trans

        for linear_name in pair.target_modules:
            linear_module = get_module_by_name(self.model, linear_name)
            linear_module.set_trans(weight_in_trans=trans, act_in_trans=trans.left_trans, save_trans=trans.left_trans)

    def visit_mlp_norm_linear_pair(self, pair: MLPNormLinearPair):
        """访问MLP层norm和线性层结构对"""
        super(FlatQuantQuantizerMapVisitor, self).visit_mlp_norm_linear_pair(pair)
        self._visit_norm_linear_pair(pair)

    def visit_mlp_linear_linear_pair(self, pair: MLPLinearLinearPair):
        """访问MLP层线性层和线性层结构对"""
        super(FlatQuantQuantizerMapVisitor, self).visit_mlp_linear_linear_pair(pair)
        pre_linear_name = pair.source_modules
        pre_linear_module = get_module_by_name(self.model, pre_linear_name)
        pre_dim_left, pre_dim_right = get_decompose_dim(pre_linear_module.weight.shape[0])
        linear_trans = GeneralMatrixTrans(pre_dim_left, 
                                            pre_dim_right, 
                                            add_diag=self.add_diag, 
                                            diag_relu=self.diag_relu, 
                                            tran_type=self.tran_type)
        self.decompose_trans_dict[pair] = linear_trans

        for linear_name in pair.target_modules:
            linear_module = get_module_by_name(self.model, linear_name)
            linear_module.set_trans(weight_in_trans=linear_trans, 
                                    act_in_trans=linear_trans,
                                    save_trans=linear_trans)

    def to_org_mode(self, prefix=""):
        """设置模型为原始模式"""
        super(FlatQuantQuantizerMapVisitor, self).to_org_mode(prefix)
        for name, quantizer in self.norm_dict.items():
            if prefix and not name.startswith(prefix):
                continue
            quantizer.to_org_mode()


    def to_calib_mode(self, prefix=""):
        """设置模型为校准模式"""
        super(FlatQuantQuantizerMapVisitor, self).to_calib_mode(prefix)
        for name, quantizer in self.norm_dict.items():
            if prefix and not name.startswith(prefix):
                continue
            quantizer.to_calib_mode()   
        self._init_diag_scale(prefix, diag_alpha=self.diag_alpha)


    def to_eval_mode(self, prefix="", quant_weight=True):
        """设置模型为评估模式"""
        super(FlatQuantQuantizerMapVisitor, self).to_eval_mode(prefix, quant_weight=False)
        for name, quantizer in self.norm_dict.items():
            if prefix and not name.startswith(prefix):
                continue
            quantizer.to_eval_mode()
        for pair, trans in self.decompose_trans_dict.items():
            if prefix and not str(pair).startswith(prefix):
                continue
            self._reparameterize_act_diag_scale(trans, pair)
            trans.to_eval_mode()
        if quant_weight:
            self.fake_quant_weight(prefix=prefix)

    def _init_diag_scale(self, prefix="", diag_alpha=0.5):
        """初始化对角线尺度"""
        for pair, trans in self.decompose_trans_dict.items():
            if trans.diag_trans is None:
                continue
            pre_linear_name = pair.source_modules
            post_linear_names = pair.target_modules
            if prefix and not pre_linear_name.startswith(prefix):
                continue
            weights = []
            for linear_name in post_linear_names:
                linear_module = get_module_by_name(self.model, linear_name)
                weights.append(linear_module.weight)
                input_max = self.act_stats[linear_name].get('input_max', None)
            if input_max is None:
                input_max = torch.full([linear_module.weight.shape[1]], 1e-5, dtype=linear_module.weight.dtype)
            weights_max = torch.cat(weights, dim=0).abs().max(dim=0)[0]
            weights_max = weights_max.to(trans.diag_trans.diag_scale)
            input_max = input_max.to(trans.diag_trans.diag_scale)
            trans.diag_trans.diag_scale.data = get_init_scale(weights_max, input_max, diag_alpha)

    def _reparameterize_act_diag_scale(self, trans: GeneralMatrixTrans, pair: StructurePair):
        if trans.diag_trans is not None:
            pre_linear_name = pair.source_modules
            pre_linear_module = get_module_by_name(self.model, pre_linear_name)
            weight = pre_linear_module.weight.data
            ori_dtype = weight.dtype
            if weight.dim() == 2:
                weight = weight.to(torch.float32) * trans.diag_trans.diag_scale.data.to(torch.float32).unsqueeze(1)
            elif weight.dim() == 1:
                weight = weight.to(torch.float32) * trans.diag_trans.diag_scale.data.to(torch.float32)
            else:
                raise ValueError(f"weight dim is not supported: {weight.dim()}")
            pre_linear_module.weight.data = weight.to(ori_dtype)

    def _visit_norm_linear_pair(self, pair: StructurePair):
        """访问norm和线性层结构对"""
        norm_name = pair.source_modules
        norm_module = get_module_by_name(self.model, norm_name)

        ln_dim_left, ln_dim_right = get_decompose_dim(norm_module.weight.shape[0])

        ln_trans = GeneralMatrixTrans(ln_dim_left, 
                                        ln_dim_right, 
                                        add_diag=self.add_diag, 
                                        diag_relu=self.diag_relu, 
                                        tran_type=self.tran_type)

        self.decompose_trans_dict[pair] = ln_trans
        flat_norm = FlatNormWrapper(norm_module, ln_trans)
        set_module_by_name(self.model, norm_name, flat_norm)
        self.norm_dict[norm_name] = flat_norm
        linear_names = pair.target_modules
        for linear_name in linear_names:
            linear_module = get_module_by_name(self.model, linear_name)
            linear_module.set_trans(weight_in_trans=ln_trans, save_trans=ln_trans)


def get_n_set_parameters_byname(model, required_names):
    params = []
    for r_name in required_names:
        for name, param in model.named_parameters():
            if name.find(r_name) > -1:
                params.append(param)
    for param in params:
        param.requires_grad = True
    return params


def get_trainable_parameters(model, base_lr=3e-5):
    """获取可训练参数"""
    params = {}
    params["linear_u"] = get_n_set_parameters_byname(model, ['linear_u'])
    params["linear_v"] = get_n_set_parameters_byname(model, ['linear_v'])
    params["trans_linear"] = get_n_set_parameters_byname(model, ['trans_linear'])
    params["linear_diag"] = get_n_set_parameters_byname(model, ['linear_diag'])
    params["diag_scale"] = get_n_set_parameters_byname(model, ['diag_scale'])
    params["clip_factor"] = get_n_set_parameters_byname(model, ['clip_factor'])
    params["round_value"] = get_n_set_parameters_byname(model, ['round_value'])
    trainable_params = [{"params": params["linear_u"], "lr": base_lr}]
    trainable_params.append({"params": params["linear_v"], "lr": base_lr})
    trainable_params.append({"params": params["trans_linear"], "lr": base_lr})
    trainable_params.append({"params": params["linear_diag"], "lr": base_lr})
    trainable_params.append({"params": params["diag_scale"], "lr": base_lr})
    trainable_params.append({"params": params["clip_factor"], "lr": base_lr * 10})
    trainable_params.append({"params": params["round_value"], "lr": base_lr * 10}) 
    need_train = any(len(value) > 0 for value in params.values())
    return params, trainable_params, need_train


def convert_config(quant_config, flat_quant_config, model):
    if quant_config.model_quant_type == QuantType.W4A4_FLATQUANT_DYNAMIC:
        return flat_quant_config, FlatQuantQuantizerMapVisitor(model, flat_quant_config)
    elif quant_config.model_quant_type == QuantType.W8A8_DYNAMIC or quant_config.model_quant_type == QuantType.W8A8:
        config = FlatQuantQuantizerConfig(w_bits=quant_config.w_bit,
                                            a_bits=quant_config.a_bit,
                                            w_asym=not quant_config.w_sym,
                                            a_asym=not quant_config.a_sym,
                                            lac=False,
                                            lwc=False,
                                            a_per_tensor=not quant_config.is_dynamic
                                            )
        return config, FakeQuantizerVisitor(model, config)
    elif quant_config.model_quant_type == QuantType.FLOAT:
        return None, None
    else:
        raise ValueError(f"Invalid mix quant_config when quantizing W4A4: {quant_config.model_quant_type}")


def quantize_model(model_bridge: ModelStructureBridge, layer_map, flat_quant_config: FlatQuantQuantizerConfig):
    pairs_dict = model_bridge.get_structure_pairs()
    pairs = []
    support_structure_pairs = [AttnNormLinearPair, AttnLinearLinearPair, MLPNormLinearPair, MLPLinearLinearPair]
    num = max([len(pairs_dict[pair_type.__name__]) for pair_type in support_structure_pairs])
    for i in range(num):
        # to keep the nature order of support_structure_pairs
        for pair_type in support_structure_pairs:
            if i < len(pairs_dict[pair_type.__name__]):
                pairs.append(pairs_dict[pair_type.__name__][i])

    mapper = QuantizerMapper()
    
    for pair in pairs:
        quant_config = [quant_config for layer_name, quant_config in layer_map.items() if pair.contain(layer_name)]
        quant_type = set([quant_config.model_quant_type for quant_config in quant_config])
        if len(quant_config) > 0:
            if len(quant_type) > 1:
                raise ValueError(f"Find different quant type in structure {pair.name},"
                                    "please check your mix-quant or rollback config")
            _, quant_visitor = convert_config(quant_config[0], flat_quant_config, model_bridge.model)
            if quant_visitor is not None:
                mapper.register_pattern(pair.name, quant_visitor)
    mapper.apply_quantizer(pairs_dict)
    return mapper

