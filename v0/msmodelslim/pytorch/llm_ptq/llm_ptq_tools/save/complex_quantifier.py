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

from typing import TYPE_CHECKING

import torch
import numpy as np

# KIA part
from msmodelslim.pytorch.llm_sparsequant.atomic_power_outlier import quant_one_weight_by_outliers
from msmodelslim.pytorch.lowbit.calibration import LlamaRMSNormBias
from msmodelslim.pytorch.lowbit.quant_modules import LinearQuantizer as LowBitLinearQuantizer
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs import fake_quantize_save

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.layer_config_manager import LayerConfigManager
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.quantizer import LinearQuantizerTimestep
from msmodelslim.pytorch.llm_sparsequant.sparsequant_modules import LinearSparseQuantizer
from msmodelslim.pytorch.llm_ptq.anti_outlier.graph_utils import NormBias
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_modules import LinearQuantizer, LinearNf4Quantizer
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import (
    export_fa_quant_params,
    is_attn_module_and_then_check_quantizer
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.simulate_tp import ParallelLinearCol
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.components import (
    FakeQuantizedLinear, 
    FlatNormWrapper, 
    asym_quant
)


def deqscale_process(input_scale, scale):
    deq_scale = input_scale * scale
    if deq_scale.ndim > 1:
        deq_scale = deq_scale.squeeze(1)
    deq_scale = deq_scale.cpu()
    return deq_scale


def change_bias(fp_weight, module):
    if module.bias is None:
        bias_shape = fp_weight.shape[0]
        fp_bias = torch.zeros(bias_shape)
    else:
        fp_bias = module.bias.cpu()
    return fp_bias


def deqscale2int64(scale):
    scale = scale.numpy()
    scale = np.frombuffer(scale.tobytes(), dtype=np.int32).astype(np.int64)
    scale = torch.tensor(scale)
    return scale


def deqscale2int64_by_dtype(scale, is_bf16):
    if is_bf16:
        return scale
    else:
        return deqscale2int64(scale)


def _get_module_quant_input(module):
    fp_weight = module.weight.cpu()
    weight_scale, weight_offset = module.quant_weight.weight_scale, module.quant_weight.weight_offset
    device = weight_scale.device if weight_scale is not None else None
    scale = weight_scale.cpu() if weight_scale is not None else None
    offset = weight_offset.cpu() if weight_offset is not None else None
    round_opt = False if isinstance(module, LowBitLinearQuantizer) else module.quant_weight.round_opt

    if hasattr(module.quant_weight, 'weight_scale_second') and \
            hasattr(module.quant_weight, 'weight_offset_second'):
        weight_scale_second = module.quant_weight.weight_scale_second
        weight_offset_second = module.quant_weight.weight_offset_second
        scale_second = weight_scale_second.cpu() if weight_scale_second is not None else None
        offset_second = weight_offset_second.cpu() if weight_offset_second is not None else None
        if scale_second is not None and offset_second is not None:
            scale = [scale, scale_second]
            offset = [offset, offset_second]

    ret = fp_weight, device, scale, offset, round_opt
    return ret


# for clean code
def generate_weight_of_rms_norm_module(name, module, model_quant_type, is_new_versoin=False):
    anti_norm_weight: torch.Tensor = module.weight
    anti_norm_bias: torch.Tensor = module.bias
    yield name + '.weight', QuantType.FLOAT, anti_norm_weight.clone().cpu()
    yield name + '.bias', QuantType.FLOAT, anti_norm_bias.clone().cpu()
    if not is_new_versoin:
        yield name + '.module.weight', model_quant_type, anti_norm_weight.clone().cpu()
        yield name + '.module.bias', model_quant_type, anti_norm_bias.clone().cpu()


# for clean code
def generate_weight_of_fa_module(name, module):
    quant_param_scale, quant_param_offset, _ = export_fa_quant_params(module, name)
    for name, param in quant_param_scale.items():
        yield name, QuantType.FAQuant, param
    for name, param in quant_param_offset.items():
        yield name, QuantType.FAQuant, param


# for clean code
def generate_weight_of_nf_module(name, module):
    yield name + '.weight', QuantType.NF4, module.weight
    if module.bias is not None:
        yield name + '.bias', QuantType.NF4, module.bias


def _pack_int4(weight) -> torch.Tensor:
    """
    Pack int4 weight to int8 weight
    @param weight: torch.Tensor, int4 weight
    @return: torch.Tensor, int8 weight
    """
    weight = weight.to(torch.int8)
    e = 0  # number of experts
    if len(weight.shape) == 2:
        k, n = weight.shape
    elif len(weight.shape) == 3:
        e, k, n = weight.shape
    n_new = n // 2 + n % 2

    if n_new != n // 2:
        raise AssertionError("n dimension should be even")
    
    weight = weight.reshape(-1, 2)
    weight0 = weight[:, :1]
    weight1 = weight[:, 1:]

    weight1_4 = torch.bitwise_left_shift(weight1, 4)
    weight2_4 = weight0 & 0b00001111

    weight_add = torch.bitwise_or(weight1_4, weight2_4)
    if e == 0:
        weight_res = weight_add.reshape(k, n_new)
    else:
        weight_res = weight_add.reshape(e, k, n_new)
    return weight_res


def _w4a16_pack_int4(save_quant_weight, trans_flag):
    """
    Pack int4 weight to int8 weight
    @param save_quant_weight: torch.Tensor, int4 weight
    @param trans_flag: bool, whether to transpose the weight
    @return: torch.Tensor, int8 weight
    """

    # per group 不推荐 Transpose，per channel 推荐 Transpose
    weight_in_k_n = save_quant_weight.transpose(-1, -2).contiguous()
    weight_trans = weight_in_k_n if not trans_flag else weight_in_k_n.transpose(-1, -2).contiguous()
    save_quant_weight = _pack_int4(weight=weight_trans)
    if not trans_flag:
        save_quant_weight = save_quant_weight.transpose(-1, -2).contiguous()
    return save_quant_weight


def _w4a8_pack_int4(save_quant_weight):
    """
    Pack int4 weight to int8 weight
    @param save_quant_weight: torch.Tensor, int4 weight
    @return: torch.Tensor, int8 weight
    """
    weight = save_quant_weight.transpose(-1, -2).contiguous()
    packed_weight_tensor = _pack_int4(weight)
    packed_weight_tensor = packed_weight_tensor.transpose(-1, -2).contiguous()
    return packed_weight_tensor


def process_scale(name, bias, tp_num):
    """
    Pack int4 weight to int8 weight
    @param name: 输入tensor名
    @param bias: sum 前bias
    @param tp_num: 推理时tp数
    @return: bias, fp32格式gmm算子所需的偏置量
    """
    if any(char in name for char in ['up_proj', 'gate_proj', 'q_proj', 'k_proj', 'v_proj']):
        up_bias = bias
        up_bias = 8 * up_bias.sum(dim=1, keepdim=True)
        bias = up_bias

    elif any(char in name for char in ['down_proj', 'o_proj']):
        pre_shape = bias.shape[0]
        sum_shape = bias.shape[1] // tp_num
        down_bias = bias.reshape(-1, sum_shape)
        down_bias = 8 * down_bias.sum(dim=1, keepdim=True)
        bias = down_bias.reshape(pre_shape, -1)
    return bias


class ComplexQuantifier:
    def __init__(self, cfg, rollback_names, torch_dtype, layer_cfg_manager, is_new_version=False):
        self.cfg = cfg
        self.rollback_names = rollback_names
        self.torch_dtype = torch_dtype
        self.layer_cfg_manager: LayerConfigManager = layer_cfg_manager

        self.skip_module_names = set()

        self.is_new_version = is_new_version

    def generate_weight_of_module(self, name: str, module: torch.nn.Module):
        if name in self.skip_module_names:
            return

        model_quant_type = self.layer_cfg_manager.get_layer_config(name).model_quant_type
        if self.cfg.use_fa_quant and is_attn_module_and_then_check_quantizer(module, name):
            yield from generate_weight_of_fa_module(name, module)
        elif isinstance(module, LinearNf4Quantizer):
            yield from generate_weight_of_nf_module(name, module)
        elif isinstance(module, ParallelLinearCol):
            yield from self.generate_weight_of_tp_module(name, module, model_quant_type)
        # 处理 Norm 对应的 weight、bias
        elif isinstance(module, NormBias):
            yield from self.generate_weight_of_norm_module(name, module, model_quant_type)
        elif isinstance(module, LlamaRMSNormBias):
            yield from generate_weight_of_rms_norm_module(name, module, model_quant_type, self.is_new_version)
        # 处理Linear、以及附属scale、offset等params
        elif isinstance(module, (LinearQuantizer, LinearSparseQuantizer, LowBitLinearQuantizer)):
            yield from self.generate_weight_of_linear_module(name, module, model_quant_type)
        elif isinstance(module, FakeQuantizedLinear):
            yield from self.generate_weight_of_flat_linear_module(name, module, model_quant_type)
        elif isinstance(module, FlatNormWrapper):
            weight = module.norm.weight
            yield name + '.weight', QuantType.FLOAT, weight.cpu()
        else:
            for key, param in module.named_parameters(name, recurse=False):
                yield key, QuantType.FLOAT, param
    
    def generate_weight_of_flat_linear_module(self, name, module, model_quant_type):
        weight_scale, weight_offset = module.weight_quantizer.get_scale_zero(module.weight)
        quant_weight = asym_quant(module.weight, weight_scale, weight_offset, module.weight_quantizer.bits, True)[0]
        yield name + '.weight', model_quant_type, quant_weight.cpu().to(torch.int8)
        if hasattr(module, 'bias') and module.bias is not None:
            yield name + '.bias', QuantType.FLOAT, module.bias
        if model_quant_type == QuantType.W8A8_DYNAMIC or model_quant_type == QuantType.W4A4_FLATQUANT_DYNAMIC:
            yield name + '.weight_scale', model_quant_type, weight_scale.cpu()
            yield name + '.weight_offset', model_quant_type, weight_offset.cpu()
            clip_ratio = module.act_quantizer.get_clip_ratio()
            if clip_ratio is not None:
                yield name + '.clip_ratio', QuantType.W4A4_FLATQUANT_DYNAMIC, clip_ratio.cpu()
        elif model_quant_type == QuantType.W8A8:
            input_scale, input_offset = module.act_quantizer.get_scale_zero(None)
            yield name + '.input_scale', model_quant_type, input_scale.cpu()
            yield name + '.input_offset', model_quant_type, input_offset.cpu()

            input_scale = input_scale.to(device=quant_weight.device)
            input_offset = input_offset.to(device=quant_weight.device)
            deq_scale = deqscale_process(input_scale, weight_scale).to(torch.float32)
            quant_weight = quant_weight.to(torch.float32)
            input_offset = input_offset.to(torch.float32)
            correction = (quant_weight.sum(dim=1) * input_offset).cpu()
            fp_bias = change_bias(module.weight, module)
            quant_bias = torch.round(fp_bias / deq_scale - correction)
            deq_scale = deqscale2int64_by_dtype(deq_scale,
                                                self.torch_dtype == torch.bfloat16)

            yield name + '.quant_bias', model_quant_type, quant_bias.cpu().to(torch.int32)
            yield name + '.deq_scale', model_quant_type, deq_scale.cpu()
        if model_quant_type == QuantType.W4A4_FLATQUANT_DYNAMIC:
            if hasattr(module, "save_trans") and module.save_trans is not None:
                save_trans = module.save_trans.get_save_params()
                for key, param in save_trans.items():
                    yield name + "." + key, QuantType.W4A4_FLATQUANT_DYNAMIC, param.cpu()

    def generate_weight_of_tp_module(self, name, module, model_quant_type):
        quant_param, _ = module.get_quant_param()
        for mod_name, mod in module.named_modules(prefix=name):
            if mod_name == name:
                continue

            for param_name, _, param in self.generate_weight_of_module(mod_name, mod):
                quant_param[param_name] = param
            self.skip_module_names.add(mod_name)

        if hasattr(self.cfg, 'tp_size'):
            self.concat_simulate_linear(name, module, quant_param)

        for name, param in quant_param.items():
            yield name, model_quant_type, param

    def generate_weight_of_norm_module(self, name, module, model_quant_type):
        # 不暴露原norm
        self.skip_module_names.add(name + '.module')

        anti_norm_weight: torch.Tensor = module.module.weight
        anti_norm_bias: torch.Tensor = module.bias
        yield name + '.weight', QuantType.FLOAT, anti_norm_weight.clone().cpu()
        yield name + '.bias', QuantType.FLOAT, anti_norm_bias.clone().cpu()
        if not self.is_new_version:
            yield name + '.module.weight', model_quant_type, anti_norm_weight.clone().cpu()
            yield name + '.module.bias', model_quant_type, anti_norm_bias.clone().cpu()

    def generate_weight_of_linear_module(self, name, module, model_quant_type):
        if not module.quant_weight.is_enable:
            return

        quant_weight, fp_weight, weight_scale, weight_offset = self.get_param_from_quantizer(module)
        if quant_weight is None:
            return

        is_new_version_w8a8_pdmix = self.is_new_version and model_quant_type == QuantType.W8A8 and self.cfg.pdmix
        model_quant_type = QuantType.W8A8_MIX if is_new_version_w8a8_pdmix else model_quant_type

        # 各种量化均需要提供 weight
        quant_weight: torch.Tensor = quant_weight.to(device=fp_weight.device)
        save_quant_weight = quant_weight.cpu().to(torch.int8)
        ori_shape = save_quant_weight.shape

        if self.is_new_version and model_quant_type in [QuantType.W4A16]:
            # w4a16 量化当前没走分阶段量化，因此 weight_scale 不是一个 list
            trans_flag = weight_scale.shape[-1] == 1
            save_quant_weight = _w4a16_pack_int4(save_quant_weight, trans_flag)
        
        if self.is_new_version and model_quant_type in [QuantType.W4A8_DYNAMIC]:
            weight = save_quant_weight.reshape(-1, module.cfg.group_size).to(torch.float32)
            second_scale = weight_scale[1].to(torch.bfloat16).reshape(-1, 1)
            first_deq_weight = (weight * second_scale).reshape(ori_shape)
            second_deq_weight = first_deq_weight * weight_scale[0].to(torch.bfloat16)
            bias = second_deq_weight
            scale_bias = process_scale(name, bias, 16)
            save_quant_weight = _w4a8_pack_int4(save_quant_weight)

        yield name + '.weight', model_quant_type, save_quant_weight
        if hasattr(module, 'bias') and module.bias is not None:
            yield name + '.bias', QuantType.FLOAT, module.bias

        # W4A8_DYNAMIC 有两种量化模式，一种分阶段量化（将浮点先量化成int8，然后再量化成 int4），一种直接量化（将浮点直接量化成 int4）
        if model_quant_type in [
            QuantType.W4A8_DYNAMIC
        ]:
            is_scale_list = isinstance(weight_scale, list) and len(weight_scale) == 2
            is_offset_list = isinstance(weight_offset, list) and len(weight_offset) == 2
            if is_scale_list and is_offset_list:
                yield name + '.weight_scale', model_quant_type, weight_scale[0].cpu()
                yield name + '.weight_offset', model_quant_type, weight_offset[0].cpu()

                weight_scale[1] = weight_scale[1].reshape(ori_shape[0], -1)
                weight_offset[1] = weight_offset[1].reshape(ori_shape[0], -1)
                yield name + '.weight_scale_second', model_quant_type, weight_scale[1].cpu()
                yield name + '.weight_offset_second', model_quant_type, weight_offset[1].cpu()
                if self.is_new_version:
                    yield name + '.scale_bias', model_quant_type, scale_bias.cpu()
            else:
                yield name + '.weight_scale', model_quant_type, weight_scale.cpu()
                yield name + '.weight_offset', model_quant_type, weight_offset.cpu()

        # W4A16/W8A16 需要提供 weight_scale、weight_offset
        if model_quant_type in [
            QuantType.W8A16,
            QuantType.W4A16,
            QuantType.W8A8_DYNAMIC,
            QuantType.W8A8,
            QuantType.W8A8_TIMESTEP,
            QuantType.W8A8_MIX
        ]:
            yield name + '.weight_scale', model_quant_type, weight_scale.cpu()
            yield name + '.weight_offset', model_quant_type, weight_offset.cpu()

        # W8A8/W8A8S 需要提供 deq_scale、quant_bias、input_scale、input_offset
        if model_quant_type in [
            QuantType.W8A8,
            QuantType.W8A8S,
            QuantType.W8A8_MIX
        ]:
            input_scale = module.quant_input.input_scale
            input_offset = module.quant_input.input_offset
            yield name + '.input_scale', model_quant_type, input_scale.cpu()
            yield name + '.input_offset', model_quant_type, input_offset.cpu()

            input_scale = input_scale.to(device=quant_weight.device)
            input_offset = input_offset.to(device=quant_weight.device)
            deq_scale = deqscale_process(input_scale, weight_scale).to(torch.float32)
            quant_weight = quant_weight.to(torch.float32)
            input_offset = input_offset.to(torch.float32)
            correction = (quant_weight.sum(dim=1) * input_offset).cpu()
            fp_bias = change_bias(fp_weight, module)
            quant_bias = torch.round(fp_bias / deq_scale - correction)
            deq_scale = deqscale2int64_by_dtype(deq_scale,
                                                self.torch_dtype == torch.bfloat16)

            yield name + '.quant_bias', model_quant_type, quant_bias.cpu().to(torch.int32)
            yield name + '.deq_scale', model_quant_type, deq_scale.cpu()

        if model_quant_type in [
            QuantType.W8A8_TIMESTEP
        ]:
            # fix for fake quantize
            module.quant_weight.int_weight_tensor = quant_weight
            ori_device = quant_weight.device
            tgt_device = 'npu' if torch.cuda.is_available() else 'cpu'
            quant_weight = quant_weight.to(tgt_device)
            weight_scale = weight_scale.to(tgt_device)

            def get_act_quant_param(quant_weight, input_scale, input_offset):
                input_scale = input_scale.to(device=quant_weight.device)
                input_offset = input_offset.to(device=quant_weight.device)
                deq_scale = deqscale_process(input_scale, weight_scale).to(torch.float32)
                quant_weight = quant_weight.to(torch.float32)
                input_offset = input_offset.to(torch.float32)
                correction = (quant_weight.sum(dim=1) * input_offset).cpu()
                fp_bias = change_bias(fp_weight, module)
                quant_bias = torch.round(fp_bias / deq_scale - correction)
                deq_scale = deqscale2int64_by_dtype(deq_scale,
                                                    self.torch_dtype == torch.bfloat16)
                return quant_bias, deq_scale

            input_scale_timestep, input_offset_timestep = module.quant_input.get_timestep_scale_offset_dict()
            quant_bias_timestep = []
            deq_scale_timestep = []
            for input_scale, input_offset in zip(input_scale_timestep, input_offset_timestep):
                quant_bias, deq_scale = get_act_quant_param(quant_weight, input_scale, input_offset)
                quant_bias_timestep.append(quant_bias)
                deq_scale_timestep.append(deq_scale)
            quant_bias_timestep = torch.stack(quant_bias_timestep)
            deq_scale_timestep = torch.stack(deq_scale_timestep)

            quant_weight = quant_weight.to(ori_device)
            weight_scale = weight_scale.to(ori_device)

            yield name + '.input_scale', model_quant_type, input_scale_timestep.cpu()
            yield name + '.input_offset', model_quant_type, input_offset_timestep.cpu()
            yield name + '.quant_bias', model_quant_type, quant_bias_timestep.cpu().to(torch.int32)
            yield name + '.deq_scale', model_quant_type, deq_scale_timestep.cpu()

    def concat_simulate_linear(self, name, module, quant_param):
        if name in self.rollback_names:
            return
        if module.cfg.model_quant_type == QuantType.FLOAT:
            return
        concat_weight_list = []
        for tp_index in range(module.cfg.tp_size):
            concat_name = '.'.join([name, f'tp_list', str(tp_index), 'weight'])
            concat_weight_list.append(quant_param.get(concat_name))
        concat_weight = torch.cat(concat_weight_list, dim=-1)
        quant_param[name + '.weight'] = concat_weight

    @torch.no_grad()
    def get_param_from_quantizer(self, module):
        quant_weight = None
        fp_weight, device, weight_scale, weight_offset, round_opt = _get_module_quant_input(module)
        if isinstance(module, LinearQuantizerTimestep):
            quant_weight, _ = fake_quantize_save(fp_weight, weight_scale, weight_offset, bit=module.cfg.w_bit,
                                                 round_opt=round_opt, device=device)
            module.set_quant_weight(quant_weight)

        if isinstance(module, LinearQuantizer):
            max_bound = -1
            if module.quant_weight.is_sym:
                max_bound = 2**(module.cfg.w_bit - 1) - 1
            quant_weight, _ = fake_quantize_save(fp_weight, weight_scale, weight_offset, bit=module.cfg.w_bit,
                                                 round_opt=round_opt, device=device,
                                                 max_bound=max_bound)
        if isinstance(module, LinearSparseQuantizer):
            _, _, quant_weight, _ = quant_one_weight_by_outliers(
                fp_weight, powerquant=self.cfg.nonuniform, fraction=self.cfg.fraction, num_bits=self.cfg.w_bit,
                per_channel=not self.cfg.mm_tensor)
        if isinstance(module, LowBitLinearQuantizer):
            if not module.cfg.is_stage_quant:
                fp_weight = module.fp_weight
            if module.disable_input:
                res = None, fp_weight, weight_scale, weight_offset
                return res
            if module.cfg.model_quant_type == QuantType.W8A8S:
                bit = 8
            else:
                bit = module.cfg.w_bit

            is_scale_list = isinstance(weight_scale, list) and len(weight_scale) == 2
            is_offset_list = isinstance(weight_offset, list) and len(weight_offset) == 2
            if is_scale_list and is_offset_list:
                # w4a8 分阶段量化，因此需要两次fake_quantize_save得到量化后的权重
                first_quant_weight, _ = fake_quantize_save(fp_weight, weight_scale[0], weight_offset[0], bit=8,
                                                           round_opt=round_opt, device=module.weight.device)
                quant_weight, _ = fake_quantize_save(first_quant_weight, weight_scale[1], weight_offset[1], bit=4,
                                                     round_opt=round_opt, device=module.weight.device,
                                                     group_size=module.cfg.group_size)
            else:
                quant_weight, _ = fake_quantize_save(fp_weight, weight_scale, weight_offset, bit=bit,
                                                     round_opt=round_opt, device=module.weight.device,
                                                     group_size=module.cfg.group_size)
        res = quant_weight, fp_weight, weight_scale, weight_offset
        return res
