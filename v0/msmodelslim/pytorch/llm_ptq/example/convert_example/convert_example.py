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

import json
import os
import stat

import numpy as np
from safetensors.torch import save_file
import torch

from ascend_utils.common.security import get_valid_write_path, SafeWriteUmask, \
                                         get_valid_read_path, get_write_directory, json_safe_dump
from ascend_utils.common.security.pytorch import safe_torch_load


def int4_to_int8_forchatglm(i4w):
    """
    ChatGLM2-6B开源int4量化权重的保存方式，会将两个int4合并成一个int8进行保存，具体实现逻辑见开源代码中quantization.py文件
    此处通过python实现了对应逻辑
    """
    n, k = i4w.shape
    weight = i4w.reshape(-1, 2)
    weight0 = weight[:, :1] << 4
    weight1 = weight[:, 1:] & 0b00001111
    i8w = weight0 | weight1
    i8w = i8w.reshape(n, k // 2).to(torch.int8)
    return i8w


def int8_to_int4_forchatglm(i8w):
    """
    仿照开源将一个int8tensor分离解码成两个int4
    """
    n, k = i8w.shape
    weight = i8w.reshape(-1, 1)
    weight1 = weight & 0b11110000
    weight1 = weight1 >> 4
    weight1 = weight1 - (weight1 > 7).to(torch.int8) * 16
    weight2 = weight & 0b00001111
    weight2 = weight2 - (weight2 > 7).to(torch.int8) * 16
    i4w = torch.concat((weight1, weight2), dim=1).reshape(n, k * 2).to(torch.int8)
    return i4w


# deqscale处理，亲和昇腾量化算子
def deqscale_process(deqscale):
    deqscale = deqscale.numpy()
    deqscale = np.frombuffer(deqscale.tobytes(), dtype=np.int32).astype(np.int64)
    deqscale = torch.tensor(deqscale)
    return deqscale


class MSModelSlimWeightProcessor:
    def __init__(self, weight):
        # 示例量化权重为ChatGLM2-6B int4量化权重
        self.modelslim_weight_dict = weight
        self.modelslim_description_json = {}
        self.num_layers = 28  # 示例量化权重DecoderLayer的层数，请根据实际修改
        self.quant_type = 'W8A16'  # 示例量化权重的量化类型，请根据实际修改
        self.modelslim_description_json['model_quant_type'] = self.quant_type
        for key in self.modelslim_weight_dict.keys():
            self.modelslim_description_json[key] = 'FLOAT'
    
    # 处理示量化权重中Linear层的权重量化参数，对应msmodelslim支持的量化类型w8a16
    def weight_process(self):
        # 获取示例量化权重中Linear层的名称，请根据实际修改
        linear_list = []
        for i in range(self.num_layers):
            linear_list.append(f"transformer.encoder.layers.{i}.self_attention.query_key_value")
            linear_list.append(f"transformer.encoder.layers.{i}.self_attention.dense")
            linear_list.append(f"transformer.encoder.layers.{i}.mlp.dense_h_to_4h")
            linear_list.append(f"transformer.encoder.layers.{i}.mlp.dense_4h_to_h")

        # 获取示例量化权重的名称，处理后增加到权重字典中，请根据实际修改
        for key in linear_list:
            weight_key = '.'.join([key, 'weight'])
            weight_scale_key = '.'.join([key, 'weight_scale'])
            weight_offset_key = '.'.join([key, 'weight_offset'])

            # 示例量化权重的weight的名称和msmodelslim的一致，名称不需要修改
            # 示例量化权重，会将两个int4权重保存成一个int8，此处需要将一个权重解码成两个int4的权重
            self.modelslim_weight_dict[weight_key] = int8_to_int4_forchatglm(self.modelslim_weight_dict[weight_key])
            self.modelslim_description_json[weight_key] = self.quant_type

            # 示例量化权重的weight_scale名称和msmodelslim的一致，名称不需要修改
            # 示例量化权重的weight_scale的shape是[n]，msmodelslim的weight_scale的shape是[n,1]
            self.modelslim_weight_dict[weight_scale_key] = \
                self.modelslim_weight_dict[weight_scale_key].unsqueeze(dim=1).to(torch.float16)
            self.modelslim_description_json[weight_scale_key] = self.quant_type

            # 示例量化权重为对称量化，不存在weight_offset，msmodelslim工具对称量化会生成全0的tensor
            self.modelslim_weight_dict[weight_offset_key] = \
                torch.zeros(self.modelslim_weight_dict[weight_scale_key].shape).to(torch.float16)
            self.modelslim_description_json[weight_offset_key] = self.quant_type

    # 处理示量化权重中Linear层的权重量化参数，对应msmodelslim支持的量化类型W8A8和W8A8S
    def weight_activation_process(self):
        # 获取开源量化权重中Linear层的名称，请根据实际修改
        
        linear_list = []
        for i in range(self.num_layers):
            linear_list.append(f"transformer.encoder.layers.{i}.self_attention.query_key_value")
            linear_list.append(f"transformer.encoder.layers.{i}.self_attention.dense")
            linear_list.append(f"transformer.encoder.layers.{i}.mlp.dense_h_to_4h")
            linear_list.append(f"transformer.encoder.layers.{i}.mlp.dense_4h_to_h")

        # 获取示例量化权重的名称，处理后增加到权重字典中，请根据实际修改
        for key in linear_list:
            # 获取开源量化权重中，量化参数的名称，请根据实际修改
            ori_weight_key = '.'.join([key, 'weight'])
            ori_input_scale_key = '.'.join([key, 'input_scale'])
            ori_input_offset_key = '.'.join([key, 'input_offset'])
            ori_deq_scale_key = '.'.join([key, '.deq_scale'])
            ori_quant_bias_key = '.'.join([key, 'quant_bias'])

            # msmodelslim生成的量化权重中kv caceh对应的scale和offset的权重名称
            msmodelslim_weight_key = '.'.join([key, 'weight'])
            msmodelslim_input_scale_key = '.'.join([key, 'input_scale'])
            msmodelslim_input_offset_key = '.'.join([key, 'input_offset'])
            msmodelslim_deq_scale_key = '.'.join([key, '.deq_scale'])
            msmodelslim_quant_bias_key = '.'.join([key, 'quant_bias'])

            self.modelslim_weight_dict[msmodelslim_weight_key] = self.modelslim_weight_dict[ori_weight_key]
            self.modelslim_weight_dict[msmodelslim_input_scale_key] = self.modelslim_weight_dict[ori_input_scale_key]
            self.modelslim_weight_dict[msmodelslim_input_offset_key] = self.modelslim_weight_dict[ori_input_offset_key]
            # 若开源量化基于fp16量化，则deqscale需要使用deqscale_process进行数据类型转换，若为bf16则不需要
            self.modelslim_weight_dict[msmodelslim_deq_scale_key] = \
                deqscale_process(self.modelslim_weight_dict[ori_deq_scale_key])
            self.modelslim_weight_dict[msmodelslim_quant_bias_key] = self.modelslim_weight_dict[ori_quant_bias_key]
        
            # 删除modelslim_weight_dict中已被改名的重复权重，请根据实际修改
            del (self.modelslim_weight_dict[ori_weight_key],
                 self.modelslim_weight_dict[ori_input_scale_key],
                 self.modelslim_weight_dict[ori_input_offset_key],
                 self.modelslim_weight_dict[ori_deq_scale_key],
                 self.modelslim_weight_dict[ori_quant_bias_key])
            del (self.modelslim_description_json[ori_weight_key],
                 self.modelslim_description_json[ori_input_scale_key],
                 self.modelslim_description_json[ori_input_offset_key],
                 self.modelslim_description_json[ori_deq_scale_key],
                 self.modelslim_description_json[ori_quant_bias_key])
            self.modelslim_description_json[msmodelslim_weight_key] = self.quant_type
            self.modelslim_description_json[msmodelslim_input_scale_key] = self.quant_type
            self.modelslim_description_json[msmodelslim_input_offset_key] = self.quant_type
            self.modelslim_description_json[msmodelslim_deq_scale_key] = self.quant_type
            self.modelslim_description_json[msmodelslim_quant_bias_key] = self.quant_type

    # 对于smooth quant算法，norm层权重更换示例
    def anti_outlier_process(self):
        # 获取开源量化权重中，norm层的名称，请根据实际修改
        norm_list = []
        for i in range(self.num_layers):
            norm_list.append(f"transformer.encoder.layers.{i}.input_layernorm")
            norm_list.append(f"transformer.encoder.layers.{i}.post_attention_layernorm")

        for key in norm_list:
            # 此处norm_weight_key和norm_bias_key值的是开源量化权重中scale后的norm weight和norm bias，请根据实际量化权重进行修改
            ori_norm_weight_key = '.'.join([key, 'weight'])
            ori_norm_bias_key = '.'.join([key, 'bias'])
            # msmodelslim生成的量化权重中，scale后的norm weight和norm bias对应norm.module.weight和norm.module.bias
            msmodelslim_norm_module_weight_key = '.'.join([key, 'module.weight'])
            msmodelslim_norm_module_bias_key = '.'.join([key, 'module.bias'])
            # msmodelslim生成的量化权重中norm.weight指的是scale前的norm weight
            msmodelslim_norm_weight_key = '.'.join([key, 'weight'])
            self.modelslim_weight_dict[msmodelslim_norm_module_weight_key] = \
                self.modelslim_weight_dict[ori_norm_weight_key]
            self.modelslim_weight_dict[msmodelslim_norm_module_bias_key] = \
                self.modelslim_weight_dict[ori_norm_bias_key]
            
            # 删除modelslim_weight_dict中已被改名的重复权重，并修改描述文件，请根据实际修改
            del self.modelslim_weight_dict[ori_norm_weight_key], self.modelslim_weight_dict[ori_norm_bias_key]
            del self.modelslim_description_json[ori_norm_weight_key], self.modelslim_description_json[ori_norm_bias_key]
            self.modelslim_description_json[msmodelslim_norm_module_weight_key] = self.quant_type
            self.modelslim_description_json[msmodelslim_norm_module_bias_key] = self.quant_type

            # 若开源权重不涉及回退操作，此处传入None即可
            self.modelslim_weight_dict[msmodelslim_norm_weight_key] = None
            self.modelslim_description_json[msmodelslim_norm_module_bias_key] = 'FLOAT'

    # kv cache量化
    def kv_cache_process(self):
        # 在描述文件中添加kv_cache量化的说明
        self.modelslim_description_json["kv_cache_type"] = "C8"
        # 获取开源量化权重中，k_proj和v_proj层的名称
        kv_linear_list = []
        for i in range(self.num_layers):
            kv_linear_list.append(f"transformer.encoder.layers.{i}.self_attention.query_key_value")
        for key in kv_linear_list:
            # 获取开源量化权重kv cache的scale和offset名称，请根据实际量化权重进行修改
            ori_kvcache_scale_key = '.'.join([key, 'kv_cacahe_scale'])
            ori_kvcache_offset_key = '.'.join([key, 'kv_cache_offset'])
            # msmodelslim生成的量化权重中kv caceh对应的scale和offset的权重名称
            msmodelslim_kvcaceh_scale_key = '.'.join([key, 'kv_cacahe_scale'])
            msmodelslim_kvcaceh_offset_key = '.'.join([key, 'kv_cache_offset'])

            self.modelslim_weight_dict[msmodelslim_kvcaceh_scale_key] = \
                self.modelslim_weight_dict[ori_kvcache_scale_key]
            self.modelslim_weight_dict[msmodelslim_kvcaceh_offset_key] = \
                self.modelslim_weight_dict[ori_kvcache_offset_key]
            del (self.modelslim_weight_dict[ori_kvcache_scale_key], 
                 self.modelslim_weight_dict[ori_kvcache_offset_key])
            del (self.modelslim_description_json[ori_kvcache_scale_key], 
                 self.modelslim_description_json[ori_kvcache_offset_key])

            self.modelslim_description_json[msmodelslim_kvcaceh_scale_key] = self.quant_type
            self.modelslim_description_json[msmodelslim_kvcaceh_offset_key] = self.quant_type

    # 储存量化权重和描述文件
    def save(self, path):
        path = get_write_directory(path)
        safetensors_name = f"quant_model_weight_{self.quant_type.lower()}.safetensors"
        json_name = f"quant_model_description_{self.quant_type.lower()}.json"
        safetensors_path = os.path.join(path, safetensors_name)
        json_path = os.path.join(path, json_name)
        safetensors_path = get_valid_write_path(safetensors_path, is_dir=False)
        json_path = get_valid_write_path(json_path, is_dir=False)
        '''
        # 如果原始权重中存在内存复用，即safetensors中的tensor share memory场景，此处提供了一种简单的处理方法
        for key, value in self.modelslim_weight_dict.items():
            self.modelslim_weight_dict[key] = value.cpu().contiguous().clone()
        '''
        with SafeWriteUmask(umask=0o377):
            save_file(self.modelslim_weight_dict, safetensors_path)
        
        json_safe_dump(self.modelslim_description_json, json_path, indent=2)


if __name__ == '__main__':
    ori_weight_path = './example_model_weight/pytorch_model.bin'  # 输入开源量化权重path
    out_path = './msmodelslim_weight'  # 输出路径
    
    ori_weight_path = get_valid_read_path(ori_weight_path)
    ori_weight_dict = safe_torch_load(ori_weight_path, map_location='cpu')  # 加载开源权重，请根据实际权重类型进行修改
    # 处理开源权重
    ms = MSModelSlimWeightProcessor(ori_weight_dict)
    # 示例权重只涉及linear层量化权重的处理，如果涉及kvcahe量化或smooth quant，请修改Processor中对应的函数并调用
    ms.weight_process()
    # 保存safetensors权重和json描述文件
    ms.save(out_path)

