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

import random

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase

from ascend_utils.common.security import check_type, SafeWriteUmask, get_valid_write_path
from ascend_utils.common import hook
from msmodelslim.pytorch.ra_compression.ra_rope_config import RARopeCompressConfig
from msmodelslim import logger

DUMMY_INPUT_LENGTH = 2500
REPET_TIMES = 4
INPUT_IDS = "input_ids"


class RARopeCompressor(object):

    def __init__(self, model, tokenizer, cfg: RARopeCompressConfig):
        check_type(model, nn.Module, param_name="model")
        check_type(cfg, RARopeCompressConfig, param_name="cfg")
        check_type(tokenizer, PreTrainedTokenizerBase, param_name="tokenizer")

        if not hasattr(model, "config"):
            raise ValueError("Model does not have attribute `config`. \
                              Model must be a huggingface model.")
        if not hasattr(model.config, 'hidden_size'):
            raise ValueError("Model must have a `config` attribute with a `hidden_size` property. \
                              Model must be a huggingface model.")
        
        num_attention_head_names = ['num_attention_heads', 'n_head']
        for name in num_attention_head_names:
            if hasattr(model.config, name):
                num_attention_heads = getattr(model.config, name)
        if not num_attention_heads or num_attention_heads == 0:
            raise ValueError("Model must have a `config` attribute with a `num_attention_heads` property. \
                              Model must be a huggingface model.")
        
        num_key_value_heads_names = ['num_key_value_heads', 'multi_query_group_num']
        for name in num_key_value_heads_names:
            if hasattr(model.config, name):
                num_key_value_heads = getattr(model.config, name)
        if not num_key_value_heads or num_key_value_heads == 0:
            raise ValueError("Model must have a `config` attribute with a `num_key_value_heads` property. \
                              Model must be a huggingface model.")
        self.model = model
        self.cfg = cfg
        self.hidden_size = self.model.config.hidden_size
        self.num_key_value_heads = num_key_value_heads
        self.num_attention_heads = num_attention_heads
        self.num_kv_per_group = self.num_attention_heads / self.num_key_value_heads
        self.tokenizer = tokenizer

    def max_every_group(self, data, n):
        result = {}
        for key, values in data.items():
            max_values = [max(values[i:i+n]) for i in range(0, len(values), n)]
            result[key] = max_values
        return result
    
    def remove_empty_list_keys(self, dictionary):
        dictionary = {k: v for k, v in dictionary.items() if v != []}
        return dictionary

    def get_compress_heads(self, save_path):
        check_type(save_path, str, param_name="save_path")

        prefix_matching_score, copying_matching_score = self.get_attention_score()
        prefix_matching_score_result = self.max_every_group(prefix_matching_score, int(self.num_kv_per_group))
        copying_matching_score_result = self.max_every_group(copying_matching_score, int(self.num_kv_per_group))

        selected_heads_prefix = self.select_top_heads(prefix_matching_score_result, self.cfg.induction_head_ratio)
        selected_heads_copying = self.select_top_heads(copying_matching_score_result, self.cfg.echo_head_ratio)

        head_dict = {
            'prefix_matching': self.remove_empty_list_keys(selected_heads_prefix), 
            'copying': self.remove_empty_list_keys(selected_heads_copying)
        }

        with SafeWriteUmask():
            output_model_path = get_valid_write_path(save_path, extensions=".pt")
            torch.save(head_dict, output_model_path)
        logger.info("heads file is stored in %r ", output_model_path)

    def select_top_heads(self, data, ratio):
        # 将所有列表里的值汇总
        all_values = [
            value 
            for key in data 
            for value in data[key]
        ]
        # 对汇总后的值进行排序
        sorted_values = sorted(all_values, reverse=True)
        # 计算前%的索引
        percent_index = round(len(sorted_values) * ratio)
        # 获取前%的值
        percent_values = sorted_values[:percent_index]
        # 创建一个新字典
        result = {}
        for key in data:
            # 获取前%的值在原列表中的索引
            percent_index_in_original_list = [i for i, value in enumerate(data[key]) if value in percent_values]
            result[key] = percent_index_in_original_list
        return result

    def get_attention_score(self):
        softmax_output = SoftmaxDumpOutput(self.num_attention_heads, self.hidden_size)
        rand_tokens = torch.tensor([random.randint(10000, 10000+DUMMY_INPUT_LENGTH) 
                                    for _ in range(DUMMY_INPUT_LENGTH)]*REPET_TIMES)
        model_tokens = self.tokenizer('', return_tensors="pt")
        if not model_tokens[INPUT_IDS].tolist()[0]:
            model_tokens = self.tokenizer('A', return_tensors="pt")
        model_tokens = {key: model_tokens[key] for key in [INPUT_IDS, 'attention_mask']}
        
        model_tokens[INPUT_IDS] = torch.tensor(torch.cat((model_tokens[INPUT_IDS][0][-1].unsqueeze(0), 
                                                          rand_tokens), dim=0)).reshape(1, -1)
        model_tokens['attention_mask'] = torch.ones(1, DUMMY_INPUT_LENGTH*REPET_TIMES + 1)
        for key in model_tokens:
            model_tokens[key] = model_tokens[key].to(self.model.device)
        with torch.no_grad(), hook.FunctionReplace(torch.nn.functional.softmax, softmax_output):
            self.model(**model_tokens)
        return softmax_output.gather_data_prefix, softmax_output.gather_data_copying


class SoftmaxDumpOutput:
    
    def __init__(self, num_attention_heads, hidden_size):
        self.head_num = 0
        self.torch_softmax = torch.nn.functional.softmax
        self.gather_data_prefix = {}
        self.gather_data_copying = {}
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size

    def __call__(self, inputs, **kwargs):
        out = self.torch_softmax(inputs, **kwargs) # torch.Size([64, 10001, 10001])
        if self.num_attention_heads == 0:
            raise ValueError("Num attention heads can not be zero.")
        cur_layer = self.head_num // self.num_attention_heads
        cur_head = self.head_num % self.num_attention_heads
        logger.info(f"The {cur_layer}-th layer {cur_head}-th head attention score has been obtained")
        self.gather_data_prefix.setdefault(cur_layer, []).append(SoftmaxDumpOutput._get_prefix_matching_score(out[0]))
        self.gather_data_copying.setdefault(cur_layer, []).append(SoftmaxDumpOutput._get_copying_matching_score(out[0]))
        self.head_num += 1
        return out
    
    @staticmethod
    def _get_prefix_matching_score(out):
        score = []
        for i, token_attn in enumerate(out[1:, 1:]):
            if i // DUMMY_INPUT_LENGTH == 0:
                continue
            score_token = []
            for j in range(i % DUMMY_INPUT_LENGTH, i, DUMMY_INPUT_LENGTH):
                score_token.append(token_attn[j+1])
            score.append(torch.sum(torch.Tensor(score_token)))
        return torch.mean(torch.Tensor(score))            
    
    @staticmethod
    def _get_copying_matching_score(out):
        score = []
        for i, token_attn in enumerate(out[1:, 1:]):
            if i // DUMMY_INPUT_LENGTH == 0:
                continue
            score_token = []
            for j in range(i % DUMMY_INPUT_LENGTH, i, DUMMY_INPUT_LENGTH):
                score_token.append(token_attn[j])
            score.append(torch.sum(torch.Tensor(score_token)))
        return torch.mean(torch.Tensor(score))