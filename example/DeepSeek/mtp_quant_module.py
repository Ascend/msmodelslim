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
import types
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from add_safetensors import find_file_with_pattern, get_weight_map
from example.common.security.path import json_safe_load, json_safe_dump
from example.common.security.path import get_valid_read_path


READ_ONLY_PERMISSION = 0o400
READ_WRITE_PERMISSION = 0o600
MAX_READ_FILE_SIZE_16G = 17179869184  # 16G, 16 * 1024 * 1024 * 1024


def remove_zero_and_shift(matrix):
    n, m = matrix.shape

    # Step 1: 找到每行第一个 0 的位置（即要删除的位置）
    # 如果某行没有 0，则默认保留所有元素（但根据题意，应该每行都有一个 0）
    zero_pos = (matrix == 0).int().argmax(dim=1)  # [n,]

    # Step 2: 构造掩码，标记要保留的元素（排除每行的第一个 0）
    # 生成一个 [n, m] 的坐标矩阵，标记每列是否等于 zero_pos
    col_indices = torch.arange(m, device=matrix.device).expand(n, -1)  # [n, m]
    mask = col_indices != zero_pos.unsqueeze(1)  # [n, m]

    # Step 3: 用掩码筛选元素（自动展平，需要重新调整形状）
    filtered = matrix[mask].view(n, m - 1)  # [n, m-1]

    # Step 4: 在最后一列补 0
    result = torch.cat([filtered, torch.zeros(n, 1, device=matrix.device)], dim=1)  # [n, m]

    return result.to(matrix)


def get_mtp_position_ids(seq_len: int, device: torch.device) -> torch.Tensor:
    """Roll main-model position ids [0..T-1] into [1..T-1, 0] for MTP RoPE."""
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
    position_ids = torch.roll(position_ids, shifts=-1, dims=-1)
    return position_ids.unsqueeze(0)


def custom_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    past_key_values_length = 0
    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if self._use_flash_attention_2:
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


class SharedHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states):
        normalized_states = self.norm(hidden_states)
        logits = self.head(normalized_states)
        return logits


class DeepseekV3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MTPLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.shared_head = SharedHead(config)

        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)


class MTPModel(PreTrainedModel):
    def __init__(self, config, model, mtp_layer):
        super().__init__(config)
        self.model = model.model
        self.vocab_size = config.vocab_size
        self.lm_head = model.lm_head
        self.post_init()

        self.mtp_decoder = model.model.layers[-1]
        self.mtp_layer = mtp_layer.to(model.device)
        self.model.layers = self.model.layers[:-1]
        self.model.config.num_hidden_layers = self.model.config.num_hidden_layers - 1
        self.config.num_hidden_layers = self.config.num_hidden_layers - 1
        self.model.forward = types.MethodType(custom_model_forward, self.model)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pre_hidden_states = outputs[0]
        hidden_states = self.model.norm(pre_hidden_states)
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        ####################### MTP LAYER ######################
        input_ids_mtp = remove_zero_and_shift(input_ids)
        position_ids = get_mtp_position_ids(input_ids_mtp.shape[-1], input_ids.device)
        input_ids_mtp[:, -1] = logits[:, -1, :].argmax(dim=1)

        input_embeds_mtp = self.mtp_layer.embed_tokens(input_ids_mtp)
        input_embeds_mtp = self.mtp_layer.enorm(input_embeds_mtp)
        hidden_states_mtp = self.mtp_layer.hnorm(pre_hidden_states)
        hidden_states_mtp = torch.cat([input_embeds_mtp, hidden_states_mtp], dim=-1)
        hidden_states_mtp = self.mtp_layer.eh_proj(hidden_states_mtp)

        attention_mask_mtp = _prepare_4d_causal_attention_mask(
            attention_mask,
            (input_ids.shape[:2]),
            input_embeds_mtp,
            0,
        )
        layer_outputs_mtp = self.mtp_decoder(
            hidden_states_mtp,
            attention_mask=attention_mask_mtp,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        self.mtp_layer.shared_head(layer_outputs_mtp[0])
        ####################### MTP LAYER ######################

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def warp_mtp_model(config, base_model, model_path):
    """
    从预训练模型加载MTP层权重，并封装为MTP_Model

    Args:
        config (Config): 模型配置对象
        base_model (PreTrainedModel): 原始基础模型
        model_path (str): 预训练模型路径
        mtp_layer_class (nn.Module): MTP层类（默认: MTP_Layer）

    Returns:
        MTP_Model: 封装后的MTP模型
    """
    mtp_layer = MTPLayer(config)
    mtp_safetensor = os.path.join(model_path, "model-00163-of-000163.safetensors")
    mtp_safetensor = get_valid_read_path(
        mtp_safetensor, size_max=MAX_READ_FILE_SIZE_16G, is_dir=False, check_user_stat=True
    )
    mtp_weight = load_file(mtp_safetensor)
    new_state_dict = {}
    for key, value in mtp_weight.items():
        new_key = key.replace('model.layers.61.', '')
        if new_key in mtp_layer.state_dict().keys():
            new_state_dict[new_key] = value
    mtp_layer.load_state_dict(new_state_dict)
    warpped_model = MTPModel(config, base_model, mtp_layer)
    return warpped_model


def post_process_mtp_quant(model_path):
    safetensor_to_check = set()
    # 加载description json
    desc_file = find_file_with_pattern(model_path, "quant_model_description*.json")
    description_data = json_safe_load(desc_file)
    new_state_dict = {}
    for key, value in description_data.items():
        new_key = key.replace('mtp_decoder', 'model.layers.61').replace('mtp_layer', 'model.layers.61')
        new_state_dict[new_key] = value
    json_safe_dump(new_state_dict, desc_file, indent=4)

    # 加载index json
    index_file = find_file_with_pattern(model_path, "quant_model_weight_*.index.json")
    index_data = json_safe_load(index_file)
    map_data = get_weight_map(index_file)
    new_state_dict = {}
    for key, value in map_data.items():
        new_key = key.replace('mtp_decoder', 'model.layers.61').replace('mtp_layer', 'model.layers.61')
        if 'model.layers.61' in new_key:
            safetensor_to_check.add(value)
        new_state_dict[new_key] = value
    index_data["weight_map"] = new_state_dict
    json_safe_dump(index_data, index_file, indent=4)

    # 加载path下所有safetensor
    for tensor_file in safetensor_to_check:
        tensor_path = os.path.join(model_path, tensor_file)
        with safe_open(tensor_path, framework="pt") as f:
            tensors = {}
            for name in f.keys():
                new_name = name.replace('mtp_decoder', 'model.layers.61').replace('mtp_layer', 'model.layers.61')
                tensors[new_name] = f.get_tensor(name)
        os.chmod(tensor_path, READ_WRITE_PERMISSION, follow_symlinks=False)
        save_file(tensors, tensor_path)
        os.chmod(tensor_path, READ_ONLY_PERMISSION, follow_symlinks=False)
