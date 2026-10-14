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
# pylint: disable=no-name-in-module,pointless-string-statement,redefined-outer-name

import torch
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch.utils.data
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_max_memory(max_memory=None, gpu_num=8, unit='MiB'):
    for i in range(gpu_num):
        if i not in max_memory:
            max_memory[i] = 0
    for key in max_memory:
        if isinstance(max_memory[key], (int, float)):
            max_memory[key] = str(max_memory[key]) + unit
    return max_memory


SEQ_LEN_OUT = 12

max_memory = generate_max_memory({"cpu": 457403})
load_path = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/llama2_7b/"
tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True, local_files_only=True, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(
    load_path,
    trust_remote_code=True,
    device_map='auto',
    torch_dtype=torch.float32,
    max_memory=max_memory,
    local_files_only=True,
)
model.eval()

'''
# for 中文模型
calib_list = ["中国的首都在哪里？",
              "请做一首诗歌：",
              "我想要学习python，该怎么学习？",
              "请帮我写一篇关于大模型推理优化的任职报告：",
              "中国最值得去的几个景点"]
'''

calib_list = ["Where is the capital of China?"]


print("testing quantized weights...")
test_prompt = "Common sense questions and answers\n\nQuestion: What is the capital of France\nFactual answer:"
test_input = tokenizer(test_prompt, return_tensors="pt")
print("model is inferring...")
model.eval()
generate_ids = model.generate(
    test_input.input_ids, attention_mask=test_input.attention_mask, max_new_tokens=SEQ_LEN_OUT
)
res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
for idx, item in enumerate(res):
    print(item)


def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt')
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset


dataset_calib = get_calib_dataset(tokenizer, calib_list)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import QuantConfig, Calibrator  # noqa: E402

quant_config = QuantConfig(
    disable_names=[
        'lm_head',
        'model.layers.0.self_attn.q_proj',
        'model.layers.0.self_attn.k_proj',
        'model.layers.0.self_attn.v_proj',
        'model.layers.0.self_attn.o_proj',
        'model.layers.0.mlp.gate_proj',
        'model.layers.0.mlp.up_proj',
        'model.layers.0.mlp.down_proj',
    ],
    do_smooth=False,
    is_lowbit=True,
    use_sigma=False,
)
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib)
calibrator.run()

'''
# quant model
model = calibrator.model

# Notice that you are a compressed low-precision model, please be careful and double check your response.
print("testing quantized weights...")
test_prompt = "Notice that you are a compressed low-precision model, please be careful and double check your response.Common sense questions and answers\n\nQuestion: What is the CEO of Huawei\nFactual answer:"
test_input = tokenizer(test_prompt, return_tensors="pt")
print("model is inferring...")
model.eval()
generate_ids = model.generate(test_input.input_ids, attention_mask=test_input.attention_mask, max_new_tokens=SEQ_LEN_OUT)
res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

for idx, item in enumerate(res):
    print(item)
'''

calibrator.save(f"{os.environ['PROJECT_PATH']}/output/llm_ptq_lowbit", save_type=['numpy', 'safe_tensor'])
print('Save quant weight success!')
