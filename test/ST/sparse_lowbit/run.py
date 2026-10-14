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
# pylint: disable=no-name-in-module

import os
import torch
import torch.utils.data
import torch_npu  # noqa: F401
from transformers import AutoTokenizer, AutoModelForCausalLM


SEQ_LEN_OUT = 12

load_path = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/llama2_7b/"
tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True, local_files_only=True, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(
    load_path, trust_remote_code=True, torch_dtype=torch.float16, local_files_only=True
).npu()


calib_list = [
    "Where is the capital of China?",
    "Please make a poem:",
    "I want to learn python, how should I learn it?",
    "Please help me write a job report on large model inference optimization:",
    "What are the most worth visiting scenic spots in China?",
]


def get_calib_dataset(tokenizers, calib_lists):
    calib_dataset = []
    for calib_data in calib_lists:
        inputs = tokenizers([calib_data], return_tensors='pt')
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'].npu(), inputs.data['attention_mask'].npu()])
    return calib_dataset


dataset_calib = get_calib_dataset(tokenizer, calib_list)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import QuantConfig, Calibrator  # noqa: E402

quant_config = QuantConfig(
    w_bit=4,
    do_smooth=False,
    dev_type='npu',
    is_lowbit=True,
    use_sigma=True,
)
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L2')
calibrator.run()

print("testing quantized weights...")
test_prompt = "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:"
test_input = tokenizer(test_prompt, return_tensors="pt")
print("model is inferring...")
model = model.npu()
model.eval()
generate_ids = model.generate(
    test_input.input_ids.npu(), attention_mask=test_input.attention_mask.npu(), max_new_tokens=SEQ_LEN_OUT
)
res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(res)
for _, item in enumerate(res):
    print(item)


calibrator.save(f"{os.environ['PROJECT_PATH']}/output/llm_ptq_lowbit", save_type=['numpy', 'safe_tensor'])
print('Save quant weight success!')
