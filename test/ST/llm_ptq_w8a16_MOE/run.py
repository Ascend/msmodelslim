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
from transformers import AutoTokenizer, AutoModelForCausalLM
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

fp16_path = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/MOE_tiny/"
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=fp16_path, trust_remote_code=True, local_files_only=True
)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=fp16_path, trust_remote_code=True, local_files_only=True
).cpu()


def get_calib_dataset(fp16_tokenizer, calib_list, device="cpu"):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = fp16_tokenizer(calib_data, return_tensors='pt')
        calib_dataset.append([inputs.data['input_ids'].to(device), inputs.data['attention_mask'].to(device)])
    return calib_dataset


calib_set = [
    "Where is the capital of China?",
    "Please make a poem:",
    "I want to learn python, how should I learn it?",
    "Please help me write a job report on large model inference optimization:",
    "What are the most worth visiting scenic spots in China?",
]
dataset_calib = get_calib_dataset(tokenizer, calib_set, 'cpu')

anti_config = AntiOutlierConfig(anti_method="m3", dev_type='cpu', dev_id=0)
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config, norm_class_name="RMSNorm")
anti_outlier.process()

disable_names = []
disable_names.append('lm_head')
quant_config = QuantConfig(
    a_bit=16,
    w_bit=8,
    disable_names=disable_names,
    dev_type='cpu',
    pr=1.0,
    w_sym=True,
    mm_tensor=False,
)
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()
calibrator.save(f"{os.environ['PROJECT_PATH']}/output/llm_ptq_w8a16_MOE", save_type=["numpy", "safe_tensor"])

print('Save quant weight success!')
