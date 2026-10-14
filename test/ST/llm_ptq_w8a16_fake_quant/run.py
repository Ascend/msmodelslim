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
import json
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import FakeQuantizeCalibrator

# for local path
LOAD_PATH = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/llama2_7b/"
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=LOAD_PATH, torch_dtype=torch.float32, trust_remote_code=True, local_files_only=True
).cpu()
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=LOAD_PATH, trust_remote_code=True, local_files_only=True
)
# 使用load_file()函数读取safetensor格式文件并将其解析为字典
safetensor_dic = load_file(f"{os.environ['PROJECT_PATH']}/resource/llm_ptq_minmax/quant_model_weight_w8a16.safetensors")
# 使用json.load()函数读取文件并将其解析为字典
with open(
    f"{os.environ['PROJECT_PATH']}/resource/llm_ptq_minmax/quant_model_description_w8a16.json", 'r', encoding='utf-8'
) as file:
    description_dic = json.load(file)
fakecalibrator = FakeQuantizeCalibrator(model, None, "cpu", description_dic, safetensor_dic)
model = fakecalibrator.model
print('fake quant weight success!')
